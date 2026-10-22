"""
Improved U-Net pipeline for old-photo restoration.

Colab setup cell (run once before this script):
    !pip install -q datasets hf_xet scikit-image tqdm

Main improvements over the original version:
- Correct and strict RGB conversion for the real dataset keys.
- Reproducible train/validation split.
- Mixed-precision training on CUDA for better speed and lower memory use.
- Learning-rate scheduler, early stopping, and best-checkpoint saving.
- Device-safe checkpoint loading.
- CSV export for training history and per-image/summary metrics.
- PSNR, SSIM, MSE, and MAE evaluation.
- Ten representative output comparisons by default, plus a combined gallery.
- High-resolution saved figures and organized output directories.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_score
from skimage.metrics import structural_similarity as ssim_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# CONFIGURATION
@dataclass(frozen=True)
class Config:
    dataset_name: str = "joshuachin/openphoto-restore-dataset"
    image_size: int = 256
    batch_size: int = 16
    validation_fraction: float = 0.10
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-6
    num_workers: int = 2
    seed: int = 42
    num_visuals: int = 10  # More than the required five examples
    base_features: Tuple[int, ...] = (64, 128, 256, 512)
    output_dir: str = "CV_Final_Project_outputs"


CFG = Config()


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fixed-size convolutions benefit from this speed optimization.
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"

OUTPUT_DIR = Path(CFG.output_dir)
FIGURES_DIR = OUTPUT_DIR / "figures"
SAMPLES_DIR = FIGURES_DIR / "sample_results"
METRICS_DIR = OUTPUT_DIR / "metrics"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for directory in (OUTPUT_DIR, FIGURES_DIR, SAMPLES_DIR, METRICS_DIR, CHECKPOINT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet_restore.pth"
LOSS_CURVE_PATH = FIGURES_DIR / "loss_profile_curve.png"
SAMPLE_GALLERY_PATH = FIGURES_DIR / "sample_results_gallery.png"
TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.csv"
PER_IMAGE_METRICS_PATH = METRICS_DIR / "test_metrics_per_image.csv"
SUMMARY_METRICS_PATH = METRICS_DIR / "test_metrics_summary.csv"
CONFIG_PATH = OUTPUT_DIR / "run_config.json"


# PHASE 1: DATA PREPARATION & PREPROCESSING
class PhotoRestoreDataset(Dataset):
    """PyTorch wrapper for paired damaged/pristine Hugging Face images."""

    def __init__(self, hf_dataset: Any, transform: transforms.Compose) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]

        # Exact keys confirmed for joshuachin/openphoto-restore-dataset.
        damaged_img = item["damaged_image"]
        pristine_img = item["pristine_image"]

        if not isinstance(damaged_img, Image.Image):
            raise TypeError(f"damaged_image at index {idx} is not a PIL image.")
        if not isinstance(pristine_img, Image.Image):
            raise TypeError(f"pristine_image at index {idx} is not a PIL image.")

        # Always force three-channel RGB input/target pairs.
        damaged_img = damaged_img.convert("RGB")
        pristine_img = pristine_img.convert("RGB")

        damaged_tensor = self.transform(damaged_img)
        pristine_tensor = self.transform(pristine_img)

        return damaged_tensor, pristine_tensor


def get_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load data, create a reproducible split, preprocess, and build loaders."""

    print("Loading dataset from Hugging Face...")
    raw_data = load_dataset(config.dataset_name)

    # The assignment permits reducing images to 256x256. ToTensor maps [0,255] -> [0,1].
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.image_size, config.image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    # Reproducible 90/10 train-validation split from the original train split.
    split = raw_data["train"].train_test_split(
        test_size=config.validation_fraction,
        seed=config.seed,
        shuffle=True,
    )

    train_dataset = PhotoRestoreDataset(split["train"], transform)
    val_dataset = PhotoRestoreDataset(split["test"], transform)
    test_dataset = PhotoRestoreDataset(raw_data["test"], transform)

    loader_generator = torch.Generator().manual_seed(config.seed)
    persistent_workers = config.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
    )

    # Batch size 1 keeps per-image metrics and visual selection simple and exact.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
    )

    print(
        "Dataset split complete: "
        f"Train={len(train_dataset)} | "
        f"Validation={len(val_dataset)} | "
        f"Test={len(test_dataset)}"
    )

    # Sanity-check one batch before model training.
    damaged_batch, pristine_batch = next(iter(train_loader))
    expected_shape = (
        damaged_batch.shape[0],
        3,
        config.image_size,
        config.image_size,
    )
    assert damaged_batch.shape == expected_shape
    assert pristine_batch.shape == expected_shape
    assert 0.0 <= damaged_batch.min().item() <= damaged_batch.max().item() <= 1.0
    assert 0.0 <= pristine_batch.min().item() <= pristine_batch.max().item() <= 1.0

    print(f"Sanity check passed. Batch shape: {tuple(damaged_batch.shape)}")
    return train_loader, val_loader, test_loader


# PHASE 2: MODEL ARCHITECTURE (U-NET)
class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net for supervised RGB image restoration with skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_channels, feature))
            current_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections.reverse()

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Robust correction for any spatial-size mismatch.
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.output_activation(self.final_conv(x))


def validate_model_shape(model: nn.Module, config: Config, device: torch.device) -> None:
    """Confirm that U-Net preserves input shape and output range."""
    model.eval()
    dummy = torch.rand(1, 3, config.image_size, config.image_size, device=device)
    with torch.inference_mode():
        output = model(dummy)

    assert output.shape == dummy.shape, (
        f"Model shape mismatch: input={dummy.shape}, output={output.shape}"
    )
    assert output.min().item() >= 0.0 and output.max().item() <= 1.0
    print(f"Model check passed. Input/output shape: {tuple(output.shape)}")


# PHASE 3: TRAINING
def create_grad_scaler(enabled: bool):
    """Create a GradScaler"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_training_history(history: List[Dict[str, float]], path: Path) -> None:
    fieldnames = ["epoch", "train_l1", "val_l1", "learning_rate"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> List[Dict[str, float]]:
    """Train U-Net with L1 loss, Adam, AMP, scheduling, and early stopping."""

    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    scaler = create_grad_scaler(USE_AMP)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_start = time.time()

    print(f"\nTraining on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {USE_AMP}")

    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs} [Train]",
            leave=False,
        )

        for damaged, pristine in train_bar:
            damaged = damaged.to(device, non_blocking=True)
            pristine = pristine.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                restored = model(damaged)
                loss = criterion(restored, pristine)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * damaged.size(0)
            train_bar.set_postfix(l1=f"{loss.item():.5f}")

        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.inference_mode():
            for damaged, pristine in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{config.epochs} [Validation]",
                leave=False,
            ):
                damaged = damaged.to(device, non_blocking=True)
                pristine = pristine.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=USE_AMP,
                ):
                    restored = model(damaged)
                    val_loss_batch = criterion(restored, pristine)

                running_val_loss += val_loss_batch.item() * damaged.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "val_l1": val_loss,
                "learning_rate": current_lr,
            }
        )
        save_training_history(history, TRAINING_HISTORY_PATH)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"Train L1={train_loss:.6f} | "
            f"Val L1={val_loss:.6f} | "
            f"LR={current_lr:.2e}"
        )

        # Save the best validation checkpoint, not merely the final epoch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": asdict(config),
                },
                BEST_MODEL_PATH,
            )
            print(f"  ✓ New best model saved to: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(
                "  Validation did not improve: "
                f"{epochs_without_improvement}/{config.early_stopping_patience}"
            )

        scheduler.step(val_loss)

        if epochs_without_improvement >= config.early_stopping_patience:
            print("Early stopping activated.")
            break

    elapsed_minutes = (time.time() - training_start) / 60.0
    print(f"Training completed in {elapsed_minutes:.2f} minutes.")
    return history


def plot_loss_curves(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_l1"] for row in history]
    val_losses = [row["val_l1"] for row in history]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_losses, marker="o", label="Training L1 Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation L1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("U-Net Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Loss curve saved to: {output_path}")


# PHASE 4 & 5: EVALUATION & VISUALIZATION
def load_best_weights(model: nn.Module, path: Path, device: torch.device) -> Dict[str, Any]:
    """Load either the improved checkpoint dict or the original plain state_dict."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = checkpoint
    else:
        # Compatibility with the original code's plain state_dict checkpoint.
        model.load_state_dict(checkpoint)
        metadata = {}

    model.to(device)
    model.eval()
    return metadata


def save_single_comparison(
    damaged_np: np.ndarray,
    pristine_np: np.ndarray,
    restored_np: np.ndarray,
    psnr_value: float,
    ssim_value: float,
    sample_number: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(damaged_np)
    axes[0].set_title("Input (Damaged)")
    axes[0].axis("off")

    axes[1].imshow(pristine_np)
    axes[1].set_title("Ground Truth (Pristine)")
    axes[1].axis("off")

    axes[2].imshow(restored_np)
    axes[2].set_title(
        f"Restored\nPSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}"
    )
    axes[2].axis("off")

    fig.suptitle(f"Test Sample {sample_number}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_gallery(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save all selected samples in one large report-ready figure."""
    if not samples:
        return

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.1 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["damaged"])
        axes[row, 0].set_title(f"Sample {sample['sample_number']} — Damaged")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(sample["pristine"])
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(sample["restored"])
        axes[row, 2].set_title(
            f"Restored | PSNR {sample['psnr']:.2f} dB | "
            f"SSIM {sample['ssim']:.4f}"
        )
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(
    rows: List[Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    per_image_fields = ["image_index", "psnr_db", "ssim", "mse", "mae"]
    with PER_IMAGE_METRICS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_image_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [
        {
            "metric": "PSNR",
            "mean": summary["mean_psnr"],
            "standard_deviation": summary["std_psnr"],
            "required_minimum": 20.0,
            "status": "PASS" if summary["mean_psnr"] >= 20.0 else "BELOW TARGET",
        },
        {
            "metric": "SSIM",
            "mean": summary["mean_ssim"],
            "standard_deviation": summary["std_ssim"],
            "required_minimum": 0.65,
            "status": "PASS" if summary["mean_ssim"] >= 0.65 else "BELOW TARGET",
        },
        {
            "metric": "MSE",
            "mean": summary["mean_mse"],
            "standard_deviation": summary["std_mse"],
            "required_minimum": "",
            "status": "Lower is better",
        },
        {
            "metric": "MAE",
            "mean": summary["mean_mae"],
            "standard_deviation": summary["std_mae"],
            "required_minimum": "",
            "status": "Lower is better",
        },
    ]

    with SUMMARY_METRICS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "mean",
                "standard_deviation",
                "required_minimum",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_and_plot(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> Dict[str, float]:
    """Evaluate the best checkpoint and save metrics plus 10 representative examples."""

    metadata = load_best_weights(model, BEST_MODEL_PATH, device)
    if metadata:
        print(
            f"Loaded best checkpoint from epoch {metadata.get('epoch', 'unknown')} "
            f"with validation L1={metadata.get('best_val_loss', float('nan')):.6f}"
        )

    test_count = len(test_loader.dataset)
    visual_count = min(config.num_visuals, test_count)

    # Select examples evenly across the test set instead of only using its first items.
    selected_indices = set(
        np.linspace(0, test_count - 1, num=visual_count, dtype=int).tolist()
    )

    metric_rows: List[Dict[str, float]] = []
    selected_samples: List[Dict[str, Any]] = []

    print("\nEvaluating model on the full test split...")

    with torch.inference_mode():
        for idx, (damaged, pristine) in enumerate(
            tqdm(test_loader, desc="Test evaluation")
        ):
            damaged_device = damaged.to(device, non_blocking=True)
            pristine_device = pristine.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                restored_device = model(damaged_device)

            restored_device = restored_device.clamp(0.0, 1.0)

            # Compute MSE/MAE in tensor form.
            mse_value = torch.mean((restored_device - pristine_device) ** 2).item()
            mae_value = torch.mean(torch.abs(restored_device - pristine_device)).item()

            damaged_np = (
                damaged[0].permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            )
            pristine_np = (
                pristine[0].permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            )
            restored_np = (
                restored_device[0]
                .detach()
                .float()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .clip(0.0, 1.0)
            )

            psnr_value = float(
                psnr_score(pristine_np, restored_np, data_range=1.0)
            )
            ssim_value = float(
                ssim_score(
                    pristine_np,
                    restored_np,
                    data_range=1.0,
                    channel_axis=-1,
                )
            )

            metric_rows.append(
                {
                    "image_index": idx + 1,
                    "psnr_db": psnr_value,
                    "ssim": ssim_value,
                    "mse": mse_value,
                    "mae": mae_value,
                }
            )

            if idx in selected_indices:
                sample_number = len(selected_samples) + 1
                sample_path = SAMPLES_DIR / f"sample_result_{sample_number:02d}.png"

                save_single_comparison(
                    damaged_np=damaged_np,
                    pristine_np=pristine_np,
                    restored_np=restored_np,
                    psnr_value=psnr_value,
                    ssim_value=ssim_value,
                    sample_number=sample_number,
                    output_path=sample_path,
                )

                selected_samples.append(
                    {
                        "sample_number": sample_number,
                        "dataset_index": idx + 1,
                        "damaged": damaged_np,
                        "pristine": pristine_np,
                        "restored": restored_np,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                    }
                )

    psnr_values = np.array([row["psnr_db"] for row in metric_rows], dtype=float)
    ssim_values = np.array([row["ssim"] for row in metric_rows], dtype=float)
    mse_values = np.array([row["mse"] for row in metric_rows], dtype=float)
    mae_values = np.array([row["mae"] for row in metric_rows], dtype=float)

    summary = {
        "mean_psnr": float(psnr_values.mean()),
        "std_psnr": float(psnr_values.std()),
        "min_psnr": float(psnr_values.min()),
        "max_psnr": float(psnr_values.max()),
        "mean_ssim": float(ssim_values.mean()),
        "std_ssim": float(ssim_values.std()),
        "min_ssim": float(ssim_values.min()),
        "max_ssim": float(ssim_values.max()),
        "mean_mse": float(mse_values.mean()),
        "std_mse": float(mse_values.std()),
        "mean_mae": float(mae_values.mean()),
        "std_mae": float(mae_values.std()),
    }

    write_metrics_csv(metric_rows, summary)
    save_combined_gallery(selected_samples, SAMPLE_GALLERY_PATH)

    print("\nFinal test metrics")
    print("-" * 76)
    print("Metric      Mean            Std. Dev.       Required        Status")
    print("-" * 76)
    print(
        f"PSNR        {summary['mean_psnr']:.4f} dB      "
        f"{summary['std_psnr']:.4f}          >= 20 dB       "
        f"{'PASS' if summary['mean_psnr'] >= 20.0 else 'BELOW TARGET'}"
    )
    print(
        f"SSIM        {summary['mean_ssim']:.6f}        "
        f"{summary['std_ssim']:.6f}        >= 0.65        "
        f"{'PASS' if summary['mean_ssim'] >= 0.65 else 'BELOW TARGET'}"
    )
    print(
        f"MSE         {summary['mean_mse']:.8f}     "
        f"{summary['std_mse']:.8f}     lower is better"
    )
    print(
        f"MAE         {summary['mean_mae']:.8f}     "
        f"{summary['std_mae']:.8f}     lower is better"
    )
    print("-" * 76)
    print(
        f"PSNR range: {summary['min_psnr']:.2f} to {summary['max_psnr']:.2f} dB"
    )
    print(
        f"SSIM range: {summary['min_ssim']:.4f} to {summary['max_ssim']:.4f}"
    )
    print(f"Saved {len(selected_samples)} separate comparison figures to: {SAMPLES_DIR}")
    print(f"Saved combined comparison gallery to: {SAMPLE_GALLERY_PATH}")
    print(f"Saved per-image metrics to: {PER_IMAGE_METRICS_PATH}")
    print(f"Saved summary metrics to: {SUMMARY_METRICS_PATH}")

    return summary


# MAIN
def main() -> None:
    print("=" * 70)
    print("OLD PHOTO RESTORATION WITH U-NET")
    print("=" * 70)
    print(f"Using compute device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA is unavailable; training on CPU will be slow.")

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(asdict(CFG), file, indent=2)

    # 1. Data preparation
    train_loader, val_loader, test_loader = get_dataloaders(CFG)

    # 2. Model construction
    model = UNet(features=CFG.base_features).to(DEVICE)
    validate_model_shape(model, CFG, DEVICE)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # 3. Training
    history = train_model(model, train_loader, val_loader, DEVICE, CFG)
    plot_loss_curves(history, LOSS_CURVE_PATH)

    # 4. Evaluation and more-than-five visual outputs
    evaluate_and_plot(model, test_loader, DEVICE, CFG)

    print("\nAll required outputs were generated successfully.")
    print(f"Project output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
