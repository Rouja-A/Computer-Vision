# ============================================================
# BONUS SECTION: ENGLISH GRADIO USER INTERFACE
# OLD PHOTO RESTORATION USING THE TRAINED U-NET
# ============================================================

!pip install -q gradio

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# 1. U-NET ARCHITECTURE
# This architecture must exactly match the trained model.
class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks:

    Convolution -> Batch Normalization -> ReLU
    Convolution -> Batch Normalization -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for supervised RGB image restoration.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: Sequence[int] = (64, 128, 256, 512)
    ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels

        # Encoder path
        for feature in features:
            self.downs.append(
                DoubleConv(
                    in_channels=current_channels,
                    out_channels=feature
                )
            )
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1] * 2
        )

        # Decoder path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )

            self.ups.append(
                DoubleConv(
                    in_channels=feature * 2,
                    out_channels=feature
                )
            )

        # Convert final feature maps to a three-channel RGB image
        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )

        # Keep output pixel values in the range [0, 1]
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        # Encoder
        for down_block in self.downs:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the encoder features for decoder skip connections
        skip_connections.reverse()

        # Decoder
        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)

            skip_feature = skip_connections[index // 2]

            # Correct any spatial-size mismatch
            if x.shape[-2:] != skip_feature.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Combine encoder and decoder features
            x = torch.cat(
                (skip_feature, x),
                dim=1
            )

            x = self.ups[index + 1](x)

        restored_image = self.final_conv(x)

        return self.output_activation(restored_image)


# 2. DEVICE SELECTION
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Inference device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Inference will run on the CPU.")


# 3. FIND THE TRAINED CHECKPOINT
possible_checkpoint_paths = [
    Path(
        "/content/CV_Final_Project_outputs/"
        "checkpoints/best_unet_restore.pth"
    ),

    Path(
        "CV_Final_Project_outputs/"
        "checkpoints/best_unet_restore.pth"
    ),

    Path("/content/best_unet_restore.pth"),

    Path("best_unet_restore.pth"),

    Path(
        "/content/drive/MyDrive/"
        "CV_Final_Project_outputs/"
        "checkpoints/best_unet_restore.pth"
    ),

    Path(
        "/content/drive/MyDrive/"
        "CV_Final_Project/"
        "best_unet_restore.pth"
    )
]


MODEL_PATH = None

for checkpoint_path in possible_checkpoint_paths:
    if checkpoint_path.exists():
        MODEL_PATH = checkpoint_path
        break


if MODEL_PATH is None:
    raise FileNotFoundError(
        "The trained model checkpoint could not be found.\n\n"
        "Expected filename: best_unet_restore.pth\n\n"
        "Check the Colab Files panel and make sure the checkpoint "
        "exists inside:\n"
        "/content/CV_Final_Project_outputs/checkpoints/"
    )


print("Checkpoint found:")
print(MODEL_PATH.resolve())


# 4. CREATE THE MODEL AND LOAD THE TRAINED WEIGHTS
ui_model = UNet(
    in_channels=3,
    out_channels=3,
    features=(64, 128, 256, 512)
).to(DEVICE)


try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

except TypeError:
    # Compatibility with older PyTorch versions
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


# The improved training code saves a checkpoint dictionary.
# This fallback also supports a plain model state dictionary.
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    ui_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_epoch = checkpoint.get(
        "epoch",
        "Unknown"
    )

    best_validation_loss = checkpoint.get(
        "best_val_loss",
        "Unknown"
    )

else:
    ui_model.load_state_dict(checkpoint)

    best_epoch = "Not stored"
    best_validation_loss = "Not stored"


ui_model.eval()

print("The trained U-Net was loaded successfully.")
print("Best saved epoch:", best_epoch)
print("Best validation loss:", best_validation_loss)


# 5. IMAGE PREPROCESSING
# This must match the preprocessing used during training.
IMAGE_SIZE = 256

ui_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        antialias=True
    ),
    transforms.ToTensor()
])

tensor_to_image = transforms.ToPILImage()


# 6. PHOTO RESTORATION FUNCTION
def restore_photo(uploaded_image):
    """
    Restore one damaged photograph using the trained U-Net.

    Returns:
        1. Resized model input
        2. Restored image
        3. Downloadable PNG file
        4. Processing status
    """

    if uploaded_image is None:
        return (
            None,
            None,
            None,
            "Please upload a damaged photograph first."
        )

    try:
        # Ensure that the image has three RGB channels
        uploaded_image = uploaded_image.convert("RGB")

        # Create a preview of the exact image resolution used by the model
        input_preview = uploaded_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Convert the image to a normalized tensor
        input_tensor = ui_transform(uploaded_image)

        # Add the batch dimension:
        # [3, 256, 256] -> [1, 3, 256, 256]
        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Run model inference without gradient calculation
        with torch.inference_mode():
            restored_tensor = ui_model(input_tensor)

        # Remove the batch dimension and return data to the CPU
        restored_tensor = (
            restored_tensor
            .squeeze(0)
            .detach()
            .float()
            .cpu()
            .clamp(0.0, 1.0)
        )

        # Convert the restored tensor to a PIL image
        restored_image = tensor_to_image(restored_tensor)

        # Create a temporary downloadable PNG file
        temporary_file = tempfile.NamedTemporaryFile(
            prefix="restored_photo_",
            suffix=".png",
            delete=False
        )

        output_path = temporary_file.name
        temporary_file.close()

        restored_image.save(
            output_path,
            format="PNG"
        )

        status_message = (
            "Restoration completed successfully.\n"
            f"Model input resolution: {IMAGE_SIZE} × {IMAGE_SIZE}\n"
            f"Inference device: {DEVICE}\n"
            f"Best saved epoch: {best_epoch}\n"
            f"Best validation loss: {best_validation_loss}"
        )

        return (
            input_preview,
            restored_image,
            output_path,
            status_message
        )

    except Exception as error:
        return (
            None,
            None,
            None,
            f"Image restoration failed:\n{error}"
        )


# 7. CLOSE ANY PREVIOUS GRADIO INTERFACE
try:
    gr.close_all()

except Exception:
    pass


# 8. BUILD THE ENGLISH USER INTERFACE
with gr.Blocks(
    title="U-Net Old Photo Restoration"
) as demo:

    gr.Markdown(
        """
# Old Photograph Restoration Using U-Net

Upload a damaged or degraded photograph and press  
**Restore Photograph**.

The trained U-Net processes the image at a resolution of
**256 × 256 pixels** and produces a restored version.

The restored output can also be downloaded as a PNG file.
        """
    )

    uploaded_image_input = gr.Image(
        label="Upload Damaged Photograph",
        type="pil",
        image_mode="RGB",
        sources=[
            "upload",
            "webcam",
            "clipboard"
        ]
    )

    with gr.Row():

        input_preview_output = gr.Image(
            label="Model Input — 256 × 256",
            type="pil",
            interactive=False
        )

        restored_image_output = gr.Image(
            label="Restored Photograph",
            type="pil",
            interactive=False
        )

    with gr.Row():

        restore_button = gr.Button(
            "Restore Photograph",
            variant="primary"
        )

        clear_button = gr.ClearButton(
            components=[
                uploaded_image_input,
                input_preview_output,
                restored_image_output
            ],
            value="Clear"
        )

    downloadable_file_output = gr.File(
        label="Download Restored PNG"
    )

    status_output = gr.Textbox(
        label="Processing Status",
        lines=5,
        interactive=False
    )

    restore_button.click(
        fn=restore_photo,
        inputs=[
            uploaded_image_input
        ],
        outputs=[
            input_preview_output,
            restored_image_output,
            downloadable_file_output,
            status_output
        ]
    )


# 9. LAUNCH THE INTERFACE
IS_COLAB = "google.colab" in sys.modules

print("Running inside Google Colab:", IS_COLAB)

demo.queue().launch(
    inline=True,
    share=True,
    debug=False,
    show_error=True
)