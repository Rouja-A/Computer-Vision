import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_dir = "Intel_Image_Classification/seg_train/seg_train"
test_dir = "Intel_Image_Classification/seg_test/seg_test"

img_size = (150, 150)
batch_size = 32
epochs_count = 15
validation_ratio = 0.2
seed_value = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=validation_ratio,
    subset="training",
    seed=seed_value,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=validation_ratio,
    subset="validation",
    seed=seed_value,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
print("\nClasses:", class_names)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(2000, seed=seed_value, reshuffle_each_iteration=True).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Improved Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# CNN Model (Optimized Capacity & Regularization)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),

    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    # Block 1
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 2
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 3
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 4 
    tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Block 5
    tf.keras.layers.Conv2D(
        512,
        (3,3),
        padding="same",
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    # Global spatial reduction
    tf.keras.layers.GlobalAveragePooling2D(),

    # Classification Head
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2), # Reduced from 0.3

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.1), # Reduced from 0.2

    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cnn_model_best_p8.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    "training_history_p8.csv"
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_count,
    callbacks=[checkpoint, csv_logger, reduce_lr, early_stop],
    verbose=2
)

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history_p8_from_history.csv", index=False)
model.save("cnn_model_final_p8.keras")

best_model = tf.keras.models.load_model("cnn_model_best_p8.keras")
test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\n=== TEST RESULTS ===")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

print("\nBest val accuracy:", history_df["val_accuracy"].max())
print("Best val loss:", history_df["val_loss"].min())

epochs = range(1, len(history_df) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, history_df["accuracy"], label="Train")
plt.plot(epochs, history_df["val_accuracy"], label="Validation")
plt.title("Accuracy - Optimized P8")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("p8_accuracy.png", dpi=300)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs, history_df["loss"], label="Train")
plt.plot(epochs, history_df["val_loss"], label="Validation")
plt.title("Loss - Optimized P8")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("p8_loss.png", dpi=300)
plt.show()