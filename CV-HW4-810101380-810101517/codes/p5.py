import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf
import pandas as pd

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

train_dir = "Intel_Image_Classification/seg_train/seg_train"

img_size = (150, 150)
batch_size = 32
seed_value = 123
validation_ratio = 0.2
epochs_count = 15

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

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(
    buffer_size=1000,
    seed=seed_value,
    reshuffle_each_iteration=True
).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),

    data_augmentation,

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="cnn_model_best_question_5.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger(
    filename="training_history_question_5.csv",
    append=False
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_count,
    callbacks=[checkpoint, csv_logger, reduce_lr],
    verbose=2
)

history_table = pd.DataFrame(history.history)

print("Training history:")
print(history_table)

history_table.to_csv("training_history_question_5_from_history.csv", index=False)

model.save("cnn_model_final_question_5.keras")

best_val_accuracy = max(history.history["val_accuracy"])
best_epoch = history.history["val_accuracy"].index(best_val_accuracy) + 1

print("Training completed.")
print("Best validation accuracy:")
print(best_val_accuracy)
print("Best epoch:")
print(best_epoch)
print("Training history has been saved as training_history_question_5.csv")
print("Best model has been saved as cnn_model_best_question_5.keras")
print("Final model has been saved as cnn_model_final_question_5.keras")