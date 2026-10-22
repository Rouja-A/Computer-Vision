import os
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = "Intel_Image_Classification/seg_train/seg_train"
test_dir = "Intel_Image_Classification/seg_test/seg_test"

img_size = (150, 150)
batch_size = 32
seed_value = 123
validation_ratio = 0.2

print("Checking dataset paths")
print("Train directory exists:", os.path.exists(train_dir))
print("Test directory exists:", os.path.exists(test_dir))

print("\nTrain directory classes:")
print(os.listdir(train_dir))

print("\nTest directory classes:")
print(os.listdir(test_dir))

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
    batch_size=batch_size
)

class_names = train_ds.class_names

print("\nClass names:")
print(class_names)

print("\nBatch shapes:")

for images, labels in train_ds.take(1):
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()