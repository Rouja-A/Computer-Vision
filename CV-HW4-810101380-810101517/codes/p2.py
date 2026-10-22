import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = "Intel_Image_Classification/seg_train/seg_train"
test_dir = "Intel_Image_Classification/seg_test/seg_test"

img_size = (150, 150)
batch_size = 32
seed_value = 123
validation_ratio = 0.2

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

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
], name="data_augmentation")

rescaling_layer = tf.keras.layers.Rescaling(1./255)

for images, labels in train_ds.take(1):
    normalized_images = rescaling_layer(images)

    print("Original pixel range:")
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())

    print("Normalized pixel range:")
    print("Minimum:", tf.reduce_min(normalized_images).numpy())
    print("Maximum:", tf.reduce_max(normalized_images).numpy())

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    selected_image = images[0]
    selected_label = labels[0]

    for i in range(9):
        augmented_image = data_augmentation(
            tf.expand_dims(selected_image, axis=0),
            training=True
        )

        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy().astype("uint8"))
        plt.title(class_names[selected_label])
        plt.axis("off")

plt.show()