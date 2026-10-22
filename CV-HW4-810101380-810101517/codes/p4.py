import tensorflow as tf

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
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

print("Model has been compiled successfully.")
print("Optimizer: Adam")
print("Loss function: Sparse Categorical Crossentropy")
print("Metric: Accuracy")