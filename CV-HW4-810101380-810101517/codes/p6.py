import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

test_dir = "Intel_Image_Classification/seg_test/seg_test"

img_size = (150, 150)
batch_size = 8

model_path = "cnn_model_final_question_5.keras"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = test_ds.class_names

model = tf.keras.models.load_model(model_path)

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test loss:")
print(test_loss)

print("Test accuracy:")
print(test_accuracy)

if test_accuracy >= 0.80:
    print("The model reached the required minimum test accuracy.")
else:
    print("The model did not reach the required minimum test accuracy.")

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(12, 12))

shown_images = 0

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(len(images)):
        if shown_images >= 9:
            break

        true_label = labels[i].numpy()
        predicted_label = predicted_labels[i]

        plt.subplot(3, 3, shown_images + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(
            "True: " + class_names[true_label] + "\nPred: " + class_names[predicted_label]
        )
        plt.axis("off")

        shown_images += 1

    if shown_images >= 9:
        break

plt.show()

wrong_indices = np.where(y_true != y_pred)[0]

print("Number of wrong predictions:")
print(len(wrong_indices))

print("Some wrong predictions:")

for index in wrong_indices[:10]:
    print(
        "True:",
        class_names[y_true[index]],
        " | Predicted:",
        class_names[y_pred[index]]
    )

error_pairs = {}

for true_label, predicted_label in zip(y_true, y_pred):
    if true_label != predicted_label:
        pair = (class_names[true_label], class_names[predicted_label])

        if pair not in error_pairs:
            error_pairs[pair] = 0

        error_pairs[pair] += 1

sorted_errors = sorted(
    error_pairs.items(),
    key=lambda item: item[1],
    reverse=True
)

print("Most frequent confusion pairs:")

for pair, count in sorted_errors[:10]:
    print(pair[0], "was predicted as", pair[1], ":", count)