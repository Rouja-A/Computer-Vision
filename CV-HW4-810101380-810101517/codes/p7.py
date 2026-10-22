import os
import pandas as pd
import matplotlib.pyplot as plt

history_file = "training_history_question_5.csv"

if not os.path.exists(history_file):
    raise FileNotFoundError(
        "training_history_question_5.csv was not found. Run Question 5 first."
    )

history = pd.read_csv(history_file)

print("Training history columns:")
print(history.columns)

if "epoch" in history.columns:
    epochs = history["epoch"] + 1
else:
    epochs = range(1, len(history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("question_7_accuracy_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["loss"], marker="o", label="Training Loss")
plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("question_7_loss_plot.png", dpi=300)
plt.show()

best_val_accuracy = history["val_accuracy"].max()
best_val_accuracy_epoch = history["val_accuracy"].idxmax() + 1

best_val_loss = history["val_loss"].min()
best_val_loss_epoch = history["val_loss"].idxmin() + 1

final_train_accuracy = history["accuracy"].iloc[-1]
final_val_accuracy = history["val_accuracy"].iloc[-1]

final_train_loss = history["loss"].iloc[-1]
final_val_loss = history["val_loss"].iloc[-1]

print("Best validation accuracy:")
print(best_val_accuracy)

print("Best validation accuracy epoch:")
print(best_val_accuracy_epoch)

print("Best validation loss:")
print(best_val_loss)

print("Best validation loss epoch:")
print(best_val_loss_epoch)

print("Final training accuracy:")
print(final_train_accuracy)

print("Final validation accuracy:")
print(final_val_accuracy)

print("Final training loss:")
print(final_train_loss)

print("Final validation loss:")
print(final_val_loss)

if final_train_accuracy > final_val_accuracy + 0.10 and final_val_loss > final_train_loss:
    print("Conclusion: The model shows signs of overfitting.")
elif final_train_accuracy < 0.60 and final_val_accuracy < 0.60:
    print("Conclusion: The model shows signs of underfitting.")
else:
    print("Conclusion: The model does not show severe overfitting or underfitting.")