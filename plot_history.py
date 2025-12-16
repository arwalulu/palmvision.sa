import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history
run_dir = Path("experiments/run_20251209-174856")  # your latest run folder
history_path = run_dir / "history.json"

with open(history_path, "r") as f:
    hist = json.load(f)

plt.figure(figsize=(8,5))
plt.plot(hist["acc"], label="Train Accuracy")
plt.plot(hist["val_acc"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(hist["loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# You already have y_true and y_pred from earlier, reuse them:
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bug","Dubas","Healthy","Honey"],
            yticklabels=["Bug","Dubas","Healthy","Honey"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EfficientNetB0 + CBAM")
plt.show()
