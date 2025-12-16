cat > figg.py << 'EOF'
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.configs.loader import load_config
from src.data.tfdata import build_datasets
from src.models.cbam import SpatialAttention
from src.models.effb0_cbam import CLASS_NAMES

def main():
    cfg = load_config()
    _, _, test_ds, _ = build_datasets(cfg)

    model_path = Path("models/palmvision_effb0_cbam.keras")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SpatialAttention": SpatialAttention},
        compile=False,
    )

    # compile so metrics/loss are defined if needed
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    y_true, y_pred = [], []

    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    print("\n========= CONFUSION MATRIX =========")
    print("Class order:", CLASS_NAMES)
    print(cm)

    print("\n========= CLASSIFICATION REPORT =========")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plot and save confusion matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - EfficientNetB0 + CBAM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    # write numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    out = Path("confusion_matrix.png")
    plt.savefig(out, dpi=200)
    print(f"\n[OK] Saved: {out.resolve()}")
EOF
