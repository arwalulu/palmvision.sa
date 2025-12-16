# src/training/train.py
import os, sys, json, datetime
import pandas as pd
from pathlib import Path
import tensorflow as tf

from src.configs.loader import load_config
from src.data.tfdata import build_datasets
from src.models.effb0_cbam import build_effb0_cbam
from src.training.callbacks import make_callbacks


def compute_class_weights(train_csv: Path, classes: list[str]) -> dict[int, float]:
    """Simple inverse-frequency class weights for mild imbalance."""
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    total = sum(counts.values())
    inv = {c: total / counts.get(c, 1) for c in classes}
    mean = sum(inv.values()) / len(inv)
    inv = {c: v / mean for c, v in inv.items()}
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return {class_to_idx[c]: float(inv[c]) for c in classes}


def main():
    cfg = load_config()

    # Reproducibility
    tf.keras.utils.set_random_seed(int(cfg.seed))

    # Datasets
    train_ds, val_ds, test_ds, idx_to_class = build_datasets(cfg)

    # Build model
    model = build_effb0_cbam(cfg)

    # Run directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("experiments") / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "config.yaml").write_text(
        Path("src/configs/default.yaml").read_text(), encoding="utf-8"
    )

    # Callbacks
    cbs = make_callbacks(run_dir)

    # Class weights
    train_csv = Path("manifests/splits/train.csv")
    class_weights = compute_class_weights(train_csv, cfg.classes)

    # Train
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.training["epochs"]),
        callbacks=cbs,
        class_weight=class_weights,
        verbose=1,
    )

    # Save experiment model
    model.save(run_dir / "final.keras")
    with (run_dir / "history.json").open("w") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in hist.history.items()},
            f,
            indent=2,
        )

    # === SAVE BACKEND INFERENCE MODEL ===
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    backend_path = models_dir / "palmvision_effb0_cbam.keras"
    model.save(backend_path)
    print(f"[INFO] Saved backend model to: {backend_path}")

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best checkpoint: {run_dir/'checkpoints'/'best-val-acc.keras'}")
    print(f"[INFO] TensorBoard logs: {run_dir/'tb'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
