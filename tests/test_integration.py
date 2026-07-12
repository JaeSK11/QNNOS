"""Integration test: synthetic data through full pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_accuracy, compute_f1, full_report
from training.hybrid_model import HybridQuantumNet
from training.trainer import train_model


def test_full_pipeline_synthetic(tmp_path):
    """Train on synthetic 2-class binary data and verify the pipeline doesn't crash."""
    n_qubits = 4
    n_classes = 2
    n_samples = 60

    rng = np.random.default_rng(42)
    # Class 0: mostly zeros; Class 1: mostly ones (separable data)
    X_class0 = rng.choice([0, 1], size=(n_samples // 2, n_qubits), p=[0.8, 0.2]).astype(
        np.float32
    )
    X_class1 = rng.choice([0, 1], size=(n_samples // 2, n_qubits), p=[0.2, 0.8]).astype(
        np.float32
    )
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    split = int(0.8 * n_samples)
    X_train = torch.tensor(X[:split])
    X_test = torch.tensor(X[split:])
    y_train = torch.tensor(y[:split], dtype=torch.long)
    y_test = torch.tensor(y[split:], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8)

    model = HybridQuantumNet(
        n_classes=n_classes, n_qubits=n_qubits, n_layers=2, backend="default.qubit"
    )

    config = {
        "n_qubits": n_qubits,
        "n_layers": 2,
        "n_classes": n_classes,
        "feature_indices": list(range(n_qubits)),
        "label_classes": ["class_0", "class_1"],
    }

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_classes=n_classes,
        y_train=y_train,
        epochs=5,
        lr=0.01,
        patience=10,
        save_dir=str(tmp_path),
        model_name="integration_model",
        config=config,
    )

    assert len(history["train_loss"]) > 0
    assert history["best_f1"] >= 0.0

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
    preds = logits.argmax(dim=1).numpy()

    acc = compute_accuracy(y_test.numpy(), preds)
    f1 = compute_f1(y_test.numpy(), preds)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

    report = full_report(y_test.numpy(), preds, class_names=["class_0", "class_1"])
    assert "accuracy" in report
    assert "confusion_matrix" in report

    # Verify checkpoint was saved
    assert (tmp_path / "integration_model.pt").exists()
    assert (tmp_path / "integration_model.json").exists()
