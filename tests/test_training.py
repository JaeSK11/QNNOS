"""Tests for training loop, early stopping, and checkpointing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.hybrid_model import HybridQuantumNet
from training.trainer import (
    compute_class_weights,
    load_resume_checkpoint,
    save_resume_checkpoint,
    train_model,
)


def _make_synthetic_data(n_samples=40, n_qubits=4, n_classes=2):
    """Create synthetic binary data for training tests."""
    rng = torch.Generator().manual_seed(42)
    X = torch.randint(0, 2, (n_samples, n_qubits), generator=rng, dtype=torch.float32)
    y = torch.randint(0, n_classes, (n_samples,), generator=rng)
    return X, y


def test_compute_class_weights():
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    weights = compute_class_weights(y, n_classes=3)
    assert weights.shape == (3,)
    assert weights[2] > weights[0]  # rarer class gets higher weight


def test_train_model_runs(tmp_path):
    n_qubits = 4
    n_classes = 2
    X, y = _make_synthetic_data(n_samples=40, n_qubits=n_qubits, n_classes=n_classes)

    split = 32
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=8)
    test_loader = DataLoader(TensorDataset(X[split:], y[split:]), batch_size=8)

    model = HybridQuantumNet(
        n_classes=n_classes, n_qubits=n_qubits, n_layers=1, backend="default.qubit"
    )

    config = {
        "n_qubits": n_qubits,
        "n_layers": 1,
        "n_classes": n_classes,
        "feature_indices": list(range(n_qubits)),
        "label_classes": ["class_0", "class_1"],
    }

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_classes=n_classes,
        y_train=y[:split],
        epochs=3,
        lr=0.01,
        patience=10,
        save_dir=str(tmp_path),
        model_name="test_model",
        config=config,
    )

    assert "train_loss" in history
    assert "val_f1" in history
    assert len(history["train_loss"]) == 3


def test_early_stopping(tmp_path):
    """With patience=1 and random data, early stopping should trigger before max epochs."""
    n_qubits = 4
    n_classes = 2
    X, y = _make_synthetic_data(n_samples=40, n_qubits=n_qubits, n_classes=n_classes)

    split = 32
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=8)
    test_loader = DataLoader(TensorDataset(X[split:], y[split:]), batch_size=8)

    model = HybridQuantumNet(
        n_classes=n_classes, n_qubits=n_qubits, n_layers=1, backend="default.qubit"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_classes=n_classes,
        y_train=y[:split],
        epochs=100,
        patience=1,
        save_dir=str(tmp_path),
        model_name="es_model",
    )

    # Should stop well before 100 epochs
    assert len(history["train_loss"]) < 100


def test_resume_checkpoint_roundtrip(tmp_path):
    """save/load_resume_checkpoint round-trips; load returns None when absent."""
    assert load_resume_checkpoint(str(tmp_path), "missing") is None

    state = {"epoch": 3, "best_f1": 0.42, "best_epoch": 2,
             "epochs_without_improvement": 1, "optimizer": None, "scheduler": None}
    save_resume_checkpoint(str(tmp_path), "m", state)
    loaded = load_resume_checkpoint(str(tmp_path), "m")
    assert loaded["epoch"] == 3
    assert loaded["best_f1"] == 0.42
    assert loaded["best_epoch"] == 2


def test_train_model_resumes_from_state(tmp_path):
    """A second train_model call with resume_state continues from the saved
    epoch instead of restarting, and writes a resume checkpoint each epoch."""
    n_qubits, n_classes = 4, 2
    X, y = _make_synthetic_data(n_samples=40, n_qubits=n_qubits, n_classes=n_classes)
    split = 32
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=8)
    test_loader = DataLoader(TensorDataset(X[split:], y[split:]), batch_size=8)

    def fresh_model():
        return HybridQuantumNet(
            n_classes=n_classes, n_qubits=n_qubits, n_layers=1, backend="default.qubit"
        )

    config = {
        "n_qubits": n_qubits, "n_layers": 1, "n_classes": n_classes,
        "feature_indices": list(range(n_qubits)), "label_classes": ["c0", "c1"],
    }

    # First leg: 2 epochs. A resume checkpoint should be written at epoch 2.
    train_model(
        model=fresh_model(), train_loader=train_loader, test_loader=test_loader,
        n_classes=n_classes, y_train=y[:split], epochs=2, patience=10,
        save_dir=str(tmp_path), model_name="rm", config=config,
    )
    state = load_resume_checkpoint(str(tmp_path), "rm")
    assert state is not None and state["epoch"] == 2

    # Second leg: resume to a total of 4 epochs -> only 2 new epochs should run.
    model = fresh_model()
    model.load_state_dict(state["model"])
    history = train_model(
        model=model, train_loader=train_loader, test_loader=test_loader,
        n_classes=n_classes, y_train=y[:split], epochs=4, patience=10,
        save_dir=str(tmp_path), model_name="rm", config=config,
        resume_state=state,
    )
    assert len(history["train_loss"]) == 2  # ran epochs 3 and 4 only
    assert load_resume_checkpoint(str(tmp_path), "rm")["epoch"] == 4
