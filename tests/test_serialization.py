"""Tests for model save/load (serialization)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.hybrid_model import HybridQuantumNet
from training.trainer import load_checkpoint, save_checkpoint


def test_save_and_load(tmp_path):
    n_qubits = 4
    n_layers = 1
    n_classes = 2

    model = HybridQuantumNet(
        n_classes=n_classes, n_qubits=n_qubits, n_layers=n_layers, backend="default.qubit"
    )

    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "feature_indices": [0, 1, 2, 3],
        "label_classes": ["Linux", "Windows"],
    }

    save_checkpoint(model, config, str(tmp_path), "test_model")

    assert (tmp_path / "test_model.pt").exists()
    assert (tmp_path / "test_model.json").exists()

    loaded_model, loaded_config = load_checkpoint(
        str(tmp_path), "test_model", backend="default.qubit"
    )
    assert loaded_config["n_qubits"] == n_qubits
    assert loaded_config["n_classes"] == n_classes
    assert loaded_config["label_classes"] == ["Linux", "Windows"]


def test_loaded_model_produces_same_output(tmp_path):
    n_qubits = 4
    n_layers = 1
    n_classes = 2

    model = HybridQuantumNet(
        n_classes=n_classes, n_qubits=n_qubits, n_layers=n_layers, backend="default.qubit"
    )
    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "feature_indices": [0, 1, 2, 3],
        "label_classes": ["Linux", "Windows"],
    }

    save_checkpoint(model, config, str(tmp_path), "repro_model")

    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        original_out = model(x)

    loaded_model, _ = load_checkpoint(str(tmp_path), "repro_model", backend="default.qubit")
    loaded_model.eval()
    with torch.no_grad():
        loaded_out = loaded_model(x)

    assert torch.allclose(original_out, loaded_out, atol=1e-6)
