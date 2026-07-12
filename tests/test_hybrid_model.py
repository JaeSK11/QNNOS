"""Tests for HybridQuantumNet: forward pass and gradient flow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from training.hybrid_model import HybridQuantumNet


def test_forward_pass():
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit")
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (1, 3)


def test_batch_forward():
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit")
    x = torch.randint(0, 2, (4, 4), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_gradient_flow():
    """Verify gradients propagate through the quantum layer."""
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit")
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found — gradient flow is broken"


def test_skip_connection_forward():
    """With use_skip, the head sees quantum expvals + raw features (3*nq + nq)."""
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit", use_skip=True
    )
    # First head layer input dim = 3*4 (expvals) + 4 (raw) = 16
    assert model.classical_head[0].in_features == 16
    x = torch.randint(0, 2, (4, 4), dtype=torch.float32)
    assert model(x).shape == (4, 3)


def test_classical_baseline_no_quantum():
    """use_quantum=False + use_skip=True is a purely classical head on raw features."""
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit",
        use_quantum=False, use_skip=True,
    )
    assert model.quantum_layer is None
    assert model.classical_head[0].in_features == 4  # raw features only
    x = torch.randint(0, 2, (4, 4), dtype=torch.float32)
    assert model(x).shape == (4, 3)
    # No quantum parameters exist in a classical baseline.
    assert not any("quantum" in n for n, _ in model.named_parameters())


def test_batchnorm_head_forward():
    """BatchNorm1d prepends a norm layer and trains with batch > 1."""
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit", use_batchnorm=True
    )
    assert isinstance(model.classical_head[0], torch.nn.BatchNorm1d)
    model.train()
    x = torch.randint(0, 2, (4, 4), dtype=torch.float32)
    assert model(x).shape == (4, 3)


def test_dropout_configurable():
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=1, backend="default.qubit", dropout=0.1
    )
    dropouts = [m for m in model.classical_head if isinstance(m, torch.nn.Dropout)]
    assert dropouts and dropouts[0].p == 0.1


def test_no_inputs_raises():
    import pytest
    with pytest.raises(ValueError, match="no inputs"):
        HybridQuantumNet(
            n_classes=3, n_qubits=4, n_layers=1, backend="default.qubit",
            use_quantum=False, use_skip=False,
        )


def test_output_varies_with_input():
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit")
    x1 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2)
