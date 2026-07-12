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


def test_output_varies_with_input():
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=2, backend="default.qubit")
    x1 = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2)
