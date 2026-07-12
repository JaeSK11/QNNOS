"""Tests for quantum circuit construction and output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pennylane as qml
import torch

from circuits.model import create_circuit


def test_circuit_creation():
    circuit, weight_shapes = create_circuit(n_qubits=4, n_layers=2, backend="default.qubit")
    assert "weights" in weight_shapes
    assert weight_shapes["weights"] == (2, 4, 2)


def test_circuit_output_shape():
    circuit, weight_shapes = create_circuit(n_qubits=4, n_layers=2, backend="default.qubit")

    inputs = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    weights = torch.randn(2, 4, 2, dtype=torch.float64)

    result = circuit(inputs, weights)
    assert len(result) == 12  # 3 observables (Z, X, Y) * 4 qubits
    for val in result:
        assert -1.0 <= float(val) <= 1.0


def test_circuit_output_changes_with_weights():
    circuit, _ = create_circuit(n_qubits=4, n_layers=2, backend="default.qubit")

    inputs = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    w1 = torch.zeros(2, 4, 2, dtype=torch.float64)
    w2 = torch.ones(2, 4, 2, dtype=torch.float64)

    r1 = [float(v) for v in circuit(inputs, w1)]
    r2 = [float(v) for v in circuit(inputs, w2)]
    assert r1 != r2


def test_circuit_crz_entangler_shapes():
    """The trainable 'crz' entangler adds a third param per qubit/layer and
    keeps the same 3*n_qubits measurement outputs."""
    circuit, weight_shapes = create_circuit(
        n_qubits=4, n_layers=2, backend="default.qubit", entangler="crz"
    )
    assert weight_shapes["weights"] == (2, 4, 3)

    inputs = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    weights = torch.randn(2, 4, 3, dtype=torch.float64)
    result = circuit(inputs, weights)
    assert len(result) == 12
    for val in result:
        assert -1.0 <= float(val) <= 1.0


def test_circuit_crz_angle_changes_output():
    """The CRZ entangler angle (the 3rd param) affects the output once qubits
    are in superposition. (With zero RY the state stays a basis state and CRZ,
    being diagonal, only adds an unobservable phase — the same reason the
    encoding CNOT ring creates no entanglement on basis states.)"""
    circuit, _ = create_circuit(
        n_qubits=4, n_layers=1, backend="default.qubit", entangler="crz"
    )
    inputs = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    w = torch.zeros(1, 4, 3, dtype=torch.float64)
    w[0, :, 0] = 0.7  # non-zero RY -> superposition, so CRZ can entangle
    w_ent = w.clone()
    w_ent[0, :, 2] = 1.5  # turn on the CRZ entangling angles
    r_off = [float(v) for v in circuit(inputs, w)]
    r_on = [float(v) for v in circuit(inputs, w_ent)]
    assert r_off != r_on


def test_invalid_entangler_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown entangler"):
        create_circuit(n_qubits=4, n_layers=1, backend="default.qubit", entangler="bogus")


def test_circuit_binary_encoding():
    """Verify that all-zero input produces different output than all-one input."""
    circuit, _ = create_circuit(n_qubits=4, n_layers=1, backend="default.qubit")

    weights = torch.zeros(1, 4, 2, dtype=torch.float64)
    zeros = torch.zeros(4, dtype=torch.float64)
    ones = torch.ones(4, dtype=torch.float64)

    r_zeros = [float(v) for v in circuit(zeros, weights)]
    r_ones = [float(v) for v in circuit(ones, weights)]
    assert r_zeros != r_ones
