"""Hybrid quantum-classical model: TorchLayer + classical head."""

import torch
import torch.nn as nn
import pennylane as qml

from circuits.model import create_circuit
from config.defaults import (
    DEFAULT_BACKEND,
    DROPOUT,
    ENTANGLER,
    MEASURE_AXES,
    N_LAYERS,
    N_QUBITS,
    SHORTCUT_OFFSETS,
    USE_BATCHNORM,
    USE_SKIP,
)


class HybridQuantumNet(nn.Module):
    """Variational quantum classifier with classical post-processing head.

    Head input is the concatenation of, optionally:
      - the quantum layer's len(measure_axes)*n_qubits Pauli expvals (when
        use_quantum; 60 for the default Z+X+Y on 20 qubits),
      - the raw n_qubits input features (when use_skip; a residual path so the
        head can use the data the circuit maps poorly, and provably >= a
        classical MLP on the raw features).

    Setting use_quantum=False with use_skip=True gives a purely classical
    baseline (raw features -> head), for A/B-ing the circuit's contribution.

    Head: [BatchNorm1d?] -> Linear(in, 128) -> ReLU -> Dropout(p) -> Linear(128, n_classes)
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        backend: str = DEFAULT_BACKEND,
        entangler: str = ENTANGLER,
        use_quantum: bool = True,
        use_skip: bool = USE_SKIP,
        use_batchnorm: bool = USE_BATCHNORM,
        dropout: float = DROPOUT,
        measure_axes: list[str] | None = None,
        shortcut_offsets: list[int] | None = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.entangler = entangler
        self.use_quantum = use_quantum
        self.use_skip = use_skip
        self.measure_axes = list(measure_axes) if measure_axes is not None else list(MEASURE_AXES)
        self.shortcut_offsets = (
            list(shortcut_offsets) if shortcut_offsets is not None else list(SHORTCUT_OFFSETS)
        )

        if use_quantum:
            circuit, weight_shapes = create_circuit(
                n_qubits, n_layers, backend, entangler,
                measure_axes=self.measure_axes, shortcut_offsets=self.shortcut_offsets,
            )
            self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
            # One expval per measured Pauli axis per qubit.
            q_out_dim = len(self.measure_axes) * n_qubits
        else:
            self.quantum_layer = None
            q_out_dim = 0

        skip_dim = n_qubits if use_skip else 0  # raw features = one per qubit
        head_in = q_out_dim + skip_dim
        if head_in == 0:
            raise ValueError(
                "Model has no inputs: enable use_quantum and/or use_skip."
            )

        head_layers = []
        if use_batchnorm:
            head_layers.append(nn.BatchNorm1d(head_in))
        head_layers += [
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        ]
        self.classical_head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (quantum expvals and/or raw features) -> head -> logits."""
        parts = []
        if self.use_quantum:
            parts.append(self.quantum_layer(x))
        if self.use_skip:
            parts.append(x)
        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.classical_head(h)
