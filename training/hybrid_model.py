"""Hybrid quantum-classical model: TorchLayer + classical head."""

import torch
import torch.nn as nn
import pennylane as qml

from circuits.model import create_circuit
from config.defaults import DEFAULT_BACKEND, N_LAYERS, N_QUBITS


class HybridQuantumNet(nn.Module):
    """Variational quantum classifier with classical post-processing head.

    Architecture:
        Input (n_qubits) -> TorchLayer(QNode) -> [3*n_qubits expvals (Z,X,Y)]
        -> Linear(3*n_qubits, 128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes) -> logits
    """

    def __init__(
        self,
        n_classes: int,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        backend: str = DEFAULT_BACKEND,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        circuit, weight_shapes = create_circuit(n_qubits, n_layers, backend)
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        q_out_dim = 3 * n_qubits  # PauliZ + PauliX + PauliY measurements
        self.classical_head = nn.Sequential(
            nn.Linear(q_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantum circuit -> classical head -> logits."""
        q_out = self.quantum_layer(x)
        return self.classical_head(q_out)
