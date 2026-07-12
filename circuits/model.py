"""Custom variational quantum circuit for OS fingerprinting classification."""

import numpy as np
import pennylane as qml

from config.backends import get_backend
from config.defaults import DEFAULT_BACKEND, N_LAYERS, N_QUBITS


def create_circuit(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    backend: str = DEFAULT_BACKEND,
) -> tuple[qml.QNode, dict]:
    """Create a variational quantum circuit QNode.

    Returns:
        qnode: The quantum circuit as a QNode with torch interface.
        weight_shapes: Dict mapping parameter names to shapes for TorchLayer.
    """
    backend_config = get_backend(backend)
    dev = qml.device(backend_config["device"], wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=backend_config["diff_method"])
    def circuit(inputs, weights):
        # Encode binary features as RY rotations (supports batched inputs)
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation="Y")

        # Ring of CNOTs (nearest-neighbor + wrap-around). NOTE: because the
        # binary features are encoded as RY(0 or pi), the post-encoding state is
        # a computational basis state, so this ring is a classical XOR
        # permutation of bits and creates NO entanglement on its own. Genuine
        # entanglement only arises once the trainable RY/RZ rotations below put
        # qubits into superposition.
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Variational layers. Each layer re-uploads the input features
        # (data re-uploading) before its trainable rotations, which increases
        # the expressivity of the circuit as a function approximator.
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation="Y")
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measure all qubits on three Pauli axes (3x information extraction)
        # Output ordering: [Z0..Z19, X0..X19, Y0..Y19] (grouped by observable).
        # To reconstruct qubit i's Bloch vector: indices [i, i+n_qubits, i+2*n_qubits].
        return [qml.expval(obs(i)) for obs in (qml.PauliZ, qml.PauliX, qml.PauliY)
                for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}

    return circuit, weight_shapes
