"""Custom variational quantum circuit for OS fingerprinting classification."""

import numpy as np
import pennylane as qml

from config.backends import get_backend
from config.defaults import DEFAULT_BACKEND, ENTANGLER, N_LAYERS, N_QUBITS


def create_circuit(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    backend: str = DEFAULT_BACKEND,
    entangler: str = ENTANGLER,
) -> tuple[qml.QNode, dict]:
    """Create a variational quantum circuit QNode.

    Args:
        entangler: per-layer entangling ring. "cnot" uses a fixed CNOT ring
            (2 trainable params/qubit/layer: RY, RZ). "crz" uses a trainable
            controlled-RZ ring (3 params/qubit/layer: RY, RZ, CRZ angle) so the
            model can learn how strongly to entangle each neighboring pair.

    Returns:
        qnode: The quantum circuit as a QNode with torch interface.
        weight_shapes: Dict mapping parameter names to shapes for TorchLayer.
    """
    if entangler not in ("cnot", "crz"):
        raise ValueError(f"Unknown entangler {entangler!r}; expected 'cnot' or 'crz'.")

    # "crz" needs a third trainable parameter per qubit for the CRZ angle.
    params_per_qubit = 3 if entangler == "crz" else 2

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
            # Entangling ring: fixed CNOT, or trainable CRZ whose angle the
            # optimizer tunes per pair (CRZ(0) = identity = no entanglement).
            for i in range(n_qubits):
                if entangler == "crz":
                    qml.CRZ(weights[layer, i, 2], wires=[i, (i + 1) % n_qubits])
                else:
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measure all qubits on three Pauli axes (3x information extraction)
        # Output ordering: [Z0..Z19, X0..X19, Y0..Y19] (grouped by observable).
        # To reconstruct qubit i's Bloch vector: indices [i, i+n_qubits, i+2*n_qubits].
        return [qml.expval(obs(i)) for obs in (qml.PauliZ, qml.PauliX, qml.PauliY)
                for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, params_per_qubit)}

    return circuit, weight_shapes
