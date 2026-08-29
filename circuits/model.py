"""Custom variational quantum circuit for OS fingerprinting classification."""

import numpy as np
import pennylane as qml

from config.backends import get_backend
from config.defaults import (
    DEFAULT_BACKEND,
    ENTANGLER,
    MEASURE_AXES,
    N_LAYERS,
    N_QUBITS,
    SHORTCUT_OFFSETS,
)

_AXIS_TO_OBS = {"Z": qml.PauliZ, "X": qml.PauliX, "Y": qml.PauliY}


def create_circuit(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    backend: str = DEFAULT_BACKEND,
    entangler: str = ENTANGLER,
    measure_axes: list[str] | None = None,
    shortcut_offsets: list[int] | None = None,
) -> tuple[qml.QNode, dict]:
    """Create a variational quantum circuit QNode.

    Args:
        entangler: per-layer entangling pattern.
            "cnot" - fixed CNOT ring (2 trainable params/qubit/layer: RY, RZ).
            "crz"  - trainable controlled-RZ ring (3 params/qubit/layer: RY, RZ,
                     CRZ angle) so the model learns how strongly to entangle each
                     neighboring pair.
            "ring_long" - fixed CNOT ring plus long-range CNOT shortcuts at
                     ``shortcut_offsets`` (2 params/qubit/layer, weight-compatible
                     with "cnot"). Adds reachability between distant qubits.
        measure_axes: Pauli axes measured per qubit (default ["Z","X","Y"] = 60
            outputs). ["Z"] = 20 outputs, much cheaper under adjoint diff.
        shortcut_offsets: extra CNOT offsets for "ring_long" (i <-> (i+k) % N).

    Returns:
        qnode: The quantum circuit as a QNode with torch interface.
        weight_shapes: Dict mapping parameter names to shapes for TorchLayer.
    """
    if entangler not in ("cnot", "crz", "ring_long"):
        raise ValueError(
            f"Unknown entangler {entangler!r}; expected 'cnot', 'crz', or 'ring_long'."
        )
    measure_axes = list(measure_axes) if measure_axes is not None else list(MEASURE_AXES)
    bad_axes = [a for a in measure_axes if a not in _AXIS_TO_OBS]
    if bad_axes:
        raise ValueError(f"Unknown measure_axes {bad_axes}; expected any of Z, X, Y.")
    shortcut_offsets = (
        list(shortcut_offsets) if shortcut_offsets is not None else list(SHORTCUT_OFFSETS)
    )

    # "crz" needs a third trainable parameter per qubit for the CRZ angle.
    params_per_qubit = 3 if entangler == "crz" else 2

    def _entangle(layer, weights):
        # Nearest-neighbor ring (all entanglers include it).
        if entangler == "crz":
            for i in range(n_qubits):
                qml.CRZ(weights[layer, i, 2], wires=[i, (i + 1) % n_qubits])
            return
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        # "ring_long" adds fixed long-range CNOT shortcuts for reachability.
        if entangler == "ring_long":
            for k in shortcut_offsets:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + k) % n_qubits])

    backend_config = get_backend(backend)
    dev = qml.device(backend_config["device"], wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=backend_config["diff_method"])
    def circuit(inputs, weights):
        # Encode ternary nPrint features {-1, 0, 1} as RY rotations mapping
        # -1 -> RY(0) = |0>, 0 -> RY(pi/2) = equator, 1 -> RY(pi) = |1>
        # (supports batched inputs). The affine shift is REQUIRED: the previous
        # RY(x * pi) encoding aliased -1 ("field absent") with +1 ("bit set"),
        # because RY(-pi) = -RY(pi) — identical up to global phase, hence
        # identical measurement statistics everywhere downstream (even through
        # the re-uploading layers). 58% of selected-feature values are -1, so
        # the aliasing was not an edge case.
        qml.AngleEmbedding((inputs + 1) * (np.pi / 2), wires=range(n_qubits), rotation="Y")

        # Ring of CNOTs (nearest-neighbor + wrap-around). NOTE: with the
        # ternary encoding, 0-valued features sit on the Bloch equator
        # (superposition), so this ring genuinely entangles whenever any
        # feature is 0 — unlike the old binary encoding, where the
        # post-encoding state was a basis state and the ring was only a
        # classical XOR permutation.
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Variational layers. Each layer re-uploads the input features
        # (data re-uploading) before its trainable rotations, which increases
        # the expressivity of the circuit as a function approximator.
        for layer in range(n_layers):
            qml.AngleEmbedding((inputs + 1) * (np.pi / 2), wires=range(n_qubits), rotation="Y")
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            _entangle(layer, weights)

        # Measure each qubit on the configured Pauli axes. Output is grouped by
        # axis then qubit, e.g. ["Z","X","Y"] -> [Z0..Z19, X0..X19, Y0..Y19].
        # Qubit i's measured components are at indices [j*n_qubits + i] for each
        # axis j, so a full-Bloch config reconstructs via [i, i+N, i+2N].
        return [qml.expval(_AXIS_TO_OBS[ax](i)) for ax in measure_axes
                for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, params_per_qubit)}

    return circuit, weight_shapes
