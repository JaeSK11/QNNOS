BACKENDS = {
    "default.qubit": {
        "device": "default.qubit",
        "diff_method": "backprop",
    },
    "lightning.qubit": {
        "device": "lightning.qubit",
        "diff_method": "adjoint",
    },
    "lightning.gpu": {
        "device": "lightning.gpu",
        "diff_method": "adjoint",
    },
}


def get_backend(name: str) -> dict:
    """Return backend configuration by name.

    Raises KeyError if the backend is not registered.
    """
    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return BACKENDS[name]
