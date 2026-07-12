# QuantumNeuralNetworkOS

Variational quantum classifier for passive OS fingerprinting using PennyLane + PyTorch, designed to run on an NVIDIA GPU (developed on an RTX 3090) via Docker.

## Overview

This project builds a hybrid quantum-classical neural network that classifies operating systems from network packet features (nPrint format). It uses a 20-qubit variational quantum circuit with 4 trainable layers and data re-uploading, operating in a 2^20-dimensional Hilbert space (~1 million dimensions). Each qubit is measured on all three Pauli axes (Z, X, Y), giving 60 expectation values that feed a small classical head.

### Architecture

```
nPrint CSV (962 cols)
    → Column removal (IPv4 src/dst, TCP ports, seq/ack)
    → SelectKBest (k=20, mutual information, fit on training split only)
    → RY encoding (20 qubits) + ring CNOT (classical XOR on the basis state; not yet entangling)
    → 4 variational layers with data re-uploading (RY encode + RY/RZ + ring CNOT; entanglement begins here)
    → PauliZ + PauliX + PauliY measurements (60 expectation values)
    → Linear(60, 128) → ReLU → Dropout(0.3) → Linear(128, n_classes)
    → Classification
```

Training uses Adam with linear LR warmup, ReduceLROnPlateau scheduling, gradient clipping, class-weighted focal loss for imbalanced classes, and early stopping on validation F1. See [docs/HowItWorks.txt](docs/HowItWorks.txt) for a step-by-step walkthrough of the circuit and training loop, [docs/TechStack_Plan.txt](docs/TechStack_Plan.txt) for design rationale, and [docs/DECISIONS.txt](docs/DECISIONS.txt) for a dated log of modeling decisions.

## Dataset

The dataset is **not included** in this repository (the raw CSVs are multi-gigabyte and contain source IP addresses from packet captures).

To train, you need packet captures converted to [nPrint](https://nprint.github.io/) format: one row per packet, 961 binary feature columns plus a final label column (the OS class), 962 columns total. Place your CSV at `data/nprint.csv` (or pass any path to the scripts). `data/labels.csv` shows the pcap-to-label mapping used during development.

## Quick Start

### Docker (Recommended)

```bash
# Prerequisites: NVIDIA driver + nvidia-container-toolkit
docker compose up quantum-train
```

### Local

```bash
pip install -r requirements.txt

# Train
python scripts/train.py data/nprint.csv model_out --backend lightning.gpu

# Evaluate
python scripts/test.py data/nprint.csv models/model_out --backend lightning.gpu

# Unit tests
pytest tests/ -v
```

### Backends

| Backend           | Use Case                    |
|-------------------|-----------------------------|
| `default.qubit`   | Quick prototyping (CPU)     |
| `lightning.qubit` | Fast CPU simulation         |
| `lightning.gpu`   | GPU production              |

## Key Parameters

Defaults live in `config/defaults.py`; most can be overridden via CLI flags on `scripts/train.py`.

| Parameter    | Default | Description                                  |
|--------------|---------|----------------------------------------------|
| `n_qubits`   | 20      | Number of qubits (= selected features)       |
| `n_layers`   | 4       | Variational layer count                      |
| `batch_size` | 512     | Training batch size                          |
| `lr`         | 0.005   | Adam learning rate (3-epoch linear warmup)   |
| `patience`   | 10      | Early stopping patience (validation F1)      |
| focal gamma  | 2.0     | Focal loss focusing parameter                |
| grad clip    | 0.75    | Max gradient norm                            |

## Comparison with a Gradient-Free Baseline

Earlier iterations of this work used a Qiskit variational classifier trained with COBYLA. This project's design differs:

|                  | Qiskit + COBYLA baseline | PennyLane (this project)          |
|------------------|--------------------------|-----------------------------------|
| Optimizer        | COBYLA (gradient-free)   | Adam (gradient-based)             |
| Batching         | Full dataset             | Mini-batches (DataLoader)         |
| Gradients        | None                     | Adjoint differentiation (GPU)     |
| Features/Qubits  | 4                        | 20                                |
| Hilbert space    | 16 dimensions            | ~1 million dimensions             |

## Known Limitations

- The held-out test split is also used as the validation set for early stopping, LR scheduling, and best-checkpoint selection, so reported test metrics are optimistically biased. A three-way train/val/test split is future work.

## Project Structure

```
QuantumNeuralNetworkOS/
├── config/          # Constants, hyperparams, backend registry
├── data/            # nPrint CSV pipeline
├── circuits/        # Custom QNode definition
├── training/        # Hybrid model + training loop
├── evaluation/      # Metrics (accuracy, F1, reports)
├── scripts/         # CLI entrypoints (train.py, test.py, monitor.py)
├── tests/           # Unit + integration tests
├── models/          # Saved checkpoints (.pt + .json)
└── docs/            # Design notes and circuit walkthrough
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
