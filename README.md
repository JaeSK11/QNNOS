# QuantumNeuralNetworkOS

Variational quantum classifier for passive OS fingerprinting using PennyLane + PyTorch, designed to run on an NVIDIA GPU (developed on an RTX 3090) via Docker.

## Overview

This project builds a hybrid quantum-classical neural network that classifies operating systems from network packet features (nPrint format). It uses a 20-qubit variational quantum circuit with 4 trainable layers and data re-uploading, operating in a 2^20-dimensional Hilbert space (~1 million dimensions). Each qubit is measured on all three Pauli axes (Z, X, Y), giving 60 expectation values that feed a small classical head.

### Architecture

```
nPrint CSV (962 cols)
    → Column removal (src_ip, IPv4 src/dst/identification, TCP ports, seq/ack)
    → SelectKBest (k=20, mutual information, fit on training split only) — or an explicit `--feature-names` list
    → Ternary RY encoding (20 qubits): x ∈ {-1, 0, 1} → RY((x+1)·π/2), so "field absent" (-1), 0 and 1 land on distinct Bloch latitudes
    → Ring CNOT (entangles wherever a feature is 0, i.e. sits on the equator)
    → 4 variational layers with data re-uploading (RY encode + RY/RZ + entangler)
    → Pauli measurements per qubit: Z+X+Y (60 values, default) or Z only (20, `--measure-axes Z`)
    → [BatchNorm1d] → Linear(head_in, 128) → ReLU → Dropout → Linear(128, n_classes)
```

The entangler after each layer is configurable: `--entangler cnot` (fixed CNOT ring, default), `crz` (trainable controlled-RZ ring, entanglement strength learned per pair), or `ring_long` (CNOT ring plus fixed long-range shortcuts at `--shortcut-offsets 5 10`, so distant qubits interact within a layer). `--skip` concatenates the 20 raw features into the head input alongside the quantum outputs, and `--no-quantum --skip` trains the identical head on raw features alone — the classical ablation. Every option is stored in the model's `.json`, so `test.py`, `eval_heldout.py`, and `--resume` rebuild the same network.

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

# Evaluate on the held-out split of the training CSV (reproduces the split; scores only the 20% held out)
python scripts/eval_heldout.py data/nprint.csv models/model_out --backend lightning.gpu

# Evaluate every row of a separate, unseen CSV
python scripts/test.py data/other.csv models/model_out --backend lightning.gpu

# Inspect: feature-importance ranking + pre/post-circuit data snapshots
python scripts/inspect_model.py data/nprint.csv models/model_out --backend lightning.qubit --n-samples 5

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
| `entangler`  | `cnot`  | `cnot`, `crz`, or `ring_long` (see above)    |
| `measure_axes` | `ZXY` | Pauli axes measured per qubit (`Z` = 20 outputs) |
| `dropout`    | 0.3     | Dropout in the classical head                |
| `batch_size` | 512     | Training batch size                          |
| `lr`         | 0.005   | Adam learning rate (3-epoch linear warmup)   |
| `patience`   | 10      | Early stopping patience (validation F1)      |
| focal gamma  | 2.0     | Focal loss focusing parameter                |
| grad clip    | 0.75    | Max gradient norm                            |

## Results

On the development dataset (12 OS-version classes, ~108K packets, stratified 20% split), the best checkpoint (`models/model_xgb20`: `ring_long` entangler, Z-only measurement, `--skip --batchnorm --dropout 0.1`, XGBoost-ranked feature basis) reaches **48.8% accuracy / 0.365 macro F1** at 12 classes — a tie with XGBoost trained on the same 20 features (49.8% / 0.348). Unconstrained XGBoost on all 783 leak-free columns reaches 63.8% / 0.655, so the 20-feature bottleneck, not the classifier, is the binding constraint, and no quantum-advantage claim is made. The full protocol, numbers at merged label granularities, and the methodology traps encountered are in [docs/Benchmarks.txt](docs/Benchmarks.txt).

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
- `scripts/test.py` scores every row of the CSV it is given. Pointing it at the training CSV mixes in the ~80% of rows the model trained on; use `scripts/eval_heldout.py` for that case, which reproduces the training split and scores only the held-out rows.
- The trainer's logged `Val_F1` is *weighted* F1. Macro F1 (the fairer number under class imbalance) is reported by `scripts/eval_heldout.py`.

## Project Structure

```
QuantumNeuralNetworkOS/
├── config/          # Constants, hyperparams, backend registry
├── data/            # nPrint CSV pipeline
├── circuits/        # Custom QNode definition
├── training/        # Hybrid model + training loop
├── evaluation/      # Metrics, feature importance, circuit-boundary snapshots
├── scripts/         # CLI entrypoints (train, test, eval_heldout, inspect_model, monitor)
├── tests/           # Unit + integration tests
├── models/          # Saved checkpoints (.pt + .json)
└── docs/            # Design decisions, circuit walkthrough, benchmarks
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
