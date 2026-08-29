# QuantumNeuralNetworkOS

Variational quantum classifier for passive OS fingerprinting using PennyLane + PyTorch, designed to run on an NVIDIA GPU (developed on an RTX 3090) via Docker.

## Overview

This project builds a hybrid quantum-classical neural network that classifies operating systems from network packet features (nPrint format). It uses a 20-qubit variational quantum circuit with 4 trainable layers and data re-uploading, operating in a 2^20-dimensional Hilbert space (~1 million dimensions). By default each qubit is measured on all three Pauli axes (Z, X, Y), giving 60 expectation values that feed a small classical head; the best-performing configuration measures Z only (20 values) and adds the raw features to the head through a skip path.

Headline result: on identical data, features, and split, the quantum model **ties** an XGBoost classifier restricted to the same 20 features (0.365 vs 0.348 macro F1, 48.8% vs 49.8% accuracy over 12 OS versions). No quantum-advantage claim is made — see [Results](#results).

## Pipeline

```
nPrint CSV (962 cols)
    → Column removal (src_ip, IPv4 src/dst/identification, TCP ports, seq/ack)
    → SelectKBest (k=20, mutual information, fit on training split only) — or an explicit `--feature-names` list
    → Ternary RY encoding (20 qubits): x ∈ {-1, 0, 1} → RY((x+1)·π/2)
    → Ring CNOT
    → 4 variational layers with data re-uploading (RY encode + RY/RZ + entangler)
    → Pauli measurements per qubit: Z+X+Y (60 values, default) or Z only (20, `--measure-axes Z`)
    → [BatchNorm1d] → Linear(head_in, 128) → ReLU → Dropout → Linear(128, n_classes)
```

Training uses Adam with linear LR warmup, ReduceLROnPlateau scheduling, gradient clipping, class-weighted focal loss for imbalanced classes, and early stopping on validation F1. See [docs/HowItWorks.txt](docs/HowItWorks.txt) for a step-by-step walkthrough of the circuit and training loop, [docs/DECISIONS.txt](docs/DECISIONS.txt) for the dated log of modeling decisions (current design at the end), [docs/Benchmarks.txt](docs/Benchmarks.txt) for the verified comparison protocol, and [docs/TechStack_Plan.txt](docs/TechStack_Plan.txt) for the original planning document and early run history.

## Quantum Circuit Design

One qubit per selected feature (20 qubits). Full gate-level diagrams are in [circuits/circuit_map.txt](circuits/circuit_map.txt).

**Encoding.** nPrint features are ternary: `1` = bit set, `0` = bit clear, `-1` = the header field is absent from the packet (about 58% of selected-feature values). Each feature x is encoded as a single `RY((x+1)·π/2)` rotation, so `-1 → |0⟩`, `0 → |+⟩` (equator), `1 → |1⟩` — three distinct, measurable states. The naive `RY(x·π)` encoding is deliberately avoided: `RY(-π) = -RY(π)` up to global phase, which makes "field absent" and "bit set" physically indistinguishable everywhere downstream (a regression test guards this).

**Initial entanglement.** A ring of CNOTs (`i → i+1 mod 20`). Because `0`-valued features sit on the equator, this ring already creates genuine entanglement for any packet with a `0` feature.

**Variational layers (×4, data re-uploading).** Each layer re-encodes the input (`RY((x+1)·π/2)` on every qubit), applies trainable `RY(θ)` and `RZ(φ)` on every qubit, then an entangler. Re-uploading lets the circuit represent higher-order functions of the input rather than a first-order Fourier series. 2 trainable angles × 20 qubits × 4 layers = **160 quantum parameters**.

**Entangler (per layer, `--entangler`).**

| Option | Gates per layer | Trainable | Notes |
|---|---|---|---|
| `cnot` (default) | 20 CNOT ring | no | cheapest; weight-compatible with `ring_long` |
| `crz` | 20 CRZ ring | yes (+1 angle/qubit/layer, 240 total) | entanglement strength learned per pair |
| `ring_long` | 20 CNOT ring + 40 shortcut CNOTs (`i → i+5`, `i → i+10`) | no | shrinks the ring's graph diameter so distant features interact within one layer; used by the best checkpoint |

**Measurement (`--measure-axes`).** Default `ZXY`: ⟨Z⟩, ⟨X⟩, ⟨Y⟩ on every qubit — the full Bloch vector, 60 outputs. `Z`: 20 outputs. Adjoint differentiation re-runs its backward sweep once per observable, so Z-only roughly halves training time per epoch (~33 h → ~17.7 h on the development set) and is what the best checkpoint uses.

**Classical head.** `[BatchNorm1d] → Linear(head_in, 128) → ReLU → Dropout(p) → Linear(128, 12)`. `head_in` is the number of expectation values plus, with `--skip`, the 20 raw features (60, 80, 40, or 20 for `--no-quantum --skip`, the classical-only ablation). Every option is stored in the model's `.json`, so `test.py`, `eval_heldout.py`, and `--resume` rebuild the same network.

| Configuration | Gates | Depth | Quantum params | Classical params |
|---|---|---|---|---|
| Default (`cnot`, `ZXY`) | 360 (180 RY, 80 RZ, 100 CNOT) | 113 | 160 | 9,356 |
| Best checkpoint (`ring_long`, `Z`, `--skip --batchnorm`) | 520 (180 RY, 80 RZ, 260 CNOT) | 137 | 160 | 6,796 (+80 BatchNorm) |

Gate counts and depth are from `qml.specs` on the 20-qubit, 4-layer circuit.

## Dataset

The dataset is **not included** in this repository (the raw CSVs are multi-gigabyte and contain source IP addresses from packet captures).

**What was used for development.** The Monday "working hours" packet capture from CIC-IDS2017 (benign traffic only), split into one pcap per operating system and converted to [nPrint](https://nprint.github.io/) format: one row per packet, 961 ternary feature columns (`1`/`0`/`-1`, where `-1` marks a header field absent from that packet) plus a label column, 962 columns total. All experiments use `Monday-WorkingHours_50th.csv`, a 1-in-50 row subsample of the full capture: **107,642 packets** over **12 OS-version classes**, split 80/20 stratified (86,113 train / 21,529 held out, `random_state=42`). `data/labels.csv` shows the pcap-to-label mapping.

| Class | Share | Class | Share |
|---|---|---|---|
| windows-10 | 27.0% | ubuntu-16.4-64b | 5.0% |
| windows-10-pro | 23.2% | windows-7-pro | 5.0% |
| mac_mac-os-x | 10.4% | ubuntu-16.4-32b | 4.4% |
| windows-8.1 | 8.5% | ubuntu-ubuntu-server | 4.0% |
| ubuntu-14.4-64b | 3.8% | windows-vista | 3.6% |
| ubuntu-14.4-32b | 3.3% | ubuntu-web-server | 1.7% |

The classes are heavily imbalanced (~16× between the largest and smallest), and several are near-duplicates at the packet level — windows-10 vs windows-10-pro in particular — which is the central difficulty of the task.

**Preprocessing.** Network-identifier columns are dropped before anything else (`config/defaults.py: COLUMNS_TO_REMOVE`: the src_ip column, IPv4 source/destination addresses and identification field, TCP ports, sequence and acknowledgment numbers), leaving 783 leak-free columns, so the model cannot memorize which host a packet came from. Twenty of those are then chosen either by `SelectKBest(mutual_info_classif)` fitted on the training split only, or by an explicit `--feature-names` list; the best checkpoint uses the top 20 columns by gain from an XGBoost classifier fitted on the same leak-free data, which surfaces TCP window-size and control-flag bits that univariate selection missed.

**Bring your own data.** Any nPrint CSV with the same 962-column layout works: place it at `data/nprint.csv` (or pass any path). Feature selection, the label set, and the split are derived from your file at training time and stored in the model's `.json`.

## Quick Start

### Docker (Recommended)

```bash
# Prerequisites: NVIDIA driver + nvidia-container-toolkit; dataset at data/nprint.csv
docker compose up quantum-train
```

`docker compose up` trains the best-known configuration (`--entangler ring_long --measure-axes Z --skip --batchnorm --dropout 0.1`) with `--resume`, so the container picks up where it left off after a crash or daemon restart. The commented-out command in `docker-compose.yml` reproduces the published `model_xgb20` checkpoint with its explicit feature list. Running `scripts/train.py` directly with no flags uses the plainer defaults from the parameter table below.

### Local

```bash
pip install -r requirements.txt

# Train
python scripts/train.py data/nprint.csv model_out --backend lightning.gpu

# Watch a running training job (reads models/training_progress_*.json)
python scripts/monitor.py models --watch

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
| `shortcut_offsets` | `[5, 10]` | Long-range CNOT offsets for `ring_long` |
| `measure_axes` | `ZXY` | Pauli axes measured per qubit (`Z` = 20 outputs) |
| `skip` / `batchnorm` | off | Raw-feature skip path / BatchNorm1d on head inputs |
| `dropout`    | 0.3     | Dropout in the classical head                |
| `batch_size` | 512     | Training batch size                          |
| `lr`         | 0.005   | Adam learning rate (3-epoch linear warmup)   |
| `patience`   | 10      | Early stopping patience (validation F1)      |
| focal gamma  | 2.0     | Focal loss focusing parameter                |
| grad clip    | 0.75    | Max gradient norm                            |

## Results

All numbers below are on the same 21,529-packet stratified split of the development dataset (seed 42), scoring the same predictions at three label granularities. The XGBoost rows were refit on the identical split with `max_depth=10`. Full protocol, the methodology traps encountered (unstratified-vs-stratified splits, macro-vs-weighted F1), and reproduction commands are in [docs/Benchmarks.txt](docs/Benchmarks.txt).

| Model | 12 OS versions | 11 classes (win10 + win10-pro merged) | 3 OS families |
|---|---|---|---|
| **QNN `model_xgb20`** (20 qubits, same 20 features) | **48.8% / 0.365** | 65.5% / 0.414 | 99.4% / 0.986 |
| XGBoost, same 20 features | 49.8% / 0.348 | 66.6% / 0.394 | 99.5% / 0.987 |
| XGBoost, all 783 leak-free features | 63.8% / 0.655 | 79.5% / 0.691 | 99.6% / 0.990 |

Cells are accuracy / macro F1.

Per class (12-way), the QNN separates macOS (0.97 F1), Windows Vista (0.78), Windows 10 (0.68), Windows 8.1 (0.49), and the Ubuntu web server (0.45), but cannot tell windows-10 from windows-10-pro (0.03 for the latter) or the Ubuntu 14.04 variants apart (0.02–0.11). Cross-family confusion is essentially zero.

What this does and does not show:

- **Feature-matched, the QNN ties XGBoost** (+0.017 macro F1, −1.0 pt accuracy). Its slight macro-F1 edge is plausibly from the class-weighted focal loss rather than the circuit; the XGBoost baseline was trained unweighted.
- **The 20-feature bottleneck is the binding constraint**, not the classifier: XGBoost with all 783 columns nearly doubles macro F1. Most version-level signal lies outside the top 20 features.
- **Family-level accuracy is a property of the features, not the circuit**: every model sits at 99.4–99.6%.
- **The circuit's marginal contribution is unmeasured.** The hybrid model also feeds raw features to its head via `--skip`; the built-in ablation (`--no-quantum --skip`, identical head/features/loss, no circuit) has not yet been run.

### Shipped checkpoints

Three trained models are included in `models/` (`.pt` weights + `.json` config, ~30–55 KB each). All use the ternary encoding, 20 qubits, 4 layers, `--skip --batchnorm --dropout 0.1`, and were trained on the development dataset described above.

| Checkpoint | Entangler | Measurement | Feature basis | Note |
|---|---|---|---|---|
| `model_xgb20` | `ring_long` | Z (20) | XGBoost gain top-20 | best; the model scored above (epoch 3, validation weighted F1 0.412) |
| `model_ringlong` | `ring_long` | Z (20) | SelectKBest (MI) | same circuit, univariate feature basis |
| `model_ternary` | `cnot` | Z+X+Y (60) | SelectKBest (MI) | first ternary-encoding run; full-Bloch measurement |

Because the configs store positional feature indices into the 783-column post-removal frame, evaluating a checkpoint requires an nPrint CSV with the same column layout as the training data.

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
- Simulation cost: one epoch on the ~86K-packet training split takes ~17.7 h (Z-only) to ~33 h (Z+X+Y) on an RTX 3090, so results are single-seed.

## Project Structure

```
QuantumNeuralNetworkOS/
├── config/          # Constants, hyperparams, backend registry
├── data/            # nPrint CSV pipeline
├── circuits/        # Custom QNode definition + circuit diagrams
├── training/        # Hybrid model + training loop
├── evaluation/      # Metrics, feature importance, circuit-boundary snapshots
├── scripts/         # CLI entrypoints (train, test, eval_heldout, inspect_model, monitor)
├── tests/           # Unit + integration tests
├── models/          # Shipped checkpoints (.pt + .json)
└── docs/            # Design decisions, circuit walkthrough, benchmarks
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
