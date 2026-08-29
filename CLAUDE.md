# QuantumNeuralNetworkOS

Variational quantum classifier for passive OS fingerprinting using PennyLane + PyTorch on NVIDIA RTX 3090.

## Architecture

- **Data pipeline** (classical): nPrint CSV (962 cols) -> column removal -> SelectKBest(k=20) -> ternary tensors (values -1/0/1; -1 = nPrint "field absent", ~58% of selected-feature values)
- **Quantum circuit** (custom QNode): ternary RY encoding (20 qubits) -> ring CNOT (entangles wherever a feature is 0, since 0 maps to the equator) -> 4 variational layers, each re-uploading the inputs then RY+RZ+ring CNOT (data re-uploading) -> PauliZ+PauliX+PauliY measurements (60 outputs)
- **Hybrid model**: TorchLayer(QNode) -> [optional BatchNorm1d] -> Linear(head_in,128) -> ReLU -> Dropout(p) -> Linear(128, n_classes). Head input is quantum expvals (60) and/or raw features (20) via `--skip`. Flags: `--skip/--no-skip`, `--batchnorm`, `--dropout`, `--quantum/--no-quantum` (`--no-quantum --skip` = classical baseline for A/B). All persisted in the model config so `test.py`/resume rebuild the right head.
- **Training**: Adam optimizer, FocalLoss (class-weighted, gamma=2.0), early stopping on val F1, adjoint differentiation

## Key Design Decisions

- 20 qubits = 20 features (one per qubit), operating in 2^20 (~1M) dimensional Hilbert space, 60 measurement outputs (Z+X+Y per qubit)
- No normalization needed: nPrint features are ternary (-1/0/1), encoded as RY((feature + 1) * pi/2) mapping -1 -> |0>, 0 -> equator, 1 -> |1>. Do NOT revert to RY(feature * pi): it aliases -1 with +1 up to global phase (RY(-pi) = -RY(pi)), making "field absent" and "bit set" physically indistinguishable — see docs/DECISIONS.txt Decision 6 and tests/test_circuits.py::test_circuit_ternary_encoding_distinguishes_absent_from_set. Old-encoding checkpoints require a fresh training run.
- Entangler is configurable (`config/defaults.py: ENTANGLER`, or `--entangler`): `cnot` = fixed ring (160 quantum params); `crz` = trainable controlled-RZ ring, entanglement strength learned per pair (240 params); `ring_long` = CNOT ring plus fixed long-range CNOT shortcuts at `SHORTCUT_OFFSETS` / `--shortcut-offsets` (default [5, 10]; same weight shapes as `cnot`). `crz` checkpoints are NOT weight-compatible with `cnot`/`ring_long`, so switching needs a fresh run.
- Measured Pauli axes are configurable (`MEASURE_AXES`, or `--measure-axes`): `ZXY` = 60 outputs (default), `Z` = 20 outputs. Adjoint-diff cost scales with observable count, so Z-only roughly halves epoch time (docs/DECISIONS.txt Decision 8). Stored in the model config; configs without the key rebuild as ZXY.
- Results and the classical-baseline protocol live in docs/Benchmarks.txt (canonical). The trainer's logged Val_F1 is *weighted* F1; `scripts/eval_heldout.py` reports macro F1 on the held-out split. `scripts/test.py` scores every row of the CSV it is given, so it is contaminated when run on the training CSV.
- Column removal indices match OsirisML (IPv4 src/dst, TCP ports, seq/ack numbers)
- Feature selection fitted on training set only (no data leakage)
- Backend abstraction: default.qubit (CPU debug), lightning.qubit (CPU fast), lightning.gpu (GPU production)

## Commands

```bash
# Train
python scripts/train.py data/nprint.csv model_out --backend lightning.gpu

# Held-out evaluation (reproduces the training split; scores only the 20% held out)
python scripts/eval_heldout.py data/nprint.csv models/model_out --backend lightning.gpu --merge windows-10,windows-10-pro

# Test on a separate, unseen CSV (scores every row)
python scripts/test.py data/other.csv models/model_out --backend lightning.gpu

# Inspect: feature-importance ranking + pre/post-circuit data snapshots
python scripts/inspect_model.py data/nprint.csv models/model_out --backend lightning.qubit --n-samples 5

# Unit tests
pytest tests/

# Docker
docker compose up quantum-train
```

## File Layout

- `config/defaults.py` - all constants, column removal ranges, hyperparams
- `config/backends.py` - backend registry (device name + diff_method)
- `data/pipeline.py` - nPrint CSV ingestion, column removal, feature selection, DataLoader creation
- `circuits/model.py` - custom QNode definition, `create_circuit()` factory
- `training/hybrid_model.py` - `HybridQuantumNet` (nn.Module wrapping TorchLayer + classical head)
- `training/trainer.py` - training loop with early stopping, checkpointing
- `evaluation/metrics.py` - accuracy, F1, classification_report, confusion_matrix
- `evaluation/inspection.py` - feature-importance ranking + pre/post-circuit data snapshots
- `scripts/train.py` - CLI entrypoint for training (prints feature-importance ranking; saves `feature_scores` in config)
- `scripts/test.py` - CLI entrypoint for evaluation of every row in a CSV
- `scripts/eval_heldout.py` - CLI: held-out-split evaluation with optional `--merge` and OS-family views (writes `<model>_heldout.{txt,npz}`)
- `scripts/inspect_model.py` - CLI: feature importance + circuit-boundary snapshots (writes `<model>_inspection.{json,txt}`). Named `inspect_model` (not `inspect`) so running it as `python scripts/inspect_model.py` cannot shadow the stdlib `inspect` module on sys.path.

## Testing

- `tests/test_data_pipeline.py` - CSV loading, column removal, feature selection
- `tests/test_circuits.py` - circuit construction, output shape, gate count
- `tests/test_hybrid_model.py` - forward pass, gradient flow
- `tests/test_inspection.py` - feature-importance ranking, pre/post-circuit snapshots
- `tests/test_training.py` - training loop, early stopping, checkpointing
- `tests/test_serialization.py` - save/load model + config
- `tests/test_integration.py` - synthetic data through full pipeline
