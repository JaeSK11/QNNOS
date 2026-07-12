# QuantumNeuralNetworkOS

Variational quantum classifier for passive OS fingerprinting using PennyLane + PyTorch on NVIDIA RTX 3090.

## Architecture

- **Data pipeline** (classical): nPrint CSV (962 cols) -> column removal -> SelectKBest(k=20) -> binary tensors
- **Quantum circuit** (custom QNode): RY encoding (20 qubits) -> ring CNOT (classical XOR on the basis state, not yet entangling) -> 4 variational layers, each re-uploading the inputs then RY+RZ+ring CNOT (data re-uploading; entanglement begins here) -> PauliZ+PauliX+PauliY measurements (60 outputs)
- **Hybrid model**: TorchLayer(QNode) -> Linear(60,128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes)
- **Training**: Adam optimizer, FocalLoss (class-weighted, gamma=2.0), early stopping on val F1, adjoint differentiation

## Key Design Decisions

- 20 qubits = 20 features (one per qubit), operating in 2^20 (~1M) dimensional Hilbert space, 60 measurement outputs (Z+X+Y per qubit)
- No normalization needed: nPrint features are binary (0/1), encoded as RY(feature * pi)
- Column removal indices match OsirisML (IPv4 src/dst, TCP ports, seq/ack numbers)
- Feature selection fitted on training set only (no data leakage)
- Backend abstraction: default.qubit (CPU debug), lightning.qubit (CPU fast), lightning.gpu (GPU production)

## Commands

```bash
# Train
python scripts/train.py data/nprint.csv model_out --backend lightning.gpu

# Test
python scripts/test.py data/nprint.csv models/model_out --backend lightning.gpu

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
- `scripts/train.py` - CLI entrypoint for training
- `scripts/test.py` - CLI entrypoint for evaluation

## Testing

- `tests/test_data_pipeline.py` - CSV loading, column removal, feature selection
- `tests/test_circuits.py` - circuit construction, output shape, gate count
- `tests/test_hybrid_model.py` - forward pass, gradient flow
- `tests/test_training.py` - training loop, early stopping, checkpointing
- `tests/test_serialization.py` - save/load model + config
- `tests/test_integration.py` - synthetic data through full pipeline
