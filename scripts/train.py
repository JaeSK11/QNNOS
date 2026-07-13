#!/usr/bin/env python3
"""CLI entrypoint: train the variational quantum classifier."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.defaults import (
    BATCH_SIZE,
    DEFAULT_BACKEND,
    DROPOUT,
    ENTANGLER,
    LEARNING_RATE,
    N_FEATURES,
    N_LAYERS,
    N_QUBITS,
    PATIENCE,
    USE_BATCHNORM,
    USE_SKIP,
)
from data.pipeline import create_dataloaders, prepare_data
from evaluation.inspection import feature_importance_ranking, format_ranking
from evaluation.metrics import full_report
from training.hybrid_model import HybridQuantumNet
from training.trainer import load_resume_checkpoint, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PennyLane variational quantum classifier for OS fingerprinting"
    )
    parser.add_argument("csv_path", help="Path to nPrint CSV file")
    parser.add_argument("model_name", help="Name for saved model")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Quantum backend")
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--n-features", type=int, default=N_FEATURES)
    parser.add_argument(
        "--entangler",
        choices=["cnot", "crz"],
        default=ENTANGLER,
        help="Per-layer entangler: 'cnot' (fixed ring) or 'crz' (trainable). "
        "A 'crz' model is not weight-compatible with a 'cnot' checkpoint.",
    )
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Head dropout probability")
    parser.add_argument(
        "--skip", action=argparse.BooleanOptionalAction, default=USE_SKIP,
        help="Concat raw features with quantum outputs (residual path).",
    )
    parser.add_argument(
        "--batchnorm", action=argparse.BooleanOptionalAction, default=USE_BATCHNORM,
        help="BatchNorm1d on head inputs (normalizes per-feature scale).",
    )
    parser.add_argument(
        "--quantum", action=argparse.BooleanOptionalAction, default=True,
        help="Use the quantum circuit. --no-quantum with --skip = classical baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previous run: reuse the saved config's feature columns "
        "and label mapping, restore the latest training state if present, else "
        "warm-start from the best-weights checkpoint.",
    )
    return parser.parse_args()


def load_saved_config(save_dir: str, model_name: str) -> dict:
    """Load the config JSON written by a previous run (required for --resume)."""
    path = Path(save_dir) / f"{model_name}.json"
    if not path.exists():
        raise SystemExit(f"--resume set but no config found at {path}; nothing to resume.")
    with open(path) as f:
        return json.load(f)


def latest_best(save_dir: str, model_name: str) -> tuple[float, int]:
    """Best F1 / epoch from this model's newest progress file, to seed a warm
    start. Scoped to model_name so a different model's run in the same dir can't
    seed an inflated best_f1 that would block the resumed run from checkpointing.
    """
    files = sorted(Path(save_dir).glob(f"training_progress_{model_name}_*.json"))
    if not files:
        return 0.0, 0
    try:
        with open(files[-1]) as f:
            p = json.load(f)
        return float(p.get("best_f1", 0.0)), int(p.get("best_epoch", 0))
    except (json.JSONDecodeError, OSError):
        return 0.0, 0


def main() -> None:
    import time

    args = parse_args()

    if not args.quantum and not args.skip:
        raise SystemExit("--no-quantum requires --skip (otherwise the model has no inputs).")

    saved_config = None
    feature_indices = None
    # Resume-or-fresh: with --resume but no prior checkpoint (the very first
    # launch), start fresh instead of erroring. This lets the container command
    # carry --resume permanently so every auto-restart continues the run, while
    # the first launch bootstraps it.
    if args.resume and not (Path(args.save_dir) / f"{args.model_name}.json").exists():
        print(
            f"--resume set but no checkpoint at {args.save_dir}/{args.model_name}.json; "
            f"starting a fresh run.",
            flush=True,
        )
        args.resume = False
    if args.resume:
        saved_config = load_saved_config(args.save_dir, args.model_name)
        # Lock architecture and feature basis to the checkpoint being resumed.
        args.n_qubits = saved_config["n_qubits"]
        args.n_layers = saved_config["n_layers"]
        args.n_features = saved_config["n_features"]
        args.entangler = saved_config.get("entangler", "cnot")
        args.quantum = saved_config.get("use_quantum", True)
        args.skip = saved_config.get("use_skip", False)
        args.batchnorm = saved_config.get("use_batchnorm", False)
        args.dropout = saved_config.get("dropout", 0.3)
        feature_indices = saved_config["feature_indices"]
        print("Resuming: reusing saved feature columns, label mapping, and architecture.", flush=True)

    t0 = time.time()
    print(f"Loading data from {args.csv_path}...", flush=True)
    data = prepare_data(args.csv_path, k=args.n_features, feature_indices=feature_indices)
    data_elapsed = time.time() - t0

    print(f"Data pipeline: {data_elapsed:.1f}s", flush=True)
    print(f"Classes: {list(data['label_encoder'].classes_)}", flush=True)
    print(f"Training samples: {len(data['X_train'])}", flush=True)
    print(f"Test samples: {len(data['X_test'])}", flush=True)
    print(f"Features selected: {data['feature_indices']}", flush=True)

    # Feature-importance ranking (mutual information). On resume the scores were
    # not recomputed (indices were reused), so fall back to the saved config's.
    feature_scores = data.get("feature_scores")
    if feature_scores is None and saved_config is not None:
        feature_scores = saved_config.get("feature_scores")
    ranking = feature_importance_ranking(data["feature_indices"], feature_scores)
    print("Feature importance (mutual information, top 10):", flush=True)
    print(format_ranking(ranking, top=10), flush=True)

    # Class distribution in training set
    class_counts = torch.bincount(data["y_train"], minlength=data["n_classes"])
    for i, (cls, cnt) in enumerate(zip(data["label_encoder"].classes_, class_counts)):
        print(f"  {cls}: {cnt} ({cnt / len(data['y_train']) * 100:.1f}%)", flush=True)

    train_loader, test_loader = create_dataloaders(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        batch_size=args.batch_size,
    )

    print(f"Batches per epoch: {len(train_loader)} train, {len(test_loader)} test", flush=True)
    print(f"Building model: {args.n_qubits} qubits, {args.n_layers} layers, "
          f"entangler={args.entangler}, quantum={args.quantum}, skip={args.skip}, "
          f"batchnorm={args.batchnorm}, dropout={args.dropout}, backend={args.backend}", flush=True)
    t1 = time.time()
    model = HybridQuantumNet(
        n_classes=data["n_classes"],
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        backend=args.backend,
        entangler=args.entangler,
        use_quantum=args.quantum,
        use_skip=args.skip,
        use_batchnorm=args.batchnorm,
        dropout=args.dropout,
    )
    build_elapsed = time.time() - t1
    q_params = (
        sum(p.numel() for p in model.quantum_layer.parameters())
        if model.quantum_layer is not None else 0
    )
    c_params = sum(p.numel() for p in model.classical_head.parameters())
    print(f"Model built: {build_elapsed:.1f}s", flush=True)
    print(f"Parameters: {q_params} quantum + {c_params} classical = {q_params + c_params} total", flush=True)

    # Reuse the original config verbatim when resuming so nothing drifts.
    config = saved_config if args.resume else {
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "n_features": args.n_features,
        "n_classes": data["n_classes"],
        "feature_indices": data["feature_indices"],
        "feature_scores": data.get("feature_scores"),
        "label_classes": list(data["label_encoder"].classes_),
        "backend": args.backend,
        "entangler": args.entangler,
        "use_quantum": args.quantum,
        "use_skip": args.skip,
        "use_batchnorm": args.batchnorm,
        "dropout": args.dropout,
    }

    resume_state = None
    if args.resume:
        # Guard: the label mapping must match the checkpoint, or class indices
        # would silently shift under the trained head.
        if list(data["label_encoder"].classes_) != saved_config["label_classes"]:
            raise SystemExit(
                "Label classes in this CSV do not match the checkpoint's "
                "label_classes; cannot resume against different data."
            )
        resume_state = load_resume_checkpoint(args.save_dir, args.model_name)
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            print(f"Loaded full resume state; continuing from epoch {resume_state['epoch']}.", flush=True)
        else:
            # No full resume checkpoint. Either the run predates resume support
            # (warm-start from best weights), or it died before the first
            # mid-epoch checkpoint was written (no weights yet -> start fresh
            # rather than crash-loop the auto-restart).
            best_path = Path(args.save_dir) / f"{args.model_name}.pt"
            if not best_path.exists():
                print(
                    "--resume set but no resume-state or best-weights checkpoint yet "
                    "(crash before the first mid-epoch save); starting fresh from "
                    "epoch 0 with the saved feature/label basis.",
                    flush=True,
                )
                resume_state = None
            else:
                model.load_state_dict(torch.load(best_path, weights_only=True))
                seed_f1, seed_epoch = latest_best(args.save_dir, args.model_name)
                resume_state = {
                    "optimizer": None,
                    "scheduler": None,
                    "epoch": 0,
                    "best_f1": seed_f1,
                    "best_epoch": seed_epoch,
                    "epochs_without_improvement": 0,
                }
                print(
                    f"Warm-starting from best weights (seed best_f1={seed_f1:.4f} "
                    f"@ epoch {seed_epoch}); optimizer/scheduler reset, LR warmup re-runs.",
                    flush=True,
                )

    print("Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_classes=data["n_classes"],
        y_train=data["y_train"],
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_dir=args.save_dir,
        model_name=args.model_name,
        config=config,
        resume_state=resume_state,
    )

    print(f"\nBest epoch: {history['best_epoch']}, Best validation F1: {history['best_f1']:.4f}")

    # Reload the best checkpoint before the final report. train_model leaves the
    # in-memory model at the LAST epoch's weights (up to `patience` epochs past
    # the best after early stopping), and save_checkpoint only persisted the
    # best weights to disk — so evaluating the in-memory model would describe a
    # different, worse network than the "Best validation F1" line above.
    best_path = Path(args.save_dir) / f"{args.model_name}.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print(f"Reloaded best checkpoint (epoch {history['best_epoch']}) for final report.")
    else:
        print("No saved checkpoint found; reporting last-epoch weights.")

    # Final evaluation on the validation split. NOTE: this split doubles as the
    # early-stopping / checkpoint-selection set, so these numbers are validation
    # (dev) metrics and are optimistically biased — they are NOT an unbiased
    # held-out test estimate. See "Known Limitations" in the README.
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y_batch)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    report = full_report(
        all_labels, all_preds, class_names=list(data["label_encoder"].classes_)
    )
    print("\n[validation split — optimistically biased, not a held-out test score]")
    print(f"Validation accuracy: {report['accuracy']:.4f}")
    print(f"Validation F1 (weighted): {report['f1_weighted']:.4f}")
    print(f"\n{report['classification_report']}")


if __name__ == "__main__":
    main()
