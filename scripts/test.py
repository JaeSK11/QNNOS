#!/usr/bin/env python3
"""CLI entrypoint: evaluate a saved variational quantum classifier."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from config.defaults import DEFAULT_BACKEND
from data.pipeline import prepare_eval_data
from evaluation.metrics import full_report
from training.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PennyLane quantum classifier"
    )
    parser.add_argument("csv_path", help="Path to nPrint CSV file")
    parser.add_argument("model_path", help="Path to model directory (without extension)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Quantum backend")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = str(Path(args.model_path).parent)
    model_name = Path(args.model_path).name

    print(f"Loading model from {model_dir}/{model_name}...")
    model, config = load_checkpoint(model_dir, model_name, backend=args.backend)

    if "feature_indices" not in config or "label_classes" not in config:
        raise SystemExit(
            "Model config is missing 'feature_indices'/'label_classes'; this "
            "checkpoint predates the reproducible-eval fix and cannot be "
            "evaluated without them. Retrain to regenerate the config."
        )

    print(f"Loading data from {args.csv_path}...")
    # Reuse the exact feature columns and label mapping the model was trained on
    # (no re-fit of feature selection or label encoding), so evaluation matches
    # training regardless of this CSV's own class distribution.
    data = prepare_eval_data(
        args.csv_path,
        feature_indices=config["feature_indices"],
        label_classes=config["label_classes"],
    )

    eval_ds = TensorDataset(data["X_eval"], data["y_eval"])
    test_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating on {len(eval_ds)} samples...")
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

    class_names = config.get("label_classes", None)
    report = full_report(all_labels, all_preds, class_names=class_names)

    print(f"\nAccuracy: {report['accuracy']:.4f}")
    print(f"F1 (weighted): {report['f1_weighted']:.4f}")
    print(f"\n{report['classification_report']}")


if __name__ == "__main__":
    main()
