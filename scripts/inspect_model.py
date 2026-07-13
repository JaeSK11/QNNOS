#!/usr/bin/env python3
"""CLI: inspect a trained model's feature importance and data at circuit boundaries.

Produces three artifacts for a handful of samples:
  1. Feature-importance ranking (mutual information) of the model's features.
  2. PRE-circuit snapshot  -- the processed binary vectors entering the QNode.
  3. POST-circuit snapshot -- the 60 Pauli expvals entering the classical head.

Writes <model_name>_inspection.{json,txt} next to the model and prints a summary.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.defaults import DEFAULT_BACKEND
from data.pipeline import prepare_eval_data
from evaluation.inspection import (
    compute_mi_ranking,
    feature_importance_ranking,
    format_ranking,
    post_circuit_snapshot,
    pre_circuit_snapshot,
)
from training.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a trained quantum classifier")
    parser.add_argument("csv_path", help="Path to nPrint CSV file")
    parser.add_argument("model_path", help="Path to model dir/name (without extension)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Quantum backend")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples to snapshot")
    parser.add_argument("--top", type=int, default=20, help="Top-N features to print")
    parser.add_argument("--out", default=None, help="Output dir (default: model dir)")
    return parser.parse_args()


def _fmt_row(vals, width=6):
    return " ".join(f"{v:>{width}.3f}" for v in vals)


def main() -> None:
    args = parse_args()
    model_dir = str(Path(args.model_path).parent)
    model_name = Path(args.model_path).name
    out_dir = Path(args.out) if args.out else Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}/{model_name}...", flush=True)
    model, config = load_checkpoint(model_dir, model_name, backend=args.backend)

    if "feature_indices" not in config or "label_classes" not in config:
        raise SystemExit("Model config lacks feature_indices/label_classes; cannot inspect.")

    feature_indices = config["feature_indices"]
    label_classes = config["label_classes"]

    print(f"Loading data from {args.csv_path}...", flush=True)
    data = prepare_eval_data(args.csv_path, feature_indices, label_classes)
    X, y = data["X_eval"], data["y_eval"]

    # 1. Feature importance. Prefer the training-time scores saved in the config
    # (same estimator selection used); else compute a fresh MI on this data.
    if config.get("feature_scores"):
        ranking = feature_importance_ranking(feature_indices, config["feature_scores"])
        importance_source = "training-time (config)"
    else:
        ranking = compute_mi_ranking(X.numpy(), y.numpy(), feature_indices)
        importance_source = "recomputed on this CSV"

    # 2 & 3. Snapshots at the circuit boundaries.
    pre = pre_circuit_snapshot(X, y, feature_indices, label_classes, args.n_samples)
    post = post_circuit_snapshot(model, X, y, label_classes, args.n_samples)

    # ---- Print summary ----
    print(f"\n=== FEATURE IMPORTANCE (mutual information, {importance_source}) ===")
    print(format_ranking(ranking, top=args.top))

    print(f"\n=== PRE-CIRCUIT SNAPSHOT ({len(pre)} samples, {len(feature_indices)} features) ===")
    print("Processed binary features fed to the QNode (RY(feature * pi) per qubit):")
    for s in pre:
        bits = "".join(str(int(v)) for v in s["vector"])
        print(f"  sample {s['sample']} [{s['label']}]: {bits}")

    print(f"\n=== POST-CIRCUIT SNAPSHOT ({args.n_samples} samples) ===")
    if post is None:
        print("  (model has no quantum layer -- --no-quantum baseline; nothing to show)")
    else:
        print("60 Pauli expectation values in [-1, 1] entering the classical head:")
        for s in post:
            print(f"  sample {s['sample']} [{s['label']}]")
            print(f"    Z: {_fmt_row(s['Z'])}")
            print(f"    X: {_fmt_row(s['X'])}")
            print(f"    Y: {_fmt_row(s['Y'])}")

    # ---- Persist artifacts ----
    payload = {
        "model": f"{model_dir}/{model_name}",
        "csv": args.csv_path,
        "feature_importance_source": importance_source,
        "feature_importance": ranking,
        "pre_circuit": pre,
        "post_circuit": post,  # None for classical baseline
        "has_quantum": post is not None,
    }
    json_path = out_dir / f"{model_name}_inspection.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = out_dir / f"{model_name}_inspection.txt"
    with open(txt_path, "w") as f:
        f.write(f"Inspection of {model_dir}/{model_name} on {args.csv_path}\n\n")
        f.write("FEATURE IMPORTANCE (mutual information)\n")
        f.write(format_ranking(ranking) + "\n\n")
        f.write("PRE-CIRCUIT (processed features entering the QNode)\n")
        for s in pre:
            bits = "".join(str(int(v)) for v in s["vector"])
            f.write(f"  sample {s['sample']} [{s['label']}]: {bits}\n")
        f.write("\nPOST-CIRCUIT (60 expvals entering the classical head)\n")
        if post is None:
            f.write("  (no quantum layer)\n")
        else:
            for s in post:
                f.write(f"  sample {s['sample']} [{s['label']}]\n")
                f.write(f"    Z: {_fmt_row(s['Z'])}\n")
                f.write(f"    X: {_fmt_row(s['X'])}\n")
                f.write(f"    Y: {_fmt_row(s['Y'])}\n")

    print(f"\nSaved: {json_path}\n       {txt_path}")


if __name__ == "__main__":
    main()
