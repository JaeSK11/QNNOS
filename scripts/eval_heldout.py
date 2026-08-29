#!/usr/bin/env python3
"""CLI entrypoint: evaluate a checkpoint on the held-out split of its training CSV.

scripts/test.py scores every row of the CSV it is given, so pointing it at the
training CSV mixes in the ~80% of rows the model was trained on. This script
instead reproduces the training-time stratified split (same seed, same feature
columns, same label ordering) and scores only the held-out portion. That is the
split which served as the validation set during training, so the weighted F1
here matches the training log's Val_F1, and macro F1 plus per-class detail are
added on top.

The same predictions can also be scored at coarser label granularities:
  --merge A,B[,C...]   merge the listed classes into one (repeatable)
  --family             collapse labels to their OS family (prefix before the
                       first '-' or '_', e.g. "ubuntu-16.4-64b" -> "ubuntu")

Writes <model>_heldout.txt (report) and <model>_heldout_preds.npz (raw
predictions) next to the model, or under --out.
"""

import argparse
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config.defaults import DEFAULT_BACKEND
from data.pipeline import prepare_data
from training.trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the held-out split of its training CSV"
    )
    parser.add_argument("csv_path", help="The nPrint CSV the model was trained on")
    parser.add_argument("model_path", help="Path to model dir/name (without extension)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Quantum backend")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Held-out fraction used at training time (default 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Split seed used at training time (default 42)",
    )
    parser.add_argument(
        "--merge", action="append", default=[], metavar="A,B[,C...]",
        help="Comma-separated class names to merge into one class for an "
        "additional scoring view; may be given more than once.",
    )
    parser.add_argument(
        "--family", action=argparse.BooleanOptionalAction, default=True,
        help="Also score at OS-family granularity (label prefix before the "
        "first '-' or '_').",
    )
    parser.add_argument("--out", default=None, help="Output dir (default: model dir)")
    return parser.parse_args()


def section(title: str, y_true, y_pred, names: list[str]) -> str:
    """Accuracy / macro F1 / weighted F1, per-class report, and confusion matrix."""
    lines = [
        f"=== {title} ===",
        f"Accuracy:    {np.mean(y_true == y_pred):.4f}",
        f"F1 macro:    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        f"F1 weighted: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
        "",
        classification_report(
            y_true, y_pred, labels=range(len(names)), target_names=names, zero_division=0
        ),
        "Confusion matrix (rows = true, cols = predicted, order as above):",
    ]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    width = max(len(str(cm.max())), 5)
    lines.append(" " * 24 + "".join(f"{n[:width]:>{width + 1}}" for n in names))
    for name, row in zip(names, cm):
        lines.append(f"{name:>23} " + "".join(f"{v:>{width + 1}}" for v in row))
    lines.append("")
    return "\n".join(lines)


def remap(names: list[str], merge_map: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Return (merged_names, index_remap) collapsing classes per merge_map.

    ``index_remap[i]`` is the merged-class index of original class ``i``, so a
    label array is converted with ``index_remap[y]``.
    """
    merged_names: list[str] = []
    remap_idx = np.zeros(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        target = merge_map.get(n, n)
        if target not in merged_names:
            merged_names.append(target)
        remap_idx[i] = merged_names.index(target)
    return merged_names, remap_idx


def family_of(label: str) -> str:
    return re.split(r"[-_]", label, maxsplit=1)[0]


def main() -> None:
    args = parse_args()
    model_dir = str(Path(args.model_path).parent)
    model_name = Path(args.model_path).name
    out_dir = Path(args.out) if args.out else Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}/{model_name}...", flush=True)
    model, config = load_checkpoint(model_dir, model_name, backend=args.backend)
    if "feature_indices" not in config or "label_classes" not in config:
        raise SystemExit("Model config lacks feature_indices/label_classes; cannot evaluate.")
    class_names = [str(c) for c in config["label_classes"]]

    print(f"Reproducing the training split from {args.csv_path}...", flush=True)
    data = prepare_data(
        args.csv_path,
        test_size=args.test_size,
        random_state=args.random_state,
        feature_indices=config["feature_indices"],
    )
    split_classes = [str(c) for c in data["label_encoder"].classes_]
    if split_classes != class_names:
        raise SystemExit(
            "Label classes in this CSV do not match the checkpoint's label_classes; "
            "the split would not reproduce the training run.\n"
            f"  CSV:   {split_classes}\n  model: {class_names}"
        )

    ds = TensorDataset(data["X_test"], data["y_test"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {len(ds)} held-out samples ({len(loader)} batches)...", flush=True)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(loader):
            preds.append(model(X_batch).argmax(dim=1))
            labels.append(y_batch)
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                print(f"  batch {i + 1}/{len(loader)}", flush=True)

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()

    npz_path = out_dir / f"{model_name}_heldout_preds.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, label_classes=np.array(class_names))

    out = [
        f"Held-out evaluation of {model_dir}/{model_name}",
        f"Split: stratified {args.test_size:.0%} of {args.csv_path}, "
        f"random_state={args.random_state} (the split that served as the "
        f"validation set during training; not an untouched holdout)",
        f"Samples: {len(ds)}",
        "",
        section(f"{len(class_names)}-class (as trained)", y_true, y_pred, class_names),
    ]

    for group in args.merge:
        members = [m.strip() for m in group.split(",") if m.strip()]
        unknown = [m for m in members if m not in class_names]
        if unknown:
            raise SystemExit(f"--merge names not in the model's classes: {unknown}")
        merged_label = "+".join(members)
        names, idx = remap(class_names, {m: merged_label for m in members})
        out.append(section(f"{len(names)}-class ({merged_label} merged)",
                           idx[y_true], idx[y_pred], names))

    if args.family:
        fam_map = {n: family_of(n) for n in class_names}
        if len(set(fam_map.values())) > 1:
            names, idx = remap(class_names, fam_map)
            out.append(section(f"{len(names)}-class OS family ({' / '.join(names)})",
                               idx[y_true], idx[y_pred], names))

    text = "\n".join(out)
    txt_path = out_dir / f"{model_name}_heldout.txt"
    txt_path.write_text(text)
    print("\n" + text)
    print(f"Saved: {txt_path}\n       {npz_path}")


if __name__ == "__main__":
    main()
