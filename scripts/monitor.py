#!/usr/bin/env python3
"""Monitor training progress by reading the progress JSON file."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.defaults import PATIENCE


def format_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def show_progress(path):
    try:
        with open(path) as f:
            p = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No progress file at {path}")
        return

    phase = p.get("phase", "unknown")
    epoch = p.get("epoch", "?")
    total_epochs = p.get("total_epochs", "?")
    elapsed = format_time(p.get("total_elapsed_seconds", 0))

    if phase == "training":
        batch = p.get("batch", 0)
        total = p.get("total_batches", 0)
        pct = (batch / total * 100) if total else 0
        eta = format_time(p.get("epoch_eta_seconds", 0))
        print(f"  Phase:     TRAINING")
        print(f"  Epoch:     {epoch}/{total_epochs}")
        print(f"  Batch:     {batch}/{total} ({pct:.1f}%)")
        print(f"  Avg Loss:  {p.get('avg_loss', 0):.4f}")
        print(f"  Batch Time:{p.get('last_batch_seconds', 0):.1f}s")
        print(f"  Epoch ETA: {eta}")
        print(f"  Elapsed:   {elapsed}")
        print(f"  Best F1:   {p.get('best_f1', 0):.4f}")
    elif phase == "validating":
        print(f"  Phase:     VALIDATING")
        print(f"  Epoch:     {epoch}/{total_epochs}")
        print(f"  Avg Loss:  {p.get('avg_loss', p.get('train_loss', 0)):.4f}")
        print(f"  Elapsed:   {elapsed}")
        print(f"  Best F1:   {p.get('best_f1', 0):.4f}")
    elif phase == "epoch_complete":
        print(f"  Phase:     EPOCH COMPLETE")
        print(f"  Epoch:     {epoch}/{total_epochs}")
        print(f"  Avg Loss:  {p.get('avg_loss', 0):.4f}")
        print(f"  Val F1:    {p.get('val_f1', 0):.4f}")
        print(f"  Best F1:   {p.get('best_f1', 0):.4f} (epoch {p.get('best_epoch', '?')})")
        stall = p.get("epochs_without_improvement", 0)
        print(f"  Stall:     {stall}/{PATIENCE} epochs without improvement")
        print(f"  Elapsed:   {elapsed}")
    else:
        print(f"  Phase: {phase}")
        print(f"  {json.dumps(p, indent=2)}")


def resolve_progress_path(progress_dir):
    """Return the newest date-stamped progress file, or the legacy path.

    The trainer writes training_progress_<model>_<date>.json per run; ISO dates
    sort lexically, so the last match is the most recent run.
    """
    candidates = sorted(Path(progress_dir).glob("training_progress_*.json"))
    if candidates:
        return candidates[-1]
    return Path(progress_dir) / "training_progress.json"


def main():
    # First non-flag argument is the progress directory (default "models").
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    progress_dir = positional[0] if positional else "models"

    watch = "--watch" in sys.argv or "-w" in sys.argv
    interval = 30

    if watch:
        print(f"Watching {progress_dir} every {interval}s (Ctrl+C to stop)\n")
        while True:
            path = resolve_progress_path(progress_dir)
            print(f"--- {time.strftime('%H:%M:%S')} ({path.name}) ---")
            show_progress(path)
            print()
            time.sleep(interval)
    else:
        show_progress(resolve_progress_path(progress_dir))


if __name__ == "__main__":
    main()
