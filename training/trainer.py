"""Training loop with Adam optimizer, early stopping, and checkpointing."""

import json
import os
import signal
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.defaults import (
    BATCH_SIZE,
    FOCAL_LOSS_GAMMA,
    LEARNING_RATE,
    LR_MIN,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    LR_WARMUP_EPOCHS,
    MAX_GRAD_NORM,
    PATIENCE,
)
from sklearn.metrics import f1_score as sklearn_f1_score

from evaluation.metrics import compute_accuracy, compute_f1
from training.hybrid_model import HybridQuantumNet


class FocalLoss(nn.Module):
    """Focal Loss with optional class weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy, well-classified examples and focuses gradient signal
    on hard, confusable examples. Combined with inverse-frequency class weights
    to address both class imbalance (alpha) and class confusion (gamma).
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = nn.functional.one_hot(targets, num_classes=logits.size(1)).float()

        # Gather p_t and log(p_t) for the true class
        p_t = (probs * targets_one_hot).sum(dim=1)
        log_p_t = (log_probs * targets_one_hot).sum(dim=1)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Per-sample loss
        loss = -focal_weight * log_p_t

        # Apply class weights if provided
        if self.weight is not None:
            alpha_t = self.weight[targets]
            loss = alpha_t * loss

        return loss.mean()


EPOCH_LOG_FILE = "training_log.txt"
PROGRESS_INTERVAL = 10  # write progress every N batches
LOG_GRAD_INTERVAL = 50  # log gradient norms every N batches
# Persist full resume state mid-epoch every N batches. Epochs can run for many
# hours (days on the full dataset); without this a hard kill (SIGKILL/OOM/power
# loss) would discard the whole in-flight epoch. The .resume.pt file is small
# (well under 1 MB for the 20-qubit model), so the write is negligible next to
# a single batch, which takes minutes under adjoint differentiation.
RESUME_CHECKPOINT_INTERVAL = 50

# Markers appended to EPOCH_LOG_FILE so the *next* run can tell how the previous
# one ended (clean END, caught-signal SHUTDOWN, or nothing => abrupt kill).
_RUN_MARKERS = ("=== RUN START", "=== RUN END", "=== RUN SHUTDOWN")


def append_log_line(save_dir: str, line: str) -> None:
    """Append a single raw line to the human-readable training log."""
    path = os.path.join(save_dir, EPOCH_LOG_FILE)
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def prior_run_exited_uncleanly(save_dir: str) -> bool:
    """True if the last run marker in the log is a START with no matching
    END/SHUTDOWN — i.e. the previous process was killed without catching a
    signal (SIGKILL, kernel OOM, or host power loss). A caught SIGTERM writes
    a SHUTDOWN marker, and a normal finish writes END, so either of those
    means the prior exit was accounted for.
    """
    path = os.path.join(save_dir, EPOCH_LOG_FILE)
    if not os.path.exists(path):
        return False
    last = None
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(_RUN_MARKERS):
                    last = line
    except OSError:
        return False
    return last is not None and last.startswith("=== RUN START")


def get_progress_filename(model_name: str, start_date: str) -> str:
    """Return progress filename namespaced by model and date, e.g.
    training_progress_model_out_2026-07-07.json.

    Namespacing by model_name keeps concurrent/sequential runs of different
    models in the same save_dir from overwriting each other's progress or
    cross-contaminating warm-start seeds (see train.py:latest_best).
    """
    return f"training_progress_{model_name}_{start_date}.json"


def write_progress(save_dir: str, status: dict, model_name: str, start_date: str) -> None:
    """Write training progress to a model+date-stamped JSON file for monitoring."""
    filename = get_progress_filename(model_name, start_date)
    path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def write_epoch_log(save_dir: str, epoch: int, epochs: int, avg_loss: float,
                    val_acc: float, val_f1: float, best_f1: float, best_epoch: int,
                    lr: float, per_class_f1: list, elapsed_seconds: float,
                    header: bool = False) -> None:
    """Append one line per epoch to a human-readable text log."""
    path = os.path.join(save_dir, EPOCH_LOG_FILE)
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "a") as f:
        if header:
            f.write(f"\n{'=' * 100}\n")
            f.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'Epoch':>6} {'Loss':>8} {'Acc':>8} {'Val_F1':>8} "
                    f"{'Best_F1':>8} {'BestEp':>7} {'LR':>10} "
                    f"{'Elapsed':>10}  Per-class F1\n")
            f.write("-" * 100 + "\n")
            return
        # Day-aware elapsed: epochs can exceed 24h, so gmtime + %H would wrap.
        h, rem = divmod(int(elapsed_seconds), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
        class_str = " ".join(f"{v:.2f}" for v in per_class_f1)
        f.write(f"{epoch:>3}/{epochs:<3} {avg_loss:>8.4f} {val_acc:>8.4f} {val_f1:>8.4f} "
                f"{best_f1:>8.4f} {best_epoch:>7} {lr:>10.6f} "
                f"{elapsed_str:>10}  [{class_str}]\n")


def compute_class_weights(y_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = torch.bincount(y_train, minlength=n_classes).float()
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * n_classes
    return weights


def save_checkpoint(
    model: HybridQuantumNet,
    config: dict,
    save_dir: str,
    name: str,
) -> None:
    """Save model weights (.pt) and config (.json)."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}.pt"))
    with open(os.path.join(save_dir, f"{name}.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_resume_checkpoint(save_dir: str, name: str, state: dict) -> None:
    """Atomically write the full training state needed to resume a run.

    Unlike save_checkpoint (which stores only the *best* model weights), this
    captures the *latest* model plus optimizer/scheduler state and loop
    counters, written every epoch. It lets a run continue exactly where it
    stopped after a crash or a Docker/daemon restart.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.resume.pt")
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_checkpoint(save_dir: str, name: str) -> dict | None:
    """Load the full training state written by save_resume_checkpoint, if any."""
    path = os.path.join(save_dir, f"{name}.resume.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False)


def load_checkpoint(
    save_dir: str,
    name: str,
    backend: str = "default.qubit",
) -> tuple[HybridQuantumNet, dict]:
    """Load model from checkpoint."""
    config_path = os.path.join(save_dir, f"{name}.json")
    with open(config_path) as f:
        config = json.load(f)

    model = HybridQuantumNet(
        n_classes=config["n_classes"],
        n_qubits=config["n_qubits"],
        n_layers=config["n_layers"],
        backend=backend,
        entangler=config.get("entangler", "cnot"),
        use_quantum=config.get("use_quantum", True),
        use_skip=config.get("use_skip", False),
        use_batchnorm=config.get("use_batchnorm", False),
        dropout=config.get("dropout", 0.3),
        # Default to the original 60-observable full-Bloch config so pre-existing
        # checkpoints (which omit these keys) rebuild to the identical model.
        measure_axes=config.get("measure_axes", ["Z", "X", "Y"]),
        shortcut_offsets=config.get("shortcut_offsets", []),
    )
    weights_path = os.path.join(save_dir, f"{name}.pt")
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model, config


def train_model(
    model: HybridQuantumNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    y_train: torch.Tensor,
    epochs: int = 50,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    save_dir: str = "models",
    model_name: str = "model",
    config: dict | None = None,
    resume_state: dict | None = None,
) -> dict:
    """Train the hybrid quantum-classical model.

    Returns dict with training history (losses, f1_scores, best_epoch).

    ``test_loader`` serves as the validation set: early stopping, LR
    scheduling, and best-checkpoint selection all key off its F1 score,
    so metrics reported on it are optimistically biased (see "Known
    Limitations" in the README).

    The best checkpoint is only written when ``config`` is provided;
    with ``config=None`` the model is trained but never saved.

    ``resume_state`` (from load_resume_checkpoint) restores optimizer,
    scheduler, epoch, and early-stopping counters so a run continues exactly
    where it stopped. The model weights themselves are restored by the caller
    before this function is called.
    """
    class_weights = compute_class_weights(y_train, n_classes)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}", flush=True)
    print(f"  min={class_weights.min():.3f} max={class_weights.max():.3f} ratio={class_weights.max()/class_weights.min():.1f}x", flush=True)
    criterion = FocalLoss(weight=class_weights, gamma=FOCAL_LOSS_GAMMA)
    print(f"Loss: FocalLoss(gamma={FOCAL_LOSS_GAMMA})", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_MIN,
    )

    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0
    history = {"train_loss": [], "val_acc": [], "val_f1": [], "learning_rates": []}

    # Restore full training state to continue a previous run.
    if resume_state is not None:
        if resume_state.get("optimizer") is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        best_f1 = resume_state.get("best_f1", 0.0)
        best_epoch = resume_state.get("best_epoch", 0)
        epochs_without_improvement = resume_state.get("epochs_without_improvement", 0)
        start_epoch = resume_state.get("epoch", 0)  # next epoch index to run
        print(
            f"Resuming from epoch {start_epoch} "
            f"(best_f1={best_f1:.4f} @ epoch {best_epoch})",
            flush=True,
        )

    total_batches = len(train_loader)
    training_start = time.time()
    start_date = time.strftime("%Y-%m-%d")

    # --- Crash forensics + graceful shutdown ---------------------------------
    # If the previous run left a START with no END/SHUTDOWN, it was killed
    # abruptly (SIGKILL/OOM/power loss) rather than stopped gracefully. Record
    # that in-repo; the authoritative cause still comes from `docker inspect`
    # (OOMKilled/ExitCode) and journald around that timestamp.
    if prior_run_exited_uncleanly(save_dir):
        note = (f"NOTE {time.strftime('%Y-%m-%d %H:%M:%S')}: previous run left no "
                f"END/SHUTDOWN marker -> killed abruptly (SIGKILL/OOM/power loss), "
                f"not a graceful stop. Confirm with `docker inspect` OOMKilled and "
                f"journald around the prior FinishedAt.")
        append_log_line(save_dir, note)
        print(note, flush=True)

    append_log_line(
        save_dir,
        f"=== RUN START {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} "
        f"host={os.uname().nodename} model={model_name} "
        f"resume_from_epoch={start_epoch} ===",
    )

    # Persist the config JSON up front (not just on the first best-F1 epoch) so
    # a restart can `--resume` as soon as the mid-epoch .resume.pt exists —
    # critical when a single epoch runs for days on the large dataset.
    if config is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{model_name}.json"), "w") as f:
            json.dump(config, f, indent=2)

    # Mutable view of loop position so the signal handler can report exactly
    # where we were and persist a resume checkpoint before exiting.
    live = {
        "epoch": start_epoch, "batch": 0, "total_batches": total_batches,
        "best_f1": best_f1, "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
    }

    def _handle_shutdown(signum, _frame):
        # Reached only at a Python bytecode boundary, i.e. BETWEEN batches. A
        # signal arriving mid-batch waits for the current batch (minutes under
        # adjoint differentiation) to finish, so Docker's stop_grace_period
        # must exceed one batch for this to fire before SIGKILL. The periodic mid-epoch checkpoint is the guaranteed
        # durability path; this handler is best-effort forensics + a final save.
        name = signal.Signals(signum).name
        msg = (f"=== RUN SHUTDOWN {time.strftime('%Y-%m-%d %H:%M:%S')}: caught "
               f"{name} at epoch {live['epoch'] + 1}, "
               f"batch {live['batch']}/{live['total_batches']} — "
               f"checkpointing then exiting ===")
        append_log_line(save_dir, msg)
        print("\n" + msg, flush=True)
        if config is not None:
            # epoch index (not +1) => resume re-runs this epoch from batch 0
            # with the partially-updated (warm) weights.
            save_resume_checkpoint(save_dir, model_name, {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": live["epoch"],
                "best_f1": live["best_f1"],
                "best_epoch": live["best_epoch"],
                "epochs_without_improvement": live["epochs_without_improvement"],
            })
            append_log_line(save_dir, f"    resume checkpoint saved at epoch {live['epoch'] + 1}")
        sys.exit(128 + signum)  # conventional exit code for a signal death

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # Write header for epoch log
    write_epoch_log(save_dir, 0, epochs, 0, 0, 0, 0, 0, lr, [], 0, header=True)

    for epoch in range(start_epoch, epochs):
        live["epoch"] = epoch
        # LR warmup: linearly ramp from lr/10 to lr over warmup epochs
        if epoch < LR_WARMUP_EPOCHS:
            warmup_lr = lr * (0.1 + 0.9 * epoch / max(LR_WARMUP_EPOCHS - 1, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for X_batch, y_batch in train_loader:
            batch_start = time.time()
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            # Measure gradient norms before clipping (read-only)
            log_grads = (n_batches + 1) % LOG_GRAD_INTERVAL == 0
            if log_grads:
                q_params = (
                    [p for p in model.quantum_layer.parameters() if p.grad is not None]
                    if model.quantum_layer is not None else []
                )
                c_params = [p for p in model.classical_head.parameters() if p.grad is not None]
                q_grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in q_params)).item() if q_params else 0.0
                c_grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in c_params)).item() if c_params else 0.0
                pre_clip_norm = torch.sqrt(torch.tensor(q_grad_norm ** 2 + c_grad_norm ** 2)).item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            live["batch"] = n_batches
            batch_elapsed = time.time() - batch_start

            # Mid-epoch durability: persist full resume state periodically so a
            # hard kill loses at most a few hours, not the whole multi-day
            # epoch. epoch index (not +1) => resume re-runs the current epoch
            # from batch 0 with the partially-updated weights.
            if config is not None and n_batches % RESUME_CHECKPOINT_INTERVAL == 0:
                save_resume_checkpoint(save_dir, model_name, {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                })

            if log_grads:
                clipped = " CLIPPED" if pre_clip_norm > MAX_GRAD_NORM else ""
                print(
                    f"  grad [{n_batches}] "
                    f"quantum={q_grad_norm:.4f} classical={c_grad_norm:.4f} "
                    f"total={pre_clip_norm:.4f}{clipped}",
                    flush=True,
                )

            # Log and write progress periodically
            if n_batches % PROGRESS_INTERVAL == 0 or n_batches == total_batches:
                elapsed = time.time() - epoch_start
                remaining_batches = total_batches - n_batches
                eta_epoch = remaining_batches * (elapsed / n_batches)
                write_progress(save_dir, {
                    "phase": "training",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "batch": n_batches,
                    "total_batches": total_batches,
                    "batch_loss": loss.item(),
                    "avg_loss": epoch_loss / n_batches,
                    "last_batch_seconds": round(batch_elapsed, 1),
                    "epoch_elapsed_seconds": round(elapsed, 1),
                    "epoch_eta_seconds": round(eta_epoch, 1),
                    "total_elapsed_seconds": round(time.time() - training_start, 1),
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "learning_rate": current_lr,
                }, model_name, start_date)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch))
                print(
                    f"  [{n_batches}/{total_batches}] "
                    f"loss={epoch_loss / n_batches:.4f} "
                    f"batch={batch_elapsed:.1f}s "
                    f"eta={eta_str}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation phase
        write_progress(save_dir, {
            "phase": "validating",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "avg_loss": avg_loss,
            "total_elapsed_seconds": round(time.time() - training_start, 1),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
        }, model_name, start_date)

        model.eval()
        all_preds = []
        all_labels = []
        val_start = time.time()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_acc = compute_accuracy(all_labels.numpy(), all_preds.numpy())
        val_f1 = compute_f1(all_labels.numpy(), all_preds.numpy())
        val_elapsed = time.time() - val_start
        epoch_elapsed = time.time() - epoch_start
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.6f} "
            f"[train={epoch_elapsed - val_elapsed:.0f}s val={val_elapsed:.0f}s]",
            flush=True,
        )

        # Per-class F1 breakdown
        per_class_f1 = sklearn_f1_score(
            all_labels.numpy(), all_preds.numpy(),
            average=None, zero_division=0,
        )
        class_strs = [f"{i}:{f:.2f}" for i, f in enumerate(per_class_f1)]
        print(f"  per-class F1: {' '.join(class_strs)}", flush=True)

        # Flag classes with zero F1
        zero_classes = [i for i, f in enumerate(per_class_f1) if f == 0.0]
        if zero_classes:
            print(f"  WARNING: classes {zero_classes} have F1=0.0 (not predicted)", flush=True)

        # Step scheduler after warmup completes
        if epoch >= LR_WARMUP_EPOCHS:
            scheduler.step(val_f1)

        # Early stopping + checkpointing (update best before logging so the
        # log reflects this epoch's best, not the previous epoch's)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if config is not None:
                save_checkpoint(model, config, save_dir, model_name)
        else:
            epochs_without_improvement += 1

        # Append epoch results to text log (after best_f1/best_epoch updated)
        write_epoch_log(
            save_dir, epoch + 1, epochs, avg_loss, val_acc, val_f1,
            best_f1, best_epoch, current_lr,
            per_class_f1.tolist(), time.time() - training_start,
        )

        write_progress(save_dir, {
            "phase": "epoch_complete",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "avg_loss": avg_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "total_elapsed_seconds": round(time.time() - training_start, 1),
            "learning_rate": current_lr,
        }, model_name, start_date)

        live.update(best_f1=best_f1, best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement)

        # Persist full state every epoch so a crash/daemon-restart can resume
        # from exactly here (next epoch index = epoch + 1).
        save_resume_checkpoint(save_dir, model_name, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        })

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    append_log_line(
        save_dir,
        f"=== RUN END {time.strftime('%Y-%m-%d %H:%M:%S')} clean: "
        f"best_f1={best_f1:.4f} @ epoch {best_epoch}, "
        f"last_epoch={live['epoch'] + 1} ===",
    )

    history["best_epoch"] = best_epoch
    history["best_f1"] = best_f1
    return history
