"""Tests for feature-importance ranking and circuit-boundary snapshots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from evaluation.inspection import (
    compute_mi_ranking,
    feature_importance_ranking,
    format_ranking,
    post_circuit_snapshot,
    pre_circuit_snapshot,
)
from training.hybrid_model import HybridQuantumNet


def test_feature_importance_ranking_sorts_desc():
    ranking = feature_importance_ranking([10, 20, 30], [0.1, 0.9, 0.5])
    assert [e["column"] for e in ranking] == [20, 30, 10]  # by score desc
    assert [e["rank"] for e in ranking] == [1, 2, 3]


def test_feature_importance_ranking_handles_no_scores():
    ranking = feature_importance_ranking([10, 20, 30], None)
    assert all(e["score"] is None for e in ranking)
    assert [e["column"] for e in ranking] == [10, 20, 30]  # selection order


def test_compute_mi_ranking():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(60, 4)).astype(float)
    y = rng.integers(0, 3, size=60)
    ranking = compute_mi_ranking(X, y, [10, 20, 30, 40])
    assert len(ranking) == 4
    assert sorted(e["column"] for e in ranking) == [10, 20, 30, 40]
    assert all(e["score"] is not None for e in ranking)
    # ranks are a 1..4 permutation
    assert sorted(e["rank"] for e in ranking) == [1, 2, 3, 4]


def test_format_ranking_truncates():
    ranking = feature_importance_ranking([1, 2, 3], [0.3, 0.2, 0.1])
    out = format_ranking(ranking, top=2)
    assert "1 more" in out


def test_pre_circuit_snapshot():
    X = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    y = torch.tensor([0, 1])
    snap = pre_circuit_snapshot(X, y, [5, 6, 7, 8], ["a", "b"], n_samples=2)
    assert len(snap) == 2
    assert snap[0]["label"] == "a"
    assert snap[0]["features"]["5"] == 1.0
    assert snap[1]["vector"] == [0.0, 1.0, 0.0, 1.0]


def test_post_circuit_snapshot_shapes():
    model = HybridQuantumNet(n_classes=3, n_qubits=4, n_layers=1, backend="default.qubit")
    X = torch.randint(0, 2, (3, 4), dtype=torch.float32)
    y = torch.tensor([0, 1, 2])
    snap = post_circuit_snapshot(model, X, y, ["a", "b", "c"], n_samples=3)
    assert len(snap) == 3
    for s in snap:
        assert len(s["Z"]) == 4 and len(s["X"]) == 4 and len(s["Y"]) == 4
        assert len(s["vector"]) == 12  # 3 * n_qubits
        assert all(-1.0 <= v <= 1.0 for v in s["vector"])


def test_post_circuit_snapshot_single_axis():
    """A Z-only model yields one 'Z' block per sample and no X/Y keys."""
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=1, backend="default.qubit", measure_axes=["Z"],
    )
    X = torch.tensor([[-1.0, 0.0, 1.0, 0.0], [1.0, 1.0, -1.0, 0.0]])
    y = torch.tensor([0, 1])
    snap = post_circuit_snapshot(model, X, y, ["a", "b", "c"], n_samples=2)
    assert len(snap) == 2
    for s in snap:
        assert len(s["Z"]) == 4 and len(s["vector"]) == 4
        assert "X" not in s and "Y" not in s
        assert all(-1.0 <= v <= 1.0 for v in s["vector"])


def test_post_circuit_snapshot_none_without_quantum():
    model = HybridQuantumNet(
        n_classes=3, n_qubits=4, n_layers=1, backend="default.qubit",
        use_quantum=False, use_skip=True,
    )
    X = torch.randint(0, 2, (3, 4), dtype=torch.float32)
    y = torch.tensor([0, 1, 2])
    assert post_circuit_snapshot(model, X, y, ["a", "b", "c"]) is None
