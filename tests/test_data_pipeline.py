"""Tests for the data pipeline: CSV loading, column removal, feature selection."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.defaults import COLUMNS_TO_REMOVE, N_FEATURES
from data.pipeline import (
    create_dataloaders,
    load_nprint,
    prepare_data,
    prepare_eval_data,
    remove_columns,
    select_features,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal nPrint-like CSV with 962 columns."""
    n_samples = 100
    n_feature_cols = 961
    rng = np.random.default_rng(42)

    features = rng.integers(0, 2, size=(n_samples, n_feature_cols))
    labels = rng.choice(["Linux", "Windows", "macOS"], size=n_samples)

    cols = [f"col_{i}" for i in range(n_feature_cols)] + ["label"]
    data = np.column_stack([features, labels])
    df = pd.DataFrame(data, columns=cols)

    csv_path = tmp_path / "test_nprint.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_nprint(sample_csv):
    features, labels = load_nprint(sample_csv)
    assert features.shape == (100, 961)
    assert len(labels) == 100
    assert set(labels) == {"Linux", "Windows", "macOS"}


def test_remove_columns(sample_csv):
    features, _ = load_nprint(sample_csv)
    reduced = remove_columns(features)
    expected_removed = len(set(c for c in COLUMNS_TO_REMOVE if c < 961))
    assert reduced.shape[1] == 961 - expected_removed


def test_select_features():
    rng = np.random.default_rng(42)
    X_train = rng.integers(0, 2, size=(80, 100)).astype(float)
    X_test = rng.integers(0, 2, size=(20, 100)).astype(float)
    y_train = rng.integers(0, 3, size=80)

    X_train_sel, X_test_sel, selector = select_features(X_train, X_test, y_train, k=20)
    assert X_train_sel.shape == (80, 20)
    assert X_test_sel.shape == (20, 20)
    assert len(selector.get_support(indices=True)) == 20


def test_prepare_data(sample_csv):
    data = prepare_data(sample_csv, k=N_FEATURES)
    assert data["X_train"].shape[1] == N_FEATURES
    assert data["X_test"].shape[1] == N_FEATURES
    assert isinstance(data["X_train"], torch.Tensor)
    assert data["X_train"].dtype == torch.float32
    assert data["y_train"].dtype == torch.long
    assert data["n_classes"] == 3
    # Mutual-information score per selected feature, aligned with feature_indices.
    assert len(data["feature_scores"]) == N_FEATURES
    assert all(isinstance(s, float) for s in data["feature_scores"])


def test_create_dataloaders(sample_csv):
    data = prepare_data(sample_csv, k=N_FEATURES)
    train_loader, test_loader = create_dataloaders(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"], batch_size=16
    )
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape[1] == N_FEATURES
    assert y_batch.dtype == torch.long


def test_select_features_reproducible():
    """Pinned random_state makes mutual-information selection deterministic."""
    rng = np.random.default_rng(0)
    X_train = rng.integers(0, 2, size=(80, 100)).astype(float)
    X_test = rng.integers(0, 2, size=(20, 100)).astype(float)
    y_train = rng.integers(0, 3, size=80)

    _, _, sel_a = select_features(X_train, X_test, y_train, k=20)
    _, _, sel_b = select_features(X_train, X_test, y_train, k=20)
    assert sel_a.get_support(indices=True).tolist() == sel_b.get_support(indices=True).tolist()


def test_prepare_eval_data_matches_training_basis(sample_csv):
    """prepare_eval_data slices the saved feature columns and label mapping
    without re-fitting selection or the label encoder."""
    train = prepare_data(sample_csv, k=N_FEATURES)
    feature_indices = train["feature_indices"]
    label_classes = list(train["label_encoder"].classes_)

    ev = prepare_eval_data(sample_csv, feature_indices, label_classes)

    assert ev["X_eval"].shape[1] == N_FEATURES
    assert ev["X_eval"].dtype == torch.float32
    assert ev["y_eval"].dtype == torch.long
    assert ev["n_classes"] == len(label_classes)
    # Label indices must respect the saved class ordering exactly.
    assert set(ev["y_eval"].tolist()).issubset(set(range(len(label_classes))))


def test_prepare_eval_data_rejects_unknown_class(sample_csv):
    """An eval label absent from the training class list is a hard error, not a
    silently shifted index."""
    with pytest.raises(ValueError, match="not seen during training"):
        prepare_eval_data(sample_csv, [0, 1, 2], ["Linux", "Windows"])  # missing macOS
