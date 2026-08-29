"""nPrint CSV ingestion, column removal, feature selection, and DataLoader creation."""

from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from config.defaults import BATCH_SIZE, COLUMNS_TO_REMOVE, N_FEATURES


def load_nprint(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load nPrint CSV and separate features from labels.

    The CSV has 962 columns: 961 feature columns and the last column is the
    label. The first column is a src_ip identifier, not a model feature; it is
    dropped later by remove_columns (index 0 of COLUMNS_TO_REMOVE).
    """
    df = pd.read_csv(csv_path, header=0)
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1]
    return features, labels


def remove_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Drop network-identifier columns (src_ip, IPv4 src/dst/identification,
    TCP ports, seq/ack numbers) listed in COLUMNS_TO_REMOVE."""
    cols_to_drop = [c for c in COLUMNS_TO_REMOVE if c < features.shape[1]]
    return features.drop(features.columns[cols_to_drop], axis=1)


def feature_names_to_indices(csv_path: str, names: list[str]) -> list[int]:
    """Map nPrint column names to positional indices in the post-column-removal
    feature frame — the basis stored as ``feature_indices`` and consumed by
    ``prepare_data(feature_indices=...)``.

    Reuses ``remove_columns`` so the removal set stays the single source of
    truth. Only the header is read (nrows=0), so this is cheap on large CSVs.
    Raises with a clear message if a name is absent from the data or falls
    inside a removed identifier column (which cannot be used as a feature).
    """
    header = pd.read_csv(csv_path, header=0, nrows=0)
    features = header.iloc[:, :-1]  # drop the label column
    kept = remove_columns(features)
    pos = {name: i for i, name in enumerate(kept.columns)}
    removed = set(features.columns) - set(kept.columns)

    in_removed = [n for n in names if n in removed]
    missing = [n for n in names if n not in pos and n not in removed]
    if missing:
        raise ValueError(f"feature names not found in {csv_path}: {missing}")
    if in_removed:
        raise ValueError(
            "these feature names fall inside removed identifier columns and "
            f"cannot be used (data leakage / dropped from the frame): {in_removed}"
        )
    return [pos[n] for n in names]


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    k: int = N_FEATURES,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Select top-k features using mutual information. Fit on training set only.

    mutual_info_classif is stochastic (nearest-neighbor estimation with random
    tie-breaking), so ``random_state`` is pinned to make selection reproducible
    across runs on identical data. The chosen indices are persisted in the
    model config as ``feature_indices`` and reused at eval time (see
    ``prepare_eval_data``) so evaluation never re-fits selection.
    """
    score_func = partial(mutual_info_classif, random_state=random_state)
    selector = SelectKBest(score_func, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector


def prepare_data(
    csv_path: str,
    k: int = N_FEATURES,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_indices: list[int] | None = None,
) -> dict:
    """Full data preparation pipeline.

    Returns dict with train/test tensors, label encoder, feature selector,
    and selected feature indices.

    The test split doubles as the validation set during training (early
    stopping, LR scheduling) — there is no separate held-out set, so final
    test metrics are optimistically biased. See "Known Limitations" in the
    README.

    If ``feature_indices`` is provided (e.g. when resuming a run), the given
    columns are sliced directly instead of re-fitting SelectKBest, so the
    resumed run uses the exact feature basis the checkpoint was trained on.
    The train/test split is deterministic for a fixed ``random_state``, so the
    split itself is already reproducible across runs.
    """
    features, labels = load_nprint(csv_path)
    features = remove_columns(features)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features.values,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    if feature_indices is not None:
        selector = None
        indices = list(feature_indices)
        X_train_sel = X_train[:, indices]
        X_test_sel = X_test[:, indices]
        # Scores come from fitting SelectKBest; unavailable when reusing indices.
        feature_scores = None
    else:
        X_train_sel, X_test_sel, selector = select_features(X_train, X_test, y_train, k=k)
        indices = selector.get_support(indices=True).tolist()
        # Mutual-information score for each selected feature (aligned with indices).
        feature_scores = [float(selector.scores_[i]) for i in indices]

    return {
        "X_train": torch.tensor(X_train_sel, dtype=torch.float32),
        "X_test": torch.tensor(X_test_sel, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "y_test": torch.tensor(y_test, dtype=torch.long),
        "label_encoder": le,
        "selector": selector,
        "feature_indices": indices,
        "feature_scores": feature_scores,
        "n_classes": len(le.classes_),
    }


def prepare_eval_data(
    csv_path: str,
    feature_indices: list[int],
    label_classes: list[str],
) -> dict:
    """Prepare evaluation data using a trained model's saved feature/label config.

    Unlike prepare_data, this does NOT re-fit feature selection or label
    encoding. It slices the exact columns the model was trained on (by the
    saved positional ``feature_indices`` into the post-column-removal frame)
    and maps labels using the model's saved ``label_classes`` ordering, so the
    model is always evaluated on the same feature basis and label mapping it
    learned. Raises ValueError if the eval CSV contains an unknown class or is
    missing expected columns.
    """
    features, labels = load_nprint(csv_path)
    features = remove_columns(features)

    n_cols = features.shape[1]
    out_of_range = [i for i in feature_indices if i >= n_cols]
    if out_of_range:
        raise ValueError(
            f"Saved feature_indices {out_of_range} exceed the {n_cols} columns "
            f"available after column removal — the eval CSV's schema does not "
            f"match the training data."
        )
    X = features.values[:, feature_indices]

    class_to_idx = {cls: i for i, cls in enumerate(label_classes)}
    unknown = sorted({str(l) for l in labels} - set(class_to_idx))
    if unknown:
        raise ValueError(
            f"Evaluation CSV contains labels not seen during training: {unknown}. "
            f"Known classes: {label_classes}"
        )
    y = np.array([class_to_idx[str(l)] for l in labels], dtype=np.int64)

    return {
        "X_eval": torch.tensor(X, dtype=torch.float32),
        "y_eval": torch.tensor(y, dtype=torch.long),
        "n_classes": len(label_classes),
    }


def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and testing."""
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    return train_loader, test_loader
