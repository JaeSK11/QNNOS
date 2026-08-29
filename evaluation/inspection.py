"""Model inspection utilities.

Two things you can look at that are otherwise invisible:

  1. Feature-importance ranking -- which of the selected features carry the most
     signal, by mutual information with the label.
  2. Data snapshots at the two interesting boundaries:
       - PRE-circuit:  the processed ternary feature vector (-1/0/1, one value
                       per qubit) that enters the QNode.
       - POST-circuit: the Pauli expectation values (one per measured axis per
                       qubit; 60 for the default Z+X+Y on 20 qubits) that leave
                       the QNode and enter the classical head.
"""

from functools import partial

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif


def feature_importance_ranking(feature_indices: list[int], feature_scores) -> list[dict]:
    """Rank features by a precomputed score (descending).

    Each entry: {rank, column, score}. If feature_scores is None (e.g. a resumed
    run reused saved indices without re-fitting selection), returns entries in
    selection order with score=None.
    """
    if feature_scores is None:
        return [{"rank": i + 1, "column": c, "score": None}
                for i, c in enumerate(feature_indices)]
    order = sorted(range(len(feature_indices)), key=lambda i: feature_scores[i], reverse=True)
    return [{"rank": r + 1, "column": feature_indices[i], "score": float(feature_scores[i])}
            for r, i in enumerate(order)]


def compute_mi_ranking(X, y, feature_indices: list[int], random_state: int = 42) -> list[dict]:
    """Compute a fresh mutual-information ranking on the given (processed) data.

    X is (n_samples, n_selected_features), column j corresponds to
    feature_indices[j]. Self-contained: does not depend on scores persisted at
    train time, so it works for any checkpoint.

    Uses discrete_features=True because nPrint features are discrete (ternary
    -1/0/1) -- this is the exact contingency-table MI estimator (deterministic, robust for small
    samples), which differs slightly from the pipeline's continuous kNN
    estimator used during feature *selection*.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    scores = mutual_info_classif(
        X, y, discrete_features=True, random_state=random_state
    )
    return feature_importance_ranking(feature_indices, scores.tolist())


def format_ranking(ranking: list[dict], top: int | None = None) -> str:
    """Human-readable ranking table (column index = post-column-removal index)."""
    rows = ranking if top is None else ranking[:top]
    lines = [f"{'Rank':>4}  {'Column':>7}  {'MI score':>10}", "-" * 26]
    for e in rows:
        s = "n/a" if e["score"] is None else f"{e['score']:.4f}"
        lines.append(f"{e['rank']:>4}  {e['column']:>7}  {s:>10}")
    if top is not None and len(ranking) > top:
        lines.append(f"     ... ({len(ranking) - top} more)")
    return "\n".join(lines)


def _label_of(y_i, label_classes):
    return label_classes[int(y_i)] if label_classes else int(y_i)


def pre_circuit_snapshot(X, y, feature_indices, label_classes=None, n_samples=5) -> list[dict]:
    """The processed feature vectors that enter the circuit (ternary -1/0/1, one per qubit)."""
    X = np.asarray(X)
    n = min(n_samples, X.shape[0])
    out = []
    for i in range(n):
        vec = [float(v) for v in X[i]]
        out.append({
            "sample": i,
            "label": _label_of(y[i], label_classes),
            "features": {str(feature_indices[j]): vec[j] for j in range(len(feature_indices))},
            "vector": vec,
        })
    return out


def post_circuit_snapshot(model, X, y, label_classes=None, n_samples=5):
    """The quantum expectation values that leave the circuit and enter the head.

    Returns None if the model has no quantum layer (--no-quantum baseline).
    Output ordering matches the circuit: grouped by measured axis, then qubit
    (e.g. [Z0..Z(nq-1), X0.., Y0..] for the default Z+X+Y). Each entry carries
    one key per measured axis ("Z", "X", "Y") plus the flat "vector".
    """
    if getattr(model, "quantum_layer", None) is None:
        return None
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = min(n_samples, X.shape[0])
    nq = model.n_qubits
    axes = list(getattr(model, "measure_axes", ["Z", "X", "Y"]))
    model.eval()
    with torch.no_grad():
        expvals = model.quantum_layer(X[:n]).cpu().numpy()
    out = []
    for i in range(n):
        row = [float(v) for v in expvals[i]]
        entry = {"sample": i, "label": _label_of(y[i], label_classes)}
        for j, ax in enumerate(axes):
            entry[ax] = row[j * nq:(j + 1) * nq]
        entry["vector"] = row
        out.append(entry)
    return out
