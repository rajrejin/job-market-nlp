"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def metrics(y_true: list[str], proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    idx = {c: i for i, c in enumerate(classes)}
    yt = np.array([idx[y] for y in y_true])
    yp = proba.argmax(axis=1)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    p, r, f, s = precision_recall_fscore_support(
        yt, yp, labels=range(len(classes)), zero_division=0
    )
    return {
        "macro_f1": round(
            float(f1_score(yt, yp, average="macro", labels=range(len(classes)), zero_division=0)), 4
        ),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "top3_accuracy": round(float(np.mean([yt[i] in top3[i] for i in range(len(yt))])), 4),
        "per_class": {
            c: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion": confusion_matrix(yt, yp, labels=range(len(classes))).tolist(),
        "classes": classes,
    }


def _macro_f1(yt: np.ndarray, yp: np.ndarray, k: int) -> float:
    return float(f1_score(yt, yp, average="macro", labels=range(k), zero_division=0))


def bootstrap_ci(
    y_true: list[str], proba: np.ndarray, classes: list[str], n: int = 1000, seed: int = 0
) -> dict[str, list[float]]:
    idx = {c: i for i, c in enumerate(classes)}
    yt = np.array([idx[y] for y in y_true])
    yp = proba.argmax(axis=1)
    rng = np.random.default_rng(seed)
    f1s, accs = [], []
    for _ in range(n):
        b = rng.integers(0, len(yt), len(yt))
        f1s.append(_macro_f1(yt[b], yp[b], len(classes)))
        accs.append(float(np.mean(yt[b] == yp[b])))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    return {"macro_f1_95ci": q(f1s), "accuracy_95ci": q(accs)}


def paired_bootstrap(
    y_true: list[str],
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    classes: list[str],
    n: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Macro-F1 difference a - b on the same resampled test postings."""
    idx = {c: i for i, c in enumerate(classes)}
    yt = np.array([idx[y] for y in y_true])
    pa, pb = proba_a.argmax(axis=1), proba_b.argmax(axis=1)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        b = rng.integers(0, len(yt), len(yt))
        diffs.append(_macro_f1(yt[b], pa[b], len(classes)) - _macro_f1(yt[b], pb[b], len(classes)))
    d = np.array(diffs)
    return {
        "mean_diff": round(float(d.mean()), 4),
        "diff_95ci": [
            round(float(np.percentile(d, 2.5)), 4),
            round(float(np.percentile(d, 97.5)), 4),
        ],
        "share_resamples_a_better": round(float(np.mean(d > 0)), 4),
    }
