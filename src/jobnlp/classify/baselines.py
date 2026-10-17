"""TF-IDF and embedding baselines with logistic regression (C chosen on the validation split)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


def _fit_best_c(
    x_tr: Any, y_tr: list[str], x_va: Any, y_va: list[str], seed: int
) -> tuple[LogisticRegression, float, list[dict[str, float]]]:
    best: tuple[float, LogisticRegression | None, float] = (-1.0, None, 0.0)
    grid = []
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(x_tr, y_tr)
        f1 = float(f1_score(y_va, clf.predict(x_va), average="macro", zero_division=0))
        grid.append({"C": c, "val_macro_f1": round(f1, 4)})
        if f1 > best[0]:
            best = (f1, clf, c)
    assert best[1] is not None
    return best[1], best[2], grid


class TfidfLR:
    name = "tfidf_lr"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=150_000,
            sublinear_tf=True,
        )

    def fit(
        self, x_tr: list[str], y_tr: list[str], x_va: list[str], y_va: list[str]
    ) -> dict[str, Any]:
        t0 = time.time()
        a = hstack([self.word.fit_transform(x_tr), self.char.fit_transform(x_tr)]).tocsr()
        b = self._x(x_va)
        self.clf, c, grid = _fit_best_c(a, y_tr, b, y_va, self.seed)
        return {"best_C": c, "grid": grid, "train_seconds": round(time.time() - t0, 1)}

    def _x(self, x: list[str]) -> Any:
        return hstack([self.word.transform(x), self.char.transform(x)]).tocsr()

    def predict_proba(self, x: list[str]) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(self._x(x)))

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.clf.classes_]


class EmbeddingLR:
    name = "e5_lr"

    def __init__(self, seed: int = 0, max_chars: int = 2000) -> None:
        from jobnlp.gap.embed import get_embedder

        self.seed = seed
        self.max_chars = max_chars
        self.embedder = get_embedder()

    def _x(self, x: list[str]) -> np.ndarray:
        return self.embedder.encode([t[: self.max_chars] for t in x])

    def fit(
        self, x_tr: list[str], y_tr: list[str], x_va: list[str], y_va: list[str]
    ) -> dict[str, Any]:
        t0 = time.time()
        self.clf, c, grid = _fit_best_c(self._x(x_tr), y_tr, self._x(x_va), y_va, self.seed)
        return {"best_C": c, "grid": grid, "train_seconds": round(time.time() - t0, 1)}

    def predict_proba(self, x: list[str]) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(self._x(x)))

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.clf.classes_]
