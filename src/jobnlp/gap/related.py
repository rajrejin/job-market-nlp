"""Threshold for 'related' skills (embedding similarity), chosen on the gold dev half.

Positive pairs: a gold surface form and the preferred ESCO label it normalises to (when the two
differ). Negative pairs: the same surface form and the preferred label of a different ESCO skill
found in the dev data. The threshold maximises F1 on these pairs.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from jobnlp.gap.match import Embedder
from jobnlp.skills.evaluate import GoldItem


def tune_related_threshold(
    dev: list[GoldItem],
    canon: Callable[[str, str], str],
    embedder: Embedder,
    esco: pd.DataFrame,
    seed: int = 0,
) -> dict[str, Any]:
    pref = dict(zip(esco.uri, esco.pref_de.fillna(esco.pref_en), strict=True))
    pos: list[tuple[str, str]] = []
    ids: list[str] = []
    for it in dev:
        for a, b, _, _ in it.spans:
            s = it.text[a:b]
            key = canon(s, it.lang)
            if key in pref and pref[key].lower() != s.lower():
                pos.append((s, pref[key]))
                ids.append(key)
    rng = random.Random(seed)
    uniq = sorted(set(ids))
    neg = []
    for (s, _), key in zip(pos, ids, strict=True):
        others = [u for u in uniq if u != key]
        if others:
            neg.append((s, pref[rng.choice(others)]))
    if not pos or not neg:
        return {
            "threshold": 0.9,
            "note": "too few pairs on dev; default used",
            "n_pos": len(pos),
            "n_neg": len(neg),
        }
    sim = lambda pairs: [embedder.similarity([a], [b])[0][0] for a, b in pairs]  # noqa: E731
    sp, sn = np.array(sim(pos)), np.array(sim(neg))
    best = (0.0, 0.9)
    for t in np.arange(0.70, 0.99, 0.01):
        tp, fp, fn = (sp >= t).sum(), (sn >= t).sum(), (sp < t).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (float(f1), float(t))
    return {
        "threshold": round(best[1], 2),
        "dev_pair_f1": round(best[0], 4),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "mean_sim_pos": round(float(sp.mean()), 4),
        "mean_sim_neg": round(float(sn.mean()), 4),
    }
