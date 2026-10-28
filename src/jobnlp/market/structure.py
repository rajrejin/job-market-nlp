"""Skill co-occurrence (PMI), role profiles (NMF) and their stability across seeds."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score


def binary_matrix(
    skills: pd.DataFrame, top_n: int = 150
) -> tuple[np.ndarray, list[str], list[str]]:
    top = skills.groupby("skill_id").posting_key.nunique().nlargest(top_n).index.tolist()
    sub = skills[skills.skill_id.isin(top)]
    keys = sorted(sub.posting_key.unique())
    row = {k: i for i, k in enumerate(keys)}
    col = {s: j for j, s in enumerate(top)}
    m = np.zeros((len(keys), len(top)), dtype=np.float32)
    for pk, sid in zip(sub.posting_key, sub.skill_id, strict=True):
        m[row[pk], col[sid]] = 1.0
    return m, keys, top


def pmi_edges(skills: pd.DataFrame, n_postings: int, min_pair: int = 10) -> pd.DataFrame:
    """Pointwise mutual information for skill pairs that co-occur in at least `min_pair`
    postings. PMI = log2(p(a,b) / (p(a) p(b)))."""
    per_posting = skills.groupby("posting_key").skill_id.apply(lambda s: sorted(set(s)))
    single = skills.groupby("skill_id").posting_key.nunique()
    pairs: dict[tuple[str, str], int] = {}
    for ids in per_posting:
        for a, b in combinations(ids, 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    labels = skills.drop_duplicates("skill_id").set_index("skill_id").label
    rows = []
    for (a, b), k in pairs.items():
        if k < min_pair:
            continue
        pmi = math.log2((k / n_postings) / ((single[a] / n_postings) * (single[b] / n_postings)))
        rows.append(
            {"skill_a": labels[a], "skill_b": labels[b], "co_postings": k, "pmi": round(pmi, 3)}
        )
    return pd.DataFrame(rows).sort_values("pmi", ascending=False).reset_index(drop=True)


def role_profiles(
    m: np.ndarray, skill_labels: list[str], k: int = 6, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
) -> dict[str, Any]:
    """NMF with k components; profiles from seed 0, stability = mean pairwise adjusted Rand
    index of posting-to-profile assignments across seeds."""
    keep = m.sum(axis=1) > 0
    x = m[keep]
    assignments = []
    profiles: list[dict[str, Any]] = []
    for seed in seeds:
        # random initialisation: the seed changes the start, so agreement across seeds is a
        # real stability test (the SVD-based "nndsvda" start is the same for every seed)
        model = NMF(n_components=k, init="random", random_state=seed, max_iter=800)
        w = model.fit_transform(x)
        assignments.append(w.argmax(axis=1))
        if seed == seeds[0]:
            h = model.components_
            sizes = np.bincount(w.argmax(axis=1), minlength=k)
            for c in range(k):
                top = np.argsort(-h[c])[:10]
                profiles.append(
                    {
                        "profile": c,
                        "postings": int(sizes[c]),
                        "top_skills": [skill_labels[j] for j in top],
                    }
                )
    aris = [adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)]
    return {
        "k": k,
        "postings": int(x.shape[0]),
        "profiles": profiles,
        "stability_mean_ari": round(float(np.mean(aris)), 4),
        "stability_min_ari": round(float(np.min(aris)), 4),
        "seeds": list(seeds),
    }
