"""Evaluation of the CV tool without CV data.

(a) Matching sanity on real postings: a posting compared with itself should score near 1;
    pairs with the same platform label should score higher than pairs with different labels
    (AUC with a bootstrap interval over pairs).
(c) Guard evaluation: real requirement bullets from postings stand in for CV bullets; the LLM
    rewrite step runs on them and every attempt is checked by the guards.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jobnlp.gap.llm import GuardedSuggestion, LocalLLM, guarded_suggestion
from jobnlp.gap.match import Embedder, analyse
from jobnlp.skills.extract import SkillExtractor


def sanity_auc(
    labelled: pd.DataFrame,
    extractor: SkillExtractor,
    embedder: Embedder | None,
    n_pairs: int = 300,
    n_self: int = 100,
    seed: int = 0,
    related_threshold: float = 0.9,
) -> dict[str, Any]:
    """labelled: rows with title, description, label (the classifier dataset)."""
    rng = random.Random(seed)
    rows = labelled.reset_index(drop=True)
    text = (rows.title.fillna("") + "\n" + rows.description.fillna("")).tolist()
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(rows.label):
        by_label.setdefault(lab, []).append(i)
    labels = [lab for lab, idx in by_label.items() if len(idx) >= 2]
    same: list[tuple[int, int]] = []
    diff: list[tuple[int, int]] = []
    while len(same) < n_pairs:
        lab = rng.choice(labels)
        a, b = rng.sample(by_label[lab], 2)
        same.append((a, b))
    while len(diff) < n_pairs:
        a, b = rng.sample(range(len(rows)), 2)
        if rows.label[a] != rows.label[b]:
            diff.append((a, b))

    def score(a: int, b: int) -> float | None:
        return analyse(text[a], text[b], extractor, embedder, related_threshold).score

    t0 = time.time()
    s_same = [score(a, b) for a, b in same]
    s_diff = [score(a, b) for a, b in diff]
    pairs = [(s, 1) for s in s_same if s is not None] + [(s, 0) for s in s_diff if s is not None]
    y = np.array([t for _, t in pairs])
    x = np.array([s for s, _ in pairs])
    auc = float(roc_auc_score(y, x))
    nprng = np.random.default_rng(seed)
    boots = []
    for _ in range(1000):
        bi = nprng.integers(0, len(y), len(y))
        if len(set(y[bi])) == 2:
            boots.append(roc_auc_score(y[bi], x[bi]))
    self_idx = rng.sample(range(len(rows)), min(n_self, len(rows)))
    self_scores = [s for s in (score(i, i) for i in self_idx) if s is not None]
    return {
        "pairs_same_label": len([s for s in s_same if s is not None]),
        "pairs_different_label": len([s for s in s_diff if s is not None]),
        "auc_same_vs_different": round(auc, 4),
        "auc_95ci": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
        "mean_score_same": round(float(np.mean([s for s in s_same if s is not None])), 4),
        "mean_score_different": round(float(np.mean([s for s in s_diff if s is not None])), 4),
        "self_match_n": len(self_scores),
        "self_match_mean": round(float(np.mean(self_scores)), 4),
        "self_match_min": round(float(np.min(self_scores)), 4),
        "seconds": round(time.time() - t0, 1),
        "related_threshold": related_threshold,
        "embeddings_used": embedder is not None,
    }


def guard_eval(
    postings: pd.DataFrame,
    extractor: SkillExtractor,
    llm: LocalLLM,
    n: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    """postings: rows with requirement text in `req` and language in `lang`."""
    rng = random.Random(seed)
    records: list[GuardedSuggestion] = []
    t0 = time.time()
    for _, row in postings.sample(frac=1, random_state=seed).iterrows():
        if len(records) >= n:
            break
        lines = [ln[2:].strip() for ln in row.req.split("\n") if ln.startswith("- ")]
        lines = [ln for ln in lines if 30 <= len(ln) <= 300]
        if len(lines) < 4:
            continue
        lang = "en" if row.lang == "en" else "de"
        keys = lambda t, lang=lang: {m.skill_id for m in extractor.extract(t, lang, use_ner=False)}  # noqa: E731
        cv_bullets = lines[:3]
        cv_keys = set().union(*(keys(b) for b in cv_bullets))
        other = [
            m
            for m in extractor.extract("\n".join(lines[3:]), lang, use_ner=False)
            if m.skill_id not in cv_keys
        ]
        if not other:
            continue
        target = rng.choice(other)
        s = guarded_suggestion(
            llm.ask,
            target.skill_id,
            target.label,
            target.text,
            "\n".join(lines[3:]),
            cv_bullets,
            cv_keys,
            keys,
        )
        records.append(s)
    first_bad = [r for r in records if r.history and r.history[0]]
    first_unsupported = [
        r
        for r in records
        if r.history and any(v.startswith("unsupported skill") for v in r.history[0])
    ]
    first_facts = [
        r
        for r in records
        if r.history and any(v.startswith(("number", "organisation")) for v in r.history[0])
    ]
    first_json = [r for r in records if r.history and any("JSON" in v for v in r.history[0])]
    return {
        "n_cases": len(records),
        "model": llm.model,
        "first_attempt_violation_rate": round(len(first_bad) / max(1, len(records)), 4),
        "first_attempt_unsupported_skill_rate": round(
            len(first_unsupported) / max(1, len(records)), 4
        ),
        "first_attempt_fact_violation_rate": round(len(first_facts) / max(1, len(records)), 4),
        "first_attempt_invalid_json_rate": round(len(first_json) / max(1, len(records)), 4),
        "final_pass_rate": round(sum(r.guard_passed for r in records) / max(1, len(records)), 4),
        "fallback_rate": round(sum(r.fallback for r in records) / max(1, len(records)), 4),
        "evidence_found_rate": round(
            sum(r.evidence_found for r in records) / max(1, len(records)), 4
        ),
        "mean_attempts": round(
            float(np.mean([r.attempts for r in records])) if records else 0.0, 3
        ),
        "after_guards_unsupported_rate_note": (
            "0 by construction for skills the extractor detects: outputs that fail are retried, "
            "then replaced by the deterministic fallback"
        ),
        "seconds": round(time.time() - t0, 1),
    }
