"""Demand of skills among postings similar to a job description.

Similar postings are the k corpus postings whose extracted skill sets overlap most with the
job's skill set (Jaccard similarity). A skill's demand is the share of those postings that
mention it. Because neighbours are chosen by shared skills, the job's own skills are favoured;
the measure is used only to rank missing skills against each other.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pandas as pd

from jobnlp.config import ROOT

SKILLS = ROOT / "data" / "processed" / "market_skills.parquet"


@lru_cache(maxsize=1)
def _posting_sets(path: Path = SKILLS) -> dict[str, frozenset[str]]:
    df = pd.read_parquet(path, columns=["posting_key", "skill_id"])
    return {str(k): frozenset(g) for k, g in df.groupby("posting_key").skill_id}


def demand_shares(
    job_keys: list[str], sets: dict[str, frozenset[str]], k: int = 100
) -> dict[str, float]:
    job = set(job_keys)
    if not job or not sets:
        return {}
    scored = []
    for key, s in sets.items():
        inter = len(job & s)
        if inter:
            scored.append((inter / len(job | s), key))
    scored.sort(reverse=True)
    neighbours = [sets[key] for _, key in scored[:k]]
    if not neighbours:
        return {}
    return {sk: round(sum(sk in n for n in neighbours) / len(neighbours), 4) for sk in job}


def demand_function(path: Path = SKILLS) -> Callable[[list[str]], dict[str, float]] | None:
    """None when the market skill table has not been built (run `jobnlp market`)."""
    if not path.exists():
        return None
    sets = _posting_sets(path)
    return lambda keys: demand_shares(keys, sets)
