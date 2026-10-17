"""Transparent match score: weighted coverage of the posting's required skills.

score = sum(w_i * c_i) / sum(w_i)
  w_i = 1.0 for a must-have skill, 0.4 for a nice-to-have skill
  c_i = 1.0 if the CV covers the skill (same canonical skill), 0.5 if the CV only has a
        related skill (embedding similarity above the threshold), 0.0 if missing

It is a heuristic readiness indicator, not a prediction of recruiter decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = {"must": 1.0, "nice": 0.4}
CREDIT = {"covered": 1.0, "related": 0.5, "missing": 0.0}


@dataclass
class ScoredSkill:
    key: str
    label: str
    necessity: str
    status: str  # covered | related | missing
    weight: float
    credit: float


def match_score(items: list[tuple[str, str, str, str]]) -> tuple[float | None, list[ScoredSkill]]:
    """items: (key, label, necessity, status). Returns (score in [0, 1] or None, breakdown)."""
    rows = [ScoredSkill(k, lab, nec, st, WEIGHTS[nec], CREDIT[st]) for k, lab, nec, st in items]
    total = sum(r.weight for r in rows)
    if total == 0:
        return None, rows
    return round(sum(r.weight * r.credit for r in rows) / total, 4), rows
