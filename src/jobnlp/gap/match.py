"""Deterministic gap computation between a job description and a CV."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from jobnlp.gap.score import WEIGHTS, ScoredSkill, match_score
from jobnlp.skills.extract import SkillExtractor, SkillMention
from jobnlp.text.lang import detect


class Embedder(Protocol):
    def similarity(self, a: list[str], b: list[str]) -> list[list[float]]: ...


@dataclass
class JobSkill:
    key: str
    label: str
    surface: str
    necessity: str
    first_pos: float  # relative position of the first mention in the posting (0 to 1)
    source: str
    status: str = "missing"
    related_to: str | None = None
    similarity: float | None = None
    demand: float | None = None
    importance: float = 0.0


@dataclass
class GapResult:
    job_lang: str
    cv_lang: str
    skills: list[JobSkill]
    cv_skill_labels: list[str]
    score: float | None
    breakdown: list[ScoredSkill]
    notes: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[JobSkill]:
        return sorted(
            (s for s in self.skills if s.status == "missing"), key=lambda s: -s.importance
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_lang": self.job_lang,
            "cv_lang": self.cv_lang,
            "score": self.score,
            "skills": [s.__dict__ for s in self.skills],
            "missing_ranked": [s.key for s in self.missing],
            "cv_skills": self.cv_skill_labels,
            "breakdown": [b.__dict__ for b in self.breakdown],
            "notes": self.notes,
        }


def group_mentions(mentions: list[SkillMention], text_len: int) -> dict[str, JobSkill]:
    """One entry per canonical skill; must-have wins over nice-to-have."""
    out: dict[str, JobSkill] = {}
    for m in mentions:
        key = m.skill_id
        nec = m.necessity
        if key in out:
            if nec == "must":
                out[key].necessity = "must"
            continue
        label = m.label if m.source in ("esco", "mined") else m.text
        out[key] = JobSkill(key, label, m.text, nec, m.start / max(1, text_len), m.source)
    return out


def importance(skill: JobSkill) -> float:
    """Rank for missing skills: necessity weight x (0.5 + demand share among similar
    postings, when known) x position (skills named in the first third count fully)."""
    demand = skill.demand if skill.demand is not None else 0.0
    position = 1.0 if skill.first_pos <= 1 / 3 else 0.8
    return round(WEIGHTS[skill.necessity] * (0.5 + demand) * position, 4)


def analyse(
    job_text: str,
    cv_text: str,
    extractor: SkillExtractor,
    embedder: Embedder | None = None,
    related_threshold: float = 0.9,
    demand: Callable[[list[str]], dict[str, float]] | None = None,
) -> GapResult:
    job_lang, cv_lang = detect(job_text), detect(cv_text)
    lang = lambda x: "en" if x == "en" else "de"  # noqa: E731
    job = group_mentions(extractor.extract(job_text, lang(job_lang)), len(job_text))
    cv_mentions = extractor.extract(cv_text, lang(cv_lang))
    cv_keys = {m.skill_id for m in cv_mentions}
    cv_labels = sorted({m.label if m.source in ("esco", "mined") else m.text for m in cv_mentions})
    notes: list[str] = []
    for s in job.values():
        if s.key in cv_keys:
            s.status = "covered"
    open_skills = [s for s in job.values() if s.status == "missing"]
    if embedder is not None and open_skills and cv_labels:
        sims = embedder.similarity([s.label for s in open_skills], cv_labels)
        for s, row in zip(open_skills, sims, strict=True):
            best = max(range(len(row)), key=lambda j: row[j])
            if row[best] >= related_threshold:
                s.status, s.related_to, s.similarity = (
                    "related",
                    cv_labels[best],
                    round(row[best], 4),
                )
    elif embedder is None:
        notes.append("embedding model unavailable: related-skill matching skipped")
    if demand is not None:
        shares = demand([s.key for s in job.values()])
        for s in job.values():
            s.demand = shares.get(s.key)
    for s in job.values():
        s.importance = importance(s)
    score, breakdown = match_score([(s.key, s.label, s.necessity, s.status) for s in job.values()])
    return GapResult(job_lang, cv_lang, list(job.values()), cv_labels, score, breakdown, notes)
