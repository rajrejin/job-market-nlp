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


def _display_label(mention: SkillMention) -> str:
    """Dictionary skills show their canonical label, NER-only finds show the text as written."""
    return mention.label if mention.source in ("esco", "mined") else mention.text


def group_mentions(mentions: list[SkillMention], text_len: int) -> dict[str, JobSkill]:
    """One entry per canonical skill; must-have wins over nice-to-have."""
    skills: dict[str, JobSkill] = {}
    for mention in mentions:
        key = mention.skill_id
        if key in skills:
            if mention.necessity == "must":
                skills[key].necessity = "must"
            continue
        skills[key] = JobSkill(
            key=key,
            label=_display_label(mention),
            surface=mention.text,
            necessity=mention.necessity,
            first_pos=mention.start / max(1, text_len),
            source=mention.source,
        )
    return skills


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
    """Compare a posting with a CV in four steps.

    1. Extract skills from both texts.
    2. A job skill is "covered" if the CV contains the same canonical skill.
    3. Otherwise it is "related" if some CV skill is close enough in embedding space.
    4. Everything else is "missing" and gets ranked by importance.
    """
    job_lang, cv_lang = detect(job_text), detect(cv_text)

    def extractor_lang(detected: str) -> str:
        return "en" if detected == "en" else "de"  # the extractor only knows these two

    job_mentions = extractor.extract(job_text, extractor_lang(job_lang))
    cv_mentions = extractor.extract(cv_text, extractor_lang(cv_lang))
    job_skills = list(group_mentions(job_mentions, len(job_text)).values())
    cv_keys = {m.skill_id for m in cv_mentions}
    cv_labels = sorted({_display_label(m) for m in cv_mentions})
    notes: list[str] = []

    for skill in job_skills:
        if skill.key in cv_keys:
            skill.status = "covered"

    uncovered = [s for s in job_skills if s.status == "missing"]
    if embedder is not None and uncovered and cv_labels:
        similarities = embedder.similarity([s.label for s in uncovered], cv_labels)
        for skill, row in zip(uncovered, similarities, strict=True):
            closest = max(range(len(row)), key=lambda j: row[j])
            if row[closest] >= related_threshold:
                skill.status = "related"
                skill.related_to = cv_labels[closest]
                skill.similarity = round(row[closest], 4)
    elif embedder is None:
        notes.append("embedding model unavailable: related-skill matching skipped")

    # Optional: how often similar postings ask for each skill, used to rank what is missing.
    if demand is not None:
        shares = demand([s.key for s in job_skills])
        for skill in job_skills:
            skill.demand = shares.get(skill.key)
    for skill in job_skills:
        skill.importance = importance(skill)

    score, breakdown = match_score([(s.key, s.label, s.necessity, s.status) for s in job_skills])
    return GapResult(job_lang, cv_lang, job_skills, cv_labels, score, breakdown, notes)
