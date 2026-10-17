"""End-to-end CV gap analysis: deterministic gap and score, then guarded LLM suggestions."""

from __future__ import annotations

from typing import Any

from jobnlp.gap.embed import get_embedder
from jobnlp.gap.llm import LocalLLM, guarded_suggestion
from jobnlp.gap.match import GapResult, analyse
from jobnlp.gap.parse_cv import bullets
from jobnlp.skills.extract import SkillExtractor
from jobnlp.text.lang import detect

DISCLAIMER = (
    "The match score is a heuristic readiness indicator (weighted coverage of the skills the "
    "posting names). It does not predict recruiter decisions or applicant tracking systems."
)


def skill_key_fn(extractor: SkillExtractor, lang: str):  # type: ignore[no-untyped-def]
    def keys(text: str) -> set[str]:
        return {m.skill_id for m in extractor.extract(text, lang, use_ner=False)}

    return keys


def run_gap(
    job_text: str,
    cv_text: str,
    extractor: SkillExtractor,
    with_embeddings: bool = True,
    related_threshold: float = 0.9,
    llm: LocalLLM | None = None,
    max_suggestions: int = 3,
) -> dict[str, Any]:
    embedder = None
    notes: list[str] = []
    if with_embeddings:
        try:
            embedder = get_embedder()
        except Exception:
            notes.append("embedding model could not be loaded: related matching skipped")
    gap: GapResult = analyse(job_text, cv_text, extractor, embedder, related_threshold)
    result = gap.as_dict()
    result["notes"] = gap.notes + notes
    result["disclaimer"] = DISCLAIMER
    result["suggestions"] = []
    if llm is None or not llm.available():
        result["notes"].append(
            "local LLM unavailable: suggestions skipped, the gap and score above are complete"
        )
        return result
    cv_lang = "en" if detect(cv_text) == "en" else "de"
    keys = skill_key_fn(extractor, cv_lang)
    cv_bullets = bullets(cv_text)[:40]
    cv_keys = keys(cv_text)
    gap_keys = {s.key for s in gap.skills}
    for skill in gap.missing[:max_suggestions]:
        if skill.key not in gap_keys:  # guard 1: only skills from the deterministic gap
            continue
        candidates = cv_bullets
        if embedder is not None and cv_bullets:
            sims = embedder.similarity([skill.label], cv_bullets)[0]
            order = sorted(range(len(cv_bullets)), key=lambda i: -sims[i])
            candidates = [cv_bullets[i] for i in order[:5]]
        s = guarded_suggestion(
            llm.ask, skill.key, skill.label, skill.surface, job_text, candidates[:5], cv_keys, keys
        )
        result["suggestions"].append(s.as_dict())
    return result
