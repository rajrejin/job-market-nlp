"""Local LLM suggestions (Ollama), structured output validated with Pydantic, guarded.

The prompt contains only the target skill, the posting lines that mention it and a few CV
bullets. Nothing is sent anywhere except the local Ollama server.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from jobnlp.gap.guards import check_facts, check_no_new_skills

DEFAULT_MODEL = "qwen2.5:3b-instruct"
MAX_RETRIES = 2

SYSTEM = (
    "You help a job seeker present their real experience. Rules: never invent experience, "
    "employers, numbers, dates, tools or skills. Use only facts written in the CV bullets you "
    "are given. If one bullet already shows weak evidence of the target skill, rewrite only "
    "that bullet so the evidence is visible in the posting's vocabulary, keeping every fact "
    "unchanged. If no bullet shows evidence, set evidence_found to false, leave rewrite empty, "
    "and propose one concrete project, course or task the person could do to build the skill. "
    "Answer in the language of the CV bullets. Reply with JSON only."
)


class Suggestion(BaseModel):
    evidence_found: bool
    bullet_index: int | None = Field(default=None, ge=0)
    rewrite: str | None = Field(default=None, max_length=400)
    suggestion: str | None = Field(default=None, max_length=400)


@dataclass
class GuardedSuggestion:
    skill_key: str
    skill_label: str
    evidence_found: bool
    original_bullet: str | None
    rewrite: str | None
    suggestion: str | None
    attempts: int
    guard_passed: bool
    fallback: bool
    violations: list[str] = field(default_factory=list)
    history: list[list[str]] = field(default_factory=list)  # violations per attempt

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class LocalLLM:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None) -> None:
        import ollama

        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def available(self) -> bool:
        try:
            names = {m.model for m in self.client.list().models}
        except Exception:
            return False
        return self.model in names

    def ask(self, prompt: str) -> str:
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            format=Suggestion.model_json_schema(),
            options={"temperature": 0, "seed": 0, "num_predict": 300},
        )
        return str(r.message.content)


def build_prompt(
    skill_label: str,
    posting_lines: list[str],
    bullets: list[str],
    feedback: list[str] | None = None,
) -> str:
    parts = [
        f"Target skill from the job posting: {skill_label}",
        "Posting lines that mention it:",
        *[f"- {line}" for line in posting_lines[:3]],
        "CV bullets (index: text):",
        *[f"{i}: {b}" for i, b in enumerate(bullets)],
    ]
    if feedback:
        parts += ["Your previous answer broke these rules, fix them:", *feedback]
    return "\n".join(parts)


def posting_lines_for(job_text: str, surface: str, limit: int = 3) -> list[str]:
    lines = [
        ln.strip("-• ").strip() for ln in job_text.split("\n") if surface.lower() in ln.lower()
    ]
    return [ln[:300] for ln in lines[:limit]]


def guarded_suggestion(
    ask: Callable[[str], str],
    skill_key: str,
    skill_label: str,
    skill_surface: str,
    job_text: str,
    bullets: list[str],
    cv_keys: set[str],
    skill_keys: Callable[[str], set[str]],
    orgs: Callable[[str], set[str]] | None = None,
) -> GuardedSuggestion:
    """Generate, check, retry up to MAX_RETRIES times, then fall back to no rewrite."""
    lines = posting_lines_for(job_text, skill_surface)
    feedback: list[str] = []
    last: Suggestion | None = None
    violations: list[str] = []
    history: list[list[str]] = []
    for attempt in range(1, MAX_RETRIES + 2):
        raw = ask(build_prompt(skill_label, lines, bullets, feedback))
        try:
            last = Suggestion.model_validate(json.loads(_json_block(raw)))
        except (ValidationError, json.JSONDecodeError, ValueError):
            violations = ["output was not valid JSON for the schema"]
            history.append(violations)
            feedback = violations
            continue
        violations = _violations(last, bullets, cv_keys, skill_key, skill_keys, orgs)
        history.append(violations)
        if not violations:
            bullet = (
                bullets[last.bullet_index]
                if last.evidence_found and last.bullet_index is not None
                else None
            )
            return GuardedSuggestion(
                skill_key,
                skill_label,
                last.evidence_found,
                bullet,
                last.rewrite if last.evidence_found else None,
                last.suggestion,
                attempt,
                True,
                False,
                history=history,
            )
        feedback = violations
    return GuardedSuggestion(
        skill_key,
        skill_label,
        False,
        None,
        None,
        (last.suggestion if last is not None and _suggestion_clean(last) else None),
        MAX_RETRIES + 1,
        False,
        True,
        violations,
        history,
    )


def _json_block(raw: str) -> str:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object")
    return m.group(0)


def _suggestion_clean(s: Suggestion) -> bool:
    return bool(s.suggestion) and len(s.suggestion or "") <= 400


def _violations(
    s: Suggestion,
    bullets: list[str],
    cv_keys: set[str],
    skill_key: str,
    skill_keys: Callable[[str], set[str]],
    orgs: Callable[[str], set[str]] | None,
) -> list[str]:
    if not s.evidence_found:
        return [] if s.suggestion else ["no evidence and no suggestion given"]
    if s.bullet_index is None or s.bullet_index >= len(bullets) or not s.rewrite:
        return ["evidence_found without a valid bullet_index and rewrite"]
    bullet = bullets[s.bullet_index]
    out: list[str] = []
    for report in (
        check_no_new_skills(s.rewrite, bullet, cv_keys, skill_key, True, skill_keys),
        check_facts(s.rewrite, bullet, orgs),
    ):
        out += report.violations
    return out
