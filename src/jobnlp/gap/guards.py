"""Guards against invented experience in LLM rewrites.

1. Every skill the LLM talks about must come from the deterministic gap computation.
2. A rewrite may only name skills found in the original bullet, elsewhere in the CV, or the
   target skill itself when the model reported evidence for it in the bullet.
3. Numbers, dates and organisation names in a rewrite must appear in the original bullet.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s?%?")


@dataclass
class GuardReport:
    passed: bool
    violations: list[str] = field(default_factory=list)


def check_skill_in_gap(skill_key: str, gap_keys: set[str]) -> GuardReport:
    ok = skill_key in gap_keys
    return GuardReport(ok, [] if ok else [f"skill {skill_key!r} is not in the computed gap"])


def check_no_new_skills(
    rewrite: str,
    bullet: str,
    cv_keys: set[str],
    target_key: str,
    evidence_found: bool,
    skill_keys: Callable[[str], set[str]],
) -> GuardReport:
    allowed = skill_keys(bullet) | cv_keys | ({target_key} if evidence_found else set())
    new = sorted(skill_keys(rewrite) - allowed)
    return GuardReport(not new, [f"unsupported skill: {k}" for k in new])


def check_facts(
    rewrite: str, bullet: str, orgs: Callable[[str], set[str]] | None = None
) -> GuardReport:
    """Numbers (incl. years and percentages) and organisation names must exist in the bullet."""
    norm = lambda s: re.sub(r"\s", "", s)  # noqa: E731
    bullet_numbers = {norm(n) for n in NUMBER.findall(bullet)}
    violations = [
        f"number not in original: {n.strip()}"
        for n in NUMBER.findall(rewrite)
        if norm(n) not in bullet_numbers
    ]
    if orgs is not None:
        low = bullet.lower()
        violations += [
            f"organisation not in original: {o}" for o in orgs(rewrite) if o.lower() not in low
        ]
    return GuardReport(not violations, violations)
