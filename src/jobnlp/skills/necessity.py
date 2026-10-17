"""Must-have versus nice-to-have, from necessity cues in the line or the governing heading."""

from __future__ import annotations

import re

NICE = re.compile(
    r"(wünschenswert|von vorteil|vorteilhaft|idealerweise|ideal wäre|gerne\b|gern\b|ein plus|"
    r"ist ein plus|kein muss|keine voraussetzung|schön zu haben|optional|bonus|nice[- ]to[- ]have|"
    r"a plus|is a plus|preferred|preferably|ideally|desirable|especially valuable|"
    r"what sets you apart|would be (?:great|beneficial|an advantage)|an advantage|beneficial|"
    r"not (?:a )?requirement|plus:|\bplus\b)",
    re.IGNORECASE,
)
# negated must cues ("nicht zwingend erforderlich", "not required") are removed before MUST
NEGATED = re.compile(
    r"\b(?:nicht|keine?|not(?:\s+a)?)\s+(?:\w+\s+){0,1}(?:zwingend|erforderlich|voraussetzung|"
    r"notwendig|required|mandatory|requirement|must)\w*(?:\s+erforderlich)?",
    re.IGNORECASE,
)
MUST = re.compile(
    r"(zwingend|erforderlich|voraussetzung|muss\b|must[- ]have|mandatory|required|essential|"
    r"notwendig|unbedingt)",
    re.IGNORECASE,
)
HEADING_MAX = 60


def _is_heading(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith("## ")
        or (0 < len(s) <= HEADING_MAX and s.endswith(":"))
        or (0 < len(s) <= 30 and not s.startswith("-") and len(s.split()) <= 4)
    )


def necessity_at(text: str, offset: int) -> str:
    """'nice' or 'must' for the skill mention starting at character `offset`.

    The clause containing the mention decides first (a 'nice' cue in the same sentence or
    list line; an explicit 'must' cue overrides); otherwise the closest heading above it."""
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    line = text[line_start : line_end if line_end >= 0 else len(text)]
    # clause = sentence within the line that contains the offset
    rel = offset - line_start
    starts = [0] + [m.end() for m in re.finditer(r"[.;!?]\s", line) if m.end() <= rel]
    ends = [m.start() + 1 for m in re.finditer(r"[.;!?](\s|$)", line) if m.start() >= rel]
    clause = line[starts[-1] : ends[0] if ends else len(line)]
    negated = NEGATED.search(clause) is not None
    plain = NEGATED.sub(" ", clause)
    if (NICE.search(clause) or negated) and not MUST.search(plain):
        return "nice"
    if MUST.search(plain):
        return "must"
    for prev in reversed(text[:line_start].split("\n")):
        if _is_heading(prev):
            return "nice" if NICE.search(prev) else "must"
    return "must"
