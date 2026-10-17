"""Rule-based section splitter for German and English postings.

Input is cleaned text where markup headings start with '## '. Plain-text input (pasted
postings, CVs) also works: short lines that match a cue or end with a colon are headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = ("company", "tasks", "requirements", "offer", "other")

CUES: dict[str, re.Pattern[str]] = {
    "requirements": re.compile(
        r"(profil|qualifikation|anforderung|voraussetzung|das bringst du mit|das bringen sie mit|"
        r"was du mitbringst|was sie mitbringen|du bringst mit|sie bringen mit|deine skills|"
        r"requirements|qualifications|who you are|what you bring|what we.?re looking for|"
        r"what we are looking for|about you|your profile|you have|you bring|must.?have|"
        r"nice.?to.?have|skills|experience|wünschenswert|kenntnisse|ideal candidate|"
        r"was wir erwarten|was wir uns wünschen|dein background)",
        re.IGNORECASE,
    ),
    "tasks": re.compile(
        r"(aufgaben|aufgabengebiet|tätigkeit|deine rolle|ihre rolle|dein job|deine mission|"
        r"das erwartet dich|was dich erwartet|was sie erwartet|verantwortung|"
        r"what you.?ll do|what you will do|what you.?ll be doing|what you.?ll own|"
        r"responsibilities|the role|your role|about the role|tasks|your mission|the job|"
        r"your impact|day to day|in this role|the opportunity)",
        re.IGNORECASE,
    ),
    "offer": re.compile(
        r"(benefits|wir bieten|was wir bieten|das bieten wir|unser angebot|deine vorteile|"
        r"ihre vorteile|warum wir|what we offer|why join|perks|compensation|what.?s in it|"
        r"we offer|our offer|vergütung|gehalt|salary)",
        re.IGNORECASE,
    ),
    "company": re.compile(
        r"(über uns|ueber uns|wer wir sind|das sind wir|unternehmen|about us|who we are|"
        r"about the company|about the team|our company|our mission|^about \w+)",
        re.IGNORECASE,
    ),
    "other": re.compile(
        r"(bewerbung|application|interview|hiring process|kontakt|contact|location|standort|"
        r"diversity|equal opportunit|chancengleichheit)",
        re.IGNORECASE,
    ),
}
ORDER = ("requirements", "tasks", "offer", "company", "other")
MAX_PLAIN_HEADING = 60


@dataclass
class Section:
    label: str
    heading: str
    text: str


def classify_heading(heading: str) -> str | None:
    h = heading.strip().strip(":").strip()
    for label in ORDER:
        if CUES[label].search(h):
            return label
    return None


def _heading_of(line: str) -> str | None:
    if line.startswith("## "):
        return line[3:].strip()
    s = line.strip()
    if 0 < len(s) <= MAX_PLAIN_HEADING and not s.startswith("- "):
        if s.endswith(":") or (classify_heading(s) and len(s.split()) <= 6):
            return s
    return None


def split_sections(text: str) -> list[Section]:
    """Split into labelled sections. Text before the first recognised heading is 'company'
    (intro); unrecognised headings continue the current section."""
    sections: list[Section] = [Section("company", "", "")]
    body: list[list[str]] = [[]]
    for line in text.split("\n"):
        head = _heading_of(line)
        label = classify_heading(head) if head is not None else None
        if label is not None and head is not None:
            sections.append(Section(label, head, ""))
            body.append([])
            continue
        body[-1].append(line[3:] if line.startswith("## ") else line)
    for sec, lines in zip(sections, body, strict=True):
        sec.text = "\n".join(lines).strip()
    return [s for s in sections if s.text or s.heading]


def section_text(sections: list[Section], label: str) -> str:
    return "\n".join(s.text for s in sections if s.label == label)
