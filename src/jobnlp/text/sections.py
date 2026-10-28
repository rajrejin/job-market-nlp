"""Rule-based section splitter for German and English postings.

Input is cleaned text where markup headings start with '## '. Plain-text input (pasted
postings, CVs) also works: short lines that match a cue or end with a colon are headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = ("company", "tasks", "requirements", "offer", "other")

# Cue lists, extended on the development half of the splitter evaluation (postings s000 to
# s049, results/section_splitter_eval.json); the held-out half was not used to change them.
CUES: dict[str, re.Pattern[str]] = {
    "requirements": re.compile(
        r"(\bprofil|qualifikation|anforderung|voraussetzung|bringst du.{0,20}mit|"
        r"bringen sie.{0,20}mit|was du mitbringst|was sie mitbringen|du bringst mit|"
        r"sie bringen mit|solltest du mitbringen|deine skills|wünschen wir uns|wir wünschen uns|"
        r"wie sie uns überzeugen|zeichnen sich aus|zeichnest du dich aus|das bist du|du passt|"
        r"passt gut zu uns|eher nichts für dich|dein hintergrund|ihr können|pluspunkte|"
        r"muss-kriterien|kein muss|cherry on top|von vorteil|requirements|qualifications|"
        r"who you are|what you bring|what we.?re looking for|what we are looking for|about you|"
        r"your profile|you have|you bring|must.?have|nice.?to.?have|skills|experience|"
        r"who thrives|isn.?t for you|tools & stack|tech stack|wünschenswert|kenntnisse|"
        r"ideal candidate|was wir erwarten|was wir uns wünschen|dein background)",
        re.IGNORECASE,
    ),
    "tasks": re.compile(
        r"(aufgaben|aufgabengebiet|tätigkeit|deine rolle|ihre rolle|dein job|das erwartet dich|"
        r"was dich erwartet|was sie erwartet|verantwortlichkeiten|deine verantwortung|"
        r"ihre verantwortung|tagesgeschäft|das machst du|was du machst|die aufgabe|"
        r"über die rolle|und danach\?|what you.?ll do|what you will do|what you.?ll be doing|"
        r"what you will be doing|what you.?ll own|responsibilities|the role|your role|"
        r"about the role|tasks|the job|your impact|your challenge|success looks like|"
        r"day to day|in this role|the opportunity)",
        re.IGNORECASE,
    ),
    "offer": re.compile(
        r"(benefits|wir (ihnen |dir )?bieten|bieten (wir|dir|ihnen)|unser angebot|vorteile|"
        r"darauf (können sie sich|kannst du dich) freuen|was du von uns erwarten|darum wir|"
        r"^warum\b|rahmenbedingungen|perspektiven|karrierechancen|look forward to|"
        r"benefit from|career at .* give you|proposition|makes this exciting|what we offer|"
        r"why join|perks|compensation|what.?s in it|we offer|our offer|vergütung|gehalt|salary)",
        re.IGNORECASE,
    ),
    "company": re.compile(
        r"(über uns|ueber uns|^über \w+|wer wir sind|das sind wir|\bunternehmen\b|mission|"
        r"einleitung|^intro|introduction|our background|how we win|^team$|about us|who we are|"
        r"about the company|about the team|our company|^about \w+)",
        re.IGNORECASE,
    ),
    "other": re.compile(
        r"(bewerbung|bewirb dich|bewerben sie sich|bewirbst du dich|application|interview|"
        r"hiring process|kontakt|contact|noch fragen|fragen zur stelle|interessiert\?|"
        r"interesse geweckt|klingt nach|worauf wartest du|zum schluss|einstiegstermin|eckdaten|"
        r"persönlich für sie da|join us$|join the team|privacy|datenschutz|location|standort|"
        r"diversity|equal opportunit|chancengleichheit)",
        re.IGNORECASE,
    ),
}
ORDER = ("requirements", "tasks", "offer", "company", "other")
MAX_PLAIN_HEADING = 60
MAX_MARKUP_HEADING_WORDS = 12


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


KEY_VALUE = re.compile(r"^[^:]{1,30}:\s+\S")  # "Standort: Hildesheim" is data, not a heading


def _heading_of(line: str) -> str | None:
    if line.startswith("## "):
        head = line[3:].strip()
        # a bold sentence is emphasis, not a heading
        return head if len(head.split()) <= MAX_MARKUP_HEADING_WORDS else None
    s = line.strip()
    if 0 < len(s) <= MAX_PLAIN_HEADING and not s.startswith("- ") and not KEY_VALUE.match(s):
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
