"""Remove personal data from posting text: emails, phone numbers, URLs, contact person names.

Names are removed in two ways: a rule for salutations ("Frau Dr. Anna Muster", "Mr John Doe")
anywhere in the text, and person entities from spaCy NER inside contact blocks (lines that
contain or follow a contact cue such as "Ansprechpartner" or "Contact").
"""

from __future__ import annotations

import re
from collections.abc import Callable

EMAIL = re.compile(r"[\w.+-]+\s?(?:@|\(at\)|\[at\])\s?[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+", re.IGNORECASE)
# phone: optional +49 / 0049 / leading 0, then digit groups with separators, 7+ digits in total
PHONE = re.compile(
    r"(?<![\w.,])(?:(?:\+|00)\d{2}[\s./-]?(?:\(0\)[\s./-]?)?|0)\d{2,5}(?:[\s./-]?\d{2,}){1,4}(?![\w])"
)
CAPWORD = r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?"
SALUTATION = re.compile(
    rf"\b(?:Frau|Herr|Herrn|Mrs?\.?|Ms\.?)\s+(?:(?:Dr|Prof)\.?\s+)*{CAPWORD}(?:\s+{CAPWORD}){{0,2}}"
)
CONTACT_CUE = re.compile(
    r"(ansprechpartner|ansprechperson|kontakt|contact|recruiter|your recruiter|"
    r"deine ansprechpartnerin|ihre ansprechpartnerin|fragen\?|questions\?)",
    re.IGNORECASE,
)

PersonFinder = Callable[[str], list[tuple[int, int]]]


def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def redact_phones(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        s = m.group(0)
        return "[PHONE]" if 7 <= _digits(s) <= 15 else s

    return PHONE.sub(sub, text)


PARTICLE = r"(?:von|van|de|der|den|du|zu|ten|ter)"
CUE_NAME = re.compile(
    rf"(?:ansprechpartner(?:in)?|ansprechperson|recruiter(?:in)?|contact|kontakt|"
    rf"wende dich an|wenden sie sich an|melde dich bei|reach out to|via|an|bei|with|mit)"
    rf"\s*:?\s+({CAPWORD}(?:\s+{PARTICLE})*\s+{CAPWORD})",
    re.IGNORECASE,
)
NAME_TOKEN = re.compile(r"^[A-ZÄÖÜ][a-zäöüßéèáàíóúñç'-]+$")


def plausible_name(span: str) -> bool:
    """Two or more title-case tokens (particles allowed): rejects single sentence-initial words
    that NER models often tag as persons."""
    toks = [t for t in span.split() if not re.fullmatch(PARTICLE, t)]
    return len(toks) >= 2 and all(NAME_TOKEN.match(t) for t in toks)


def redact(text: str, person_finder: PersonFinder | None = None) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    text = redact_phones(text)
    text = SALUTATION.sub("[NAME]", text)
    lines = text.split("\n")
    in_contact = 0
    for i, line in enumerate(lines):
        if CONTACT_CUE.search(line):
            in_contact = 3  # the cue line and the next two lines
        if in_contact > 0:
            spans = [
                (m.start(1), m.end(1))
                for m in CUE_NAME.finditer(line)
                if plausible_name(m.group(1))
            ]
            if person_finder is not None:
                spans += [(s, e) for s, e in person_finder(line) if plausible_name(line[s:e])]
            for start, end in _merge(spans):
                line = line[:start] + "[NAME]" + line[end:]
            lines[i] = line
            in_contact -= 1
    return "\n".join(lines)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping spans; returned right to left for in-place replacement."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged[::-1]


def spacy_person_finder() -> PersonFinder:
    """Person entities from the German and the English spaCy model (union)."""
    import spacy

    off = ["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler"]
    models = [spacy.load("de_core_news_lg", disable=off), spacy.load("en_core_web_md", disable=off)]

    def find(line: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for nlp in models:
            out += [
                (e.start_char, e.end_char) for e in nlp(line).ents if e.label_ in {"PER", "PERSON"}
            ]
        return out

    return find


def has_contact_data(text: str) -> dict[str, int]:
    """Counts of residual personal data patterns (used by the redaction tests)."""
    return {
        "email": len(EMAIL.findall(text)),
        "url": len(URL.findall(text)),
        "phone": sum(7 <= _digits(m.group(0)) <= 15 for m in PHONE.finditer(text)),
        "salutation_name": len(SALUTATION.findall(text)),
    }
