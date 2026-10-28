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
    r"deine ansprechpartnerin|ihre ansprechpartnerin|fragen\?|questions\?|"
    r"persönlich für sie da|ihr personalprofi|personaldisponent|disponent)",
    re.IGNORECASE,
)
# a line naming a role, next to which a bare name is a contact person
ROLE_LINE = re.compile(
    r"^\s*(##\s*)?[\w ()/&.-]{0,40}(disponent(in)?|recruiter(in)?|referent(in)?|manager(in)?|"
    r"officer|leiter(in)?|geschäftsführer(in)?|ansprechpartner(in)?|personalberater(in)?|"
    r"talent acquisition|people|hr business partner|recruiting|inhaber(in)?|founder|ceo|cto|coo)"
    r"[\w ()/&.-]{0,40}$",
    re.IGNORECASE,
)

CLOSING_LINES = 6
PersonFinder = Callable[[str, str], list[tuple[int, int]]]


def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def redact_phones(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        s = m.group(0)
        return "[PHONE]" if 7 <= _digits(s) <= 15 else s

    return PHONE.sub(sub, text)


PARTICLE = r"(?:von|van|de|der|den|du|zu|ten|ter)"
CUE_NAME = re.compile(
    rf"\b(?:ansprechpartner(?:in)?|ansprechperson|recruiter(?:in)?|contact|kontakt|"
    rf"wende dich an|wenden sie sich an|melde dich bei|reach out to|via|an|bei|with|mit)"
    rf"\s*:?\s+({CAPWORD}(?:\s+{PARTICLE})*\s+{CAPWORD})",
    re.IGNORECASE,
)
NAME_TOKEN = re.compile(r"^[A-ZÄÖÜ][a-zäöüßéèáàíóúñç'-]+$")


NOT_NAME = {
    "Du",
    "Sie",
    "Wir",
    "Ihr",
    "Dein",
    "Deine",
    "Ihre",
    "Unser",
    "Unsere",
    "Wenn",
    "Als",
    "Bei",
    "Mit",
    "Der",
    "Die",
    "Das",
    "Director",
    "Manager",
    "Lead",
    "Head",
    "Sales",
    "Account",
    "Group",
    "Team",
    "Senior",
    "Junior",
    "Engineer",
    "Consultant",
    "Office",
    "Marketing",
    "About",
    "Ideal",
    "Fluent",
    "Why",
    "Join",
    "Our",
    "Your",
    "Product",
}


def plausible_name(span: str) -> bool:
    """Two or more title-case tokens (particles allowed), none of them a pronoun or role word:
    rejects sentence-initial words and job titles that NER models often tag as persons."""
    toks = [t for t in span.split() if not re.fullmatch(PARTICLE, t)]
    return (
        len(toks) >= 2
        and all(NAME_TOKEN.match(t) for t in toks)
        and not any(t in NOT_NAME for t in toks)
    )


ROLE_NAME = re.compile(
    rf"\b(?:Manager(?:in)?|Leiter(?:in)?|Head of \w+|CEO|CTO|COO|CFO|CPO|Founder|Co-Founder|"
    rf"Gründer(?:in)?|Mitgründer(?:in)?|Geschäftsführer(?:in)?|Recruiter(?:in)?|Teamlead|"
    rf"Team Lead|Inhaber(?:in)?|Partner(?:in)?|Director|Direktor(?:in)?)\s*,?\s+"
    rf"({CAPWORD}(?:\s+{PARTICLE})*\s+{CAPWORD})"
)


def _replace(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in _merge(spans):
        text = text[:start] + "[NAME]" + text[end:]
    return text


def redact(text: str, person_finder: PersonFinder | None = None, lang: str = "de") -> str:
    """Remove emails, URLs, phone numbers and person names.

    Names: salutation rule and role-title rule everywhere; cue rule and, when a person finder
    is given, multi-word person entities inside contact blocks. Whole-text NER was tried and
    rejected: it removed product names such as "Claude Code" or "Instagram Reels"
    (docs/DECISIONS.md)."""
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    text = redact_phones(text)
    text = SALUTATION.sub("[NAME]", text)
    text = _replace(
        text,
        [(m.start(1), m.end(1)) for m in ROLE_NAME.finditer(text) if plausible_name(m.group(1))],
    )
    lines = text.split("\n")
    # contact blocks: two lines before and after every contact cue (the name often comes first,
    # as in "Anna Muster / Personaldisponentin / Persönlich für Sie da")
    window: set[int] = set()
    after_cue: set[int] = set()  # the preposition rule ("mit Anna Muster") only after a cue
    for i, line in enumerate(lines):
        if CONTACT_CUE.search(line):
            window.update(range(max(0, i - 2), min(len(lines), i + 3)))
            after_cue.update(range(i, min(len(lines), i + 3)))
    for i in sorted(window):
        line = lines[i]
        spans = [
            (m.start(1), m.end(1))
            for m in CUE_NAME.finditer(line)
            if i in after_cue and plausible_name(m.group(1))
        ]
        if person_finder is not None:
            both = set(person_finder(line, "de")) & set(person_finder(line, "en"))
            spans += [(s, e) for s, e in both if plausible_name(line[s:e])]
        lines[i] = _replace(line, spans)
    # a line that is only a name, next to a role line or confirmed as a person by NER
    for i, line in enumerate(lines):
        bare = line.strip().removeprefix("## ").strip()
        if not plausible_name(bare) or len(bare.split()) > 3:
            continue
        near_role = any(0 <= j < len(lines) and ROLE_LINE.search(lines[j]) for j in (i - 1, i + 1))
        # NER alone is weak on two-word lines ("Fluent German", "About Airwallex"): both the
        # German and the English model must tag the whole line as a person
        closing = i >= len(lines) - CLOSING_LINES  # contact details sit at the end
        models = (lang,) if closing else ("de", "en")
        is_person = person_finder is not None and all(
            any(e - s == len(bare) for s, e in person_finder(bare, model_lang))
            for model_lang in models
        )
        if near_role or is_person:
            lines[i] = line.replace(bare, "[NAME]")
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
    """Person entities from the spaCy model for the text's language (German model otherwise)."""
    import spacy

    off = ["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler"]
    models = {
        "de": spacy.load("de_core_news_lg", disable=off),
        "en": spacy.load("en_core_web_md", disable=off),
    }

    def find(text: str, lang: str) -> list[tuple[int, int]]:
        nlp = models.get(lang, models["de"])
        return [(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ in {"PER", "PERSON"}]

    return find


def has_contact_data(text: str) -> dict[str, int]:
    """Counts of residual personal data patterns (used by the redaction tests)."""
    return {
        "email": len(EMAIL.findall(text)),
        "url": len(URL.findall(text)),
        "phone": sum(7 <= _digits(m.group(0)) <= 15 for m in PHONE.finditer(text)),
        "salutation_name": len(SALUTATION.findall(text)),
    }
