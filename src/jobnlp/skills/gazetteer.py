"""Dictionary matching of skill phrases (ESCO labels and mined terms) on surface and lemma forms.

Matching runs on spaCy tokens. Each phrase is indexed twice: by lower-cased surface tokens and
by lower-cased lemmas, so German inflection ("Datenbanken" for the label "Datenbank") matches.
Short labels (three characters or fewer, such as SQL or R) only match case-sensitively, and
one-letter labels only next to another tool match, to avoid matching ordinary words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from spacy.language import Language
from spacy.tokens import Doc

MAX_TOKENS = 5
CONTEXT_CHARS = 40
NON_NOUN = {"VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "NUM"}


@dataclass(frozen=True)
class Phrase:
    skill_id: str
    label: str
    source: str  # "esco" or "mined"
    kind: str  # esco skill type or "tool" for mined terms


@dataclass
class Match:
    start: int  # character offsets
    end: int
    text: str
    skill_id: str
    label: str
    source: str
    kind: str


@dataclass
class Gazetteer:
    surface: dict[str, Phrase] = field(default_factory=dict)
    lemma: dict[str, Phrase] = field(default_factory=dict)
    exact_short: dict[str, Phrase] = field(default_factory=dict)
    blocked: set[str] = field(default_factory=set)

    def add(
        self, nlp: Language, phrases: Iterable[tuple[str, Phrase]], override: bool = False
    ) -> None:
        """Index phrases; with override=True they replace existing entries with the same key."""
        items = [(t.strip(), p) for t, p in phrases if t and t.strip()]
        docs = nlp.pipe([t for t, _ in items], batch_size=512)
        for (text, phrase), doc in zip(items, docs, strict=True):
            if len(doc) == 0 or len(doc) > MAX_TOKENS:
                continue
            if (
                phrase.source == "esco"
                and phrase.kind != "language"
                and (all(t.is_stop for t in doc) or (len(doc) == 1 and doc[0].pos_ in NON_NOUN))
            ):
                continue  # single verbs, adjectives etc. ("reported", "deutschen") over-match
            keys: list[tuple[dict[str, Phrase], str]]
            if len(text) <= 3:
                keys = [(self.exact_short, text)]
            elif phrase.source == "mined":
                # names of tools and products: no lemma key ("control" must not match
                # "Controlling")
                keys = [(self.surface, _key(t.lower_ for t in doc))]
            else:
                keys = [
                    (self.surface, _key(t.lower_ for t in doc)),
                    (self.lemma, _key(t.lemma_.lower() for t in doc)),
                ]
            for table, key in keys:
                if override:
                    table[key] = phrase
                else:
                    table.setdefault(key, phrase)

    def _compound_stem(self, token: str) -> Phrase | None:
        """'Deutschkenntnisse', 'SQL-Kenntnisse', 'Python-Erfahrung' -> the stem's phrase."""
        m = COMPOUND_SUFFIX.match(token)
        if not m or len(m.group("stem")) < 1:
            return None
        stem = m.group("stem").rstrip("-")
        return self.exact_short.get(stem) or self.surface.get(stem.lower())

    def copy(self) -> Gazetteer:
        return Gazetteer(
            dict(self.surface), dict(self.lemma), dict(self.exact_short), set(self.blocked)
        )

    def surface_key(self, nlp: Language, text: str) -> str:
        return _key(t.lower_ for t in nlp(text))

    def match(self, doc: Doc) -> list[Match]:
        out: list[Match] = []
        i, n = 0, len(doc)
        while i < n:
            found: tuple[int, Phrase] | None = None
            for size in range(min(MAX_TOKENS, n - i), 0, -1):
                span = doc[i : i + size]
                if size == 1 and span.text in self.exact_short:
                    found = (size, self.exact_short[span.text])
                    break
                k_surface = _key(t.lower_ for t in span)
                if k_surface in self.blocked:
                    continue
                phrase = self.surface.get(k_surface) or self.lemma.get(
                    _key(t.lemma_.lower() for t in span)
                )
                if phrase is not None:
                    found = (size, phrase)
                    break
            if found is None:
                stem_phrase = self._compound_stem(doc[i].text)
                if stem_phrase is not None:
                    found = (1, stem_phrase)
            if found is None:
                i += 1
                continue
            size, phrase = found
            span = doc[i : i + size]
            out.append(
                Match(
                    span.start_char,
                    span.end_char,
                    span.text,
                    phrase.skill_id,
                    phrase.label,
                    phrase.source,
                    phrase.kind,
                )
            )
            i += size
        return _context_filter(out)


COMPOUND_SUFFIX = re.compile(r"^(?P<stem>.+?)-?(kenntnisse|kenntnissen|erfahrung|skills)$", re.I)


def _key(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def _context_filter(matches: list[Match]) -> list[Match]:
    """Keep one-letter matches (R, C) only near another short tool match such as SQL or Python."""
    keep: list[Match] = []
    for m in matches:
        if len(m.text) == 1:
            near = [
                o
                for o in matches
                if o is not m and len(o.text) > 1 and abs(o.start - m.start) <= CONTEXT_CHARS
            ]
            if not near:
                continue
        keep.append(m)
    return keep
