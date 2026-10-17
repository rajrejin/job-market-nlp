"""Skill extraction: gazetteer (ESCO and mined terms), optional learned NER, and necessity."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd
import spacy
from spacy.language import Language

from jobnlp.skills.gazetteer import Gazetteer, Match, Phrase
from jobnlp.skills.necessity import necessity_at

PAREN_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")
OFF = ["parser", "ner"]


@lru_cache(maxsize=2)
def load_nlp(lang: str) -> Language:
    name = "en_core_web_md" if lang == "en" else "de_core_news_lg"
    return spacy.load(name, disable=OFF)


def esco_phrases(esco: pd.DataFrame) -> dict[str, list[tuple[str, Phrase]]]:
    """Labels per language. A trailing qualifier is dropped: 'Python (Computerprogrammierung)'
    is indexed as 'Python' as well."""
    out: dict[str, list[tuple[str, Phrase]]] = {"de": [], "en": []}
    for rec in esco.to_dict("records"):
        pref = rec["pref_de"] or rec["pref_en"] or ""
        phrase = Phrase(str(rec["uri"]), str(pref), "esco", str(rec["skill_type"]))
        for lang, labels in (
            ("de", [rec["pref_de"], *list(rec["alt_de"])]),
            ("en", [rec["pref_en"], *list(rec["alt_en"])]),
        ):
            for label in labels:
                if not label:
                    continue
                out[lang].append((str(label), phrase))
                short = PAREN_QUALIFIER.sub("", str(label))
                if short and short != label:
                    out[lang].append((short, phrase))
    return out


def mined_phrases(terms: Iterable[str]) -> list[tuple[str, Phrase]]:
    return [(t, Phrase(f"mined:{t.lower()}", t, "mined", "tool")) for t in terms]


def build_gazetteer(
    esco: pd.DataFrame, mined: Iterable[str] = (), blocked: Iterable[str] = ()
) -> Gazetteer:
    """Mined terms are added first so that they win over ESCO labels with the same key."""
    gaz = Gazetteer(blocked={b.lower() for b in blocked})
    mined_list = list(mined)
    phrases = esco_phrases(esco)
    for lang in ("de", "en"):
        nlp = load_nlp(lang)
        gaz.add(nlp, mined_phrases(mined_list))
        gaz.add(nlp, phrases[lang])
    return gaz


@dataclass
class SkillMention:
    start: int
    end: int
    text: str
    skill_id: str
    label: str
    source: str
    kind: str
    necessity: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SkillExtractor:
    def __init__(self, gazetteer: Gazetteer, ner: Language | None = None) -> None:
        self.gazetteer = gazetteer
        self.ner = ner

    def gazetteer_matches(self, text: str, lang: str) -> list[Match]:
        return self.gazetteer.match(load_nlp(lang)(text))

    def ner_matches(self, text: str) -> list[Match]:
        if self.ner is None:
            return []
        doc = self.ner(text)
        return [
            Match(e.start_char, e.end_char, e.text, f"ner:{e.text.lower()}", e.text, "ner", "skill")
            for e in doc.ents
        ]

    def extract(self, text: str, lang: str, use_ner: bool = True) -> list[SkillMention]:
        matches = self.gazetteer_matches(text, lang)
        if use_ner:
            matches = merge_matches(matches, self.ner_matches(text))
        return [
            SkillMention(
                m.start,
                m.end,
                m.text,
                m.skill_id,
                m.label,
                m.source,
                m.kind,
                necessity_at(text, m.start),
            )
            for m in matches
        ]


def merge_matches(primary: list[Match], extra: list[Match]) -> list[Match]:
    """Union of two match lists; an extra match overlapping a primary one is dropped."""
    out = list(primary)
    for m in extra:
        if not any(m.start < p.end and p.start < m.end for p in primary):
            out.append(m)
    return sorted(out, key=lambda m: m.start)
