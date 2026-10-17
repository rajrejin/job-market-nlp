"""Mining skill terms that ESCO lacks (tools, products, frameworks) from requirement sections.

Candidates are items that follow a cue ("Kenntnisse in", "Erfahrung mit", "experience with",
"z. B.", "such as") or sit in a parenthesised list, split on list separators. A candidate is
kept as a `mined` skill when it recurs after cues in enough postings from enough employers and
appears after a cue in a large enough share of the postings that mention it at all.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

CUE = re.compile(
    r"(?:kenntnisse?\s+(?:in|im|der|von|mit|über)|erfahrung(?:en)?\s+(?:mit|in|im|bei)|"
    r"umgang\s+mit|vertraut\s+mit|beherrsch\w*|experience\s+(?:with|in|using)|knowledge\s+of|"
    r"proficien\w*\s+(?:in|with)|familiar\w*\s+with|expertise\s+(?:in|with)|skilled\s+in|"
    r"hands-on\s+(?:with|experience\s+with)|z\.\s?b\.|e\.\s?g\.|such\s+as|wie\s+z\.\s?b\.|"
    r"insbesondere|including|tools?:|stack:)\s*(?P<tail>[^.;:\n]{2,160})",
    re.IGNORECASE,
)
PAREN = re.compile(r"\(([^()]{2,160})\)")
SPLIT = re.compile(
    r",|/|\(|\)|\s\+\s|\s&\s|\bund\b|\boder\b|\band\b|\bor\b|\bsowie\b|\bbzw\.?|\betc\.?|"
    r"\bo\.\s?ä\.?|\bu\.\s?a\.?|\bähnlich\w*|\bsimilar\b|\bvergleichbar\w*|\bcomparable\b",
    re.IGNORECASE,
)
TOKEN_OK = re.compile(r"^[A-Za-zÄÖÜäöüß0-9][\w.+#/-]*$")
STOP = {
    "erfahrung",
    "kenntnisse",
    "kenntnissen",
    "bereich",
    "umgang",
    "tools",
    "tool",
    "systeme",
    "systemen",
    "software",
    "lösungen",
    "methoden",
    "technologien",
    "plattformen",
    "themen",
    "projekte",
    "prozesse",
    "experience",
    "knowledge",
    "skills",
    "systems",
    "platforms",
    "frameworks",
    "technologies",
    "deutsch",
    "englisch",
    "german",
    "english",
    "it",
    "the",
    "a",
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "mit",
    "in",
    "of",
    "with",
    "ideally",
    "idealerweise",
    "gerne",
    "vorteil",
    "plus",
    "bonus",
    "ms",
    "office",
    "erste",
    "gute",
    "sehr",
    "fundierte",
    "solide",
    "strong",
    "good",
    "basic",
    "advanced",
    "relevant",
    "relevante",
    "moderne",
    "modernen",
    "gängigen",
    "gängige",
    "common",
    "other",
    "andere",
    "weitere",
    "etc",
    "bzw",
    # found as noise in the first mining run (results reviewed 2026-10-17)
    "wort",
    "schrift",
    "zeugnisse",
    "muss",
    "menschen",
    "planung",
    "anwendungen",
    "senior",
    "junior",
    "agentur",
    "niveau",
    "level",
    "jahre",
    "years",
    "team",
    "teams",
    "kunden",
    "customers",
}
BULLET_MAX_WORDS = 8
LANGUAGE_LEVEL = re.compile(r"^[ABC][12]\+?$")
# slash compounds that name one thing (CI/CD, S/4HANA, TCP/IP, UI/UX, A/B) are protected from
# splitting: the slash is swapped for a placeholder before splitting and restored after
SLASH_COMPOUND = re.compile(r"\b([A-Za-z0-9]{1,4})/([A-Za-z0-9]{1,6})\b")
SLASH_MARK = "⁄"


@dataclass
class MinedTerm:
    term: str
    df_cue: int
    df_all: int
    employers: int

    @property
    def cue_ratio(self) -> float:
        return self.df_cue / self.df_all if self.df_all else 0.0


def clean_item(item: str) -> str | None:
    """Trim an item to a short name: drop leading lower-case words (adjectives, articles)
    and anything after the third token; require a capital letter or digit somewhere."""
    raw = item.strip()
    if raw.endswith("-"):
        return None  # fragment of a German ellipsis compound ("Wirtschafts- und ...")
    toks = raw.strip("–•*\"'“”„").split()
    while toks and (toks[0].lower() in STOP or (toks[0][:1].islower() and len(toks) > 1)):
        toks = toks[1:]
    toks = [t.strip(".,;:!?\"'“”„").replace(SLASH_MARK, "/") for t in toks[:3]]
    toks = [t for t in toks if t]
    if not toks or not all(TOKEN_OK.match(t) for t in toks):
        return None
    term = " ".join(toks)
    if term.lower() in STOP or len(term) < 2 or len(term) > 40 or LANGUAGE_LEVEL.match(term):
        return None
    if not re.search(r"[A-ZÄÖÜ0-9]", term):
        return None
    if term.isdigit():
        return None
    return term


def candidates(text: str) -> set[str]:
    found: set[str] = set()
    tails = [m.group("tail") for m in CUE.finditer(text)]
    tails += [m.group(1) for m in PAREN.finditer(text)]
    # short list lines ("- Teamfähigkeit, Zuverlässigkeit und Eigeninitiative") name skills
    # directly, without a cue phrase
    tails += [
        line.strip()[2:]
        for line in text.split("\n")
        if line.strip().startswith("- ") and len(line.split()) <= BULLET_MAX_WORDS
    ]
    for tail in tails:
        tail = SLASH_COMPOUND.sub(rf"\1{SLASH_MARK}\2", tail)
        for item in SPLIT.split(tail):
            term = clean_item(item)
            if term:
                found.add(term)
    return found


def mine(docs: Iterable[tuple[str, str]]) -> list[MinedTerm]:
    """docs: (employer, requirement text). Returns every candidate with its statistics."""
    docs = list(docs)
    cue_docs: Counter[str] = Counter()
    employers: dict[str, set[str]] = defaultdict(set)
    for employer, text in docs:
        for term in candidates(text):
            cue_docs[term] += 1
            employers[term].add(employer)
    terms = list(cue_docs)
    patterns = {t: re.compile(rf"(?<![\w]){re.escape(t)}(?![\w])") for t in terms}
    all_docs: Counter[str] = Counter()
    for _, text in docs:
        for t in terms:
            if t in text and patterns[t].search(text):
                all_docs[t] += 1
    return [
        MinedTerm(t, cue_docs[t], max(all_docs[t], cue_docs[t]), len(employers[t])) for t in terms
    ]


def select(
    terms: list[MinedTerm],
    known: set[str],
    min_df: int = 3,
    min_employers: int = 2,
    min_cue_ratio: float = 0.2,
) -> list[MinedTerm]:
    """Terms that pass the thresholds and are not already known (lower-cased) labels."""
    return sorted(
        (
            t
            for t in terms
            if t.df_cue >= min_df
            and t.employers >= min_employers
            and t.cue_ratio >= min_cue_ratio
            and t.term.lower() not in known
        ),
        key=lambda t: (-t.df_cue, t.term),
    )
