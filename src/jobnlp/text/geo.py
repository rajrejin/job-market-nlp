"""Map free-text locations to German places with GeoNames (CC-BY 4.0, geonames.org)."""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

STATES = {
    "01": "Baden-Württemberg",
    "02": "Bayern",
    "03": "Bremen",
    "04": "Hamburg",
    "05": "Hessen",
    "06": "Niedersachsen",
    "07": "Nordrhein-Westfalen",
    "08": "Rheinland-Pfalz",
    "09": "Saarland",
    "10": "Schleswig-Holstein",
    "11": "Brandenburg",
    "12": "Mecklenburg-Vorpommern",
    "13": "Sachsen",
    "14": "Sachsen-Anhalt",
    "15": "Thüringen",
    "16": "Berlin",
}
COUNTRY_WORDS = {"germany", "deutschland", "de"}
FOREIGN = re.compile(
    r"\b(austria|österreich|switzerland|schweiz|suisse|netherlands|nederland|france|uk|"
    r"united kingdom|spain|españa|italy|italia|poland|polska|portugal|belgium|ireland|usa|"
    r"united states|london|paris|amsterdam|vienna|wien|zurich|zürich|madrid|barcelona|lisbon|"
    r"warsaw|dublin|prague|brussels|milan|stockholm|copenhagen)\b",
    re.IGNORECASE,
)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


@dataclass(frozen=True)
class Place:
    name: str
    state: str
    population: int


class GermanPlaces:
    def __init__(self, geonames_zip: Path, min_population: int = 5000) -> None:
        self.index: dict[str, Place] = {}
        with zipfile.ZipFile(geonames_zip) as z:
            raw = io.TextIOWrapper(z.open("DE.txt"), encoding="utf-8")
            rows = []
            for line in raw:
                f = line.rstrip("\n").split("\t")
                if f[6] != "P" or int(f[14] or 0) < min_population:
                    continue
                rows.append(f)
        # larger places win name collisions
        rows.sort(key=lambda f: int(f[14] or 0))
        for f in rows:
            place = Place(f[1], STATES.get(f[10], ""), int(f[14]))
            names = {f[1], f[2]} | {
                a for a in f[3].split(",") if len(a) >= 4 and a.isascii() and a[0].isupper()
            }
            for n in names:
                self.index[fold(n)] = place

    def match(self, location: str) -> Place | None:
        """Best German place mentioned in a location string, or None.

        Strings that name a foreign country or city are rejected even if they also contain a
        German place, to avoid multi-country postings being counted as German."""
        if not location or FOREIGN.search(location):
            return None
        best: Place | None = None
        for part in re.split(r"[,;/|()\-]| und | and ", location):
            key = fold(part)
            words = key.split()
            # try the full part, then shorter word windows (e.g. "Frankfurt am Main Office")
            for n in range(len(words), 0, -1):
                for i in range(len(words) - n + 1):
                    cand = self.index.get(" ".join(words[i : i + n]))
                    if cand and (best is None or cand.population > best.population):
                        best = cand
                if best:
                    break
        return best

    def is_germany(self, location: str) -> bool:
        if not location or FOREIGN.search(location):
            return False
        if self.match(location):
            return True
        return any(w in COUNTRY_WORDS for w in fold(location).split()) and "de" != fold(location)
