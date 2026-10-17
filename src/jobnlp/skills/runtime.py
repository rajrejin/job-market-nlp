"""Load the tuned extractor (ESCO labels, mined terms, blocked keys), cached in stores/."""

from __future__ import annotations

import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd

from jobnlp.config import ROOT
from jobnlp.skills.extract import SkillExtractor, build_gazetteer
from jobnlp.skills.gazetteer import Gazetteer

ESCO = ROOT / "data" / "interim" / "esco_skills.parquet"
CONFIG = ROOT / "stores" / "skills_config.json"
CACHE = ROOT / "stores" / "gazetteer.pkl"
NER_DIR = ROOT / "models" / "ner_weak"


def _fingerprint() -> str:
    h = hashlib.sha256()
    for p in (ESCO, CONFIG):
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()


def load_gazetteer(esco_path: Path = ESCO, config_path: Path = CONFIG) -> Gazetteer:
    fp = _fingerprint()
    if CACHE.exists():
        with CACHE.open("rb") as f:
            cached_fp, gaz = pickle.load(f)
        if cached_fp == fp:
            return gaz  # type: ignore[no-any-return]
    cfg: dict[str, list[str]] = {"mined_terms": [], "blocked_keys": []}
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gaz = build_gazetteer(pd.read_parquet(esco_path), blocked=cfg["blocked_keys"])
    from jobnlp.skills.extract import load_nlp, mined_phrases

    for lang in ("de", "en"):
        gaz.add(load_nlp(lang), mined_phrases(cfg["mined_terms"]), override=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump((fp, gaz), f)
    return gaz


@lru_cache(maxsize=1)
def get_extractor(use_ner: bool = False) -> SkillExtractor:
    ner = None
    if use_ner and NER_DIR.exists():
        import spacy

        ner = spacy.load(NER_DIR)
    return SkillExtractor(load_gazetteer(), ner)
