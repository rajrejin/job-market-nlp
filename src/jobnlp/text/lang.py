"""Language detection with lingua; langdetect is used only to measure agreement."""

from __future__ import annotations

from functools import lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

LANGS = [
    Language.GERMAN,
    Language.ENGLISH,
    Language.DUTCH,
    Language.FRENCH,
    Language.SPANISH,
    Language.ITALIAN,
    Language.POLISH,
    Language.PORTUGUESE,
]
CODES = {Language.GERMAN: "de", Language.ENGLISH: "en"}


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*LANGS).build()


def detect(text: str, max_chars: int = 3000) -> str:
    """'de', 'en' or 'other'."""
    if not text.strip():
        return "other"
    lang = _detector().detect_language_of(text[:max_chars])
    return CODES.get(lang, "other") if lang else "other"


def detect_langdetect(text: str, max_chars: int = 3000) -> str:
    from langdetect import DetectorFactory
    from langdetect import detect as ld_detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        code = ld_detect(text[:max_chars])
    except LangDetectException:
        return "other"
    return code if code in {"de", "en"} else "other"
