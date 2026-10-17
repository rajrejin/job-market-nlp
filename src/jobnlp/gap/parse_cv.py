"""Read a CV (plain text, PDF or DOCX) in memory and split it into sections.

CV content is personal data: it is processed in memory only, never logged and never written to
disk by this module.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

MAX_BYTES = 2_000_000
MAX_CHARS = 30_000
ALLOWED = {"txt", "pdf", "docx"}

SECTION_CUES: dict[str, re.Pattern[str]] = {
    "experience": re.compile(
        r"^(berufserfahrung|beruflicher werdegang|erfahrung|praxiserfahrung|work experience|"
        r"professional experience|experience|employment|career)\b",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"^(ausbildung|bildung|studium|bildungsweg|education|academic background|qualifications)\b",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"^(kenntnisse|fähigkeiten|kompetenzen|skills|technical skills|it-kenntnisse|"
        r"sprachkenntnisse|languages|sprachen|tools|technologies)\b",
        re.IGNORECASE,
    ),
    "projects": re.compile(r"^(projekte|projects|publikationen|publications)\b", re.IGNORECASE),
}


class CVError(ValueError):
    """Raised for unsupported or oversized input; the message never contains CV text."""


@dataclass
class CVSection:
    label: str
    text: str


def read_cv(data: bytes, filename: str) -> str:
    if len(data) > MAX_BYTES:
        raise CVError(f"file larger than {MAX_BYTES // 1_000_000} MB")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    if ext not in ALLOWED:
        raise CVError("only .txt, .pdf and .docx files are accepted")
    if ext == "txt":
        text = data.decode("utf-8", errors="replace")
    elif ext == "pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:20])
        except Exception as exc:  # malformed PDFs raise many exception types
            raise CVError("the PDF could not be read") from exc
    else:
        from docx import Document

        try:
            doc = Document(io.BytesIO(data))
        except Exception as exc:
            raise CVError("the DOCX file could not be read") from exc
        text = "\n".join(p.text for p in doc.paragraphs)
    return clip(text)


def clip(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text.strip()


def split_cv(text: str) -> list[CVSection]:
    sections = [CVSection("other", "")]
    lines: list[list[str]] = [[]]
    for line in text.split("\n"):
        s = line.strip().strip(":").strip()
        label = next(
            (lab for lab, pat in SECTION_CUES.items() if len(s) <= 40 and pat.match(s)), None
        )
        if label:
            sections.append(CVSection(label, ""))
            lines.append([])
        else:
            lines[-1].append(line)
    for sec, ls in zip(sections, lines, strict=True):
        sec.text = "\n".join(ls).strip()
    return [s for s in sections if s.text]


def bullets(text: str, min_chars: int = 25) -> list[str]:
    """Lines that read like CV bullets (used as rewrite candidates)."""
    out = []
    for line in text.split("\n"):
        s = line.strip().lstrip("-•*▪●◦·").strip()
        if len(s) >= min_chars:
            out.append(s)
    return out
