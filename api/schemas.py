"""Request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MAX_CHARS = 30_000


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    lang: str | None = Field(default=None, pattern="^(de|en)$")


class SkillOut(BaseModel):
    text: str
    start: int
    end: int
    skill_id: str
    label: str
    source: str
    necessity: str


class ExtractOut(BaseModel):
    lang: str
    skills: list[SkillOut]


class ClassifyOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    model: str


class GapIn(BaseModel):
    job_text: str = Field(min_length=1, max_length=MAX_CHARS)
    cv_text: str = Field(min_length=1, max_length=MAX_CHARS)
    suggestions: bool = True


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    error: ErrorBody


class MarketOut(BaseModel):
    group_by: str
    rows: list[dict[str, Any]]
    suppressed_groups: list[dict[str, Any]]
    min_group_size: int
    note: str
