"""Skill demand aggregates with Wilson intervals and small-group suppression (DuckDB)."""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from jobnlp.market.uncertainty import wilson_interval

SENIOR = re.compile(r"\b(senior|sr\.|lead|principal|head of|staff|leitung|leiter\w*)\b", re.I)
ENTRY = re.compile(
    r"\b(junior|jr\.|einsteiger\w*|berufseinsteiger\w*|graduate|werkstudent\w*|praktik\w*|"
    r"intern(ship)?|trainee|azubi|ausbildung zum|working student|entry[- ]level)\b",
    re.I,
)
EXPERIENCED = re.compile(
    r"(mehrjährig\w*|langjährig\w*|several years|\b[3-9]\+?\s*(jahre|years)|"
    r"\bmindestens\s+[3-9]\s+jahre|experienced\b|berufserfahren)",
    re.I,
)
DATA_ROLES = {
    "data_scientist": re.compile(r"data scien|datenwissenschaft", re.I),
    "data_analyst": re.compile(
        r"data analy|datenanaly|business intelligence|bi analyst|"
        r"analytics (engineer|manager)|product analyst",
        re.I,
    ),
    "data_engineer": re.compile(
        r"data engineer|daten.?engineer|analytics engineer|"
        r"data platform|dataops",
        re.I,
    ),
    "ml_ai_engineer": re.compile(
        r"machine learning|ml engineer|\bml\b|\bai engineer|\bki[- ]|"
        r"artificial intelligence|\bllm|deep learning|mlops|applied ai|"
        r"applied scientist|computer vision|\bnlp\b",
        re.I,
    ),
}


def experience_level(title: str, text: str) -> str:
    if ENTRY.search(title):
        return "entry"
    if SENIOR.search(title):
        return "senior"
    if EXPERIENCED.search(text):
        return "experienced"
    if ENTRY.search(text[:1500]):
        return "entry"
    return "unspecified"


def data_role(title: str) -> str | None:
    for role, pat in DATA_ROLES.items():
        if pat.search(title):
            return role
    return None


def shares(
    postings: pd.DataFrame,
    skills: pd.DataFrame,
    group_col: str | None,
    min_group: int = 30,
    min_skill_count: int = 5,
    top: int = 30,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Share of postings mentioning each skill, per group, with 95% Wilson intervals.
    Groups with fewer than `min_group` postings are suppressed and listed separately."""
    con = duckdb.connect()
    con.register("p", postings)
    con.register("s", skills)
    g = f"p.{group_col}" if group_col else "'all'"
    sizes = con.execute(f"SELECT {g} AS grp, COUNT(*) AS n FROM p GROUP BY grp").df()
    counts = con.execute(
        f"""SELECT {g} AS grp, s.skill_id, ANY_VALUE(s.label) AS label,
                   COUNT(DISTINCT s.posting_key) AS k
            FROM s JOIN p ON s.posting_key = p.posting_key
            GROUP BY grp, s.skill_id"""
    ).df()
    con.close()
    suppressed = [
        {"group": str(grp), "n_postings": int(n)}
        for grp, n in zip(sizes.grp, sizes.n, strict=True)
        if int(n) < min_group
    ]
    ok = sizes[sizes.n >= min_group]
    df = counts.merge(ok, on="grp")
    df = df[df.k >= min_skill_count].copy()
    df["share"] = df.k / df.n
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(df.k, df.n, strict=True)]
    df["ci_low"] = [round(a, 4) for a, _ in ci]
    df["ci_high"] = [round(b, 4) for _, b in ci]
    df["share"] = df.share.round(4)
    df = df.sort_values(["grp", "share"], ascending=[True, False]).groupby("grp").head(top)
    out = df.rename(columns={"grp": "group", "k": "postings_with_skill", "n": "postings_in_group"})
    return out.reset_index(drop=True), suppressed


def employer_concentration(postings: pd.DataFrame) -> dict[str, float | int]:
    counts = postings.employer.fillna("unknown").value_counts()
    s = counts / counts.sum()
    return {
        "postings": int(counts.sum()),
        "employers": int(counts.size),
        "top1_share": round(float(s.iloc[0]), 4),
        "top10_share": round(float(s.head(10).sum()), 4),
        "hhi": round(float((s**2).sum()), 5),
        "employers_with_one_posting": int((counts == 1).sum()),
    }
