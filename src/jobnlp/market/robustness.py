"""Trend tests and robustness checks for skill shares.

Trends: the publication window is split at its median date into an early and a late half; a
skill counts as emerging or declining only when it has enough mentions and the bootstrap
interval of its share difference excludes zero, also after a Bonferroni correction for the
number of skills tested. With one harvest the window is short and survivorship-biased, so any
flagged skill is a hypothesis, not a finding.
Robustness: the overall top-skill shares are recomputed under alternative rules and compared
with the baseline by rank correlation and largest absolute difference.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# staffing agencies and recruiters, by words in the employer name (heuristic, see DECISIONS.md)
STAFFING = re.compile(
    r"(personal|zeitarbeit|prostaff|staffing|recruit|executives|im auftrag|vermittlung|"
    r"zag menschen|amadeus fire|humancapital|talente|promedis|randstad|adecco|manpower|hays\b|"
    r"orizon|tempton|brunel|ferchau|experis)",
    re.IGNORECASE,
)


def is_staffing(employer: str | None) -> bool:
    return bool(employer) and bool(STAFFING.search(str(employer)))


def top_shares(postings: pd.DataFrame, skills: pd.DataFrame, top: int = 30) -> pd.Series:
    """Share of postings per skill label (top `top` by share)."""
    keys = set(postings.posting_key)
    s = skills[skills.posting_key.isin(keys)]
    shares = s.groupby("label").posting_key.nunique() / max(1, len(keys))
    return shares.sort_values(ascending=False).head(top)


def compare(base: pd.Series, other: pd.Series) -> dict[str, Any]:
    labels = list(base.index)
    other_full = other.reindex(labels).fillna(0.0)
    rho = spearmanr(base.values, other_full.values).statistic if len(labels) > 2 else float("nan")
    diff = (other_full - base).abs()
    return {
        "spearman_rank_correlation_top": round(float(rho), 4),
        "max_abs_share_difference": round(float(diff.max()), 4),
        "skill_with_max_difference": str(diff.idxmax()) if len(diff) else None,
        "top10_overlap": len(
            set(base.index[:10]) & set(other.sort_values(ascending=False).index[:10])
        ),
    }


def robustness(all_postings: pd.DataFrame, skills: pd.DataFrame, top: int = 30) -> dict[str, Any]:
    """all_postings: German postings with usable text, duplicates included (is_dup flag)."""
    dedup = all_postings[~all_postings.is_dup]
    base = top_shares(dedup, skills, top)
    variants = {
        "with_duplicates": all_postings,
        "one_posting_per_employer": dedup.drop_duplicates("employer"),
        "without_staffing_agencies": dedup[~dedup.employer.apply(is_staffing)],
        "arbeitnow_only": dedup[dedup.source == "arbeitnow"],
        "career_sites_only": dedup[dedup.source == "crawl"],
        "german_language_only": dedup[dedup.lang == "de"],
    }
    out: dict[str, Any] = {"baseline_postings": int(len(dedup)), "top_n": top, "variants": {}}
    for name, sub in variants.items():
        out["variants"][name] = {
            "postings": int(len(sub)),
            **compare(base, top_shares(sub, skills, 10_000)),
        }
    out["staffing_agency_postings"] = int(dedup.employer.apply(is_staffing).sum())
    return out


def suppression_sensitivity(
    postings: pd.DataFrame, cols: list[str], sizes: tuple[int, ...] = (10, 30, 50)
) -> dict[str, Any]:
    """How many groups survive each minimum group size."""
    out: dict[str, Any] = {}
    for col in cols:
        counts = postings[col].astype(str).value_counts()
        out[col] = {
            f"min_{s}": {
                "groups_kept": int((counts >= s).sum()),
                "postings_in_kept_groups": int(counts[counts >= s].sum()),
            }
            for s in sizes
        }
    return out


def trend_test(
    postings: pd.DataFrame,
    skills: pd.DataFrame,
    min_mentions: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
    window_days: int = 100,
) -> dict[str, Any]:
    """Early versus late half of the last `window_days` of publication dates (run per source:
    mixing sources would compare sources, not time)."""
    dates = pd.to_datetime(postings.published, errors="coerce")
    p = postings.assign(_date=dates).dropna(subset=["_date"])
    if not p.empty:
        p = p[p._date >= p._date.max() - pd.Timedelta(days=window_days)]
    if p.empty:
        return {"tested": 0, "note": "no publication dates"}
    cut = pd.Timestamp(p._date.median())
    early, late = p[p._date <= cut], p[p._date > cut]
    keys_e, keys_l = early.posting_key.to_numpy(), late.posting_key.to_numpy()
    counts = (
        skills[skills.posting_key.isin(set(p.posting_key))]
        .groupby("skill_id")
        .posting_key.nunique()
    )
    tested = counts[counts >= min_mentions].index.tolist()
    labels = skills.drop_duplicates("skill_id").set_index("skill_id").label
    has = {sid: set(skills.loc[skills.skill_id == sid, "posting_key"]) for sid in tested}
    rng = np.random.default_rng(seed)
    alpha_bonf = 0.05 / max(1, len(tested))
    rows = []
    for sid in tested:
        e = np.array([k in has[sid] for k in keys_e], dtype=float)
        l_ = np.array([k in has[sid] for k in keys_l], dtype=float)
        if len(e) == 0 or len(l_) == 0:
            continue
        diffs = np.array(
            [rng.choice(l_, len(l_)).mean() - rng.choice(e, len(e)).mean() for _ in range(n_boot)]
        )
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        lo_b, hi_b = np.percentile(diffs, [100 * alpha_bonf / 2, 100 * (1 - alpha_bonf / 2)])
        rows.append(
            {
                "skill": str(labels[sid]),
                "share_early": round(float(e.mean()), 4),
                "share_late": round(float(l_.mean()), 4),
                "diff": round(float(l_.mean() - e.mean()), 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "flag_uncorrected": bool(lo > 0 or hi < 0),
                "flag_bonferroni": bool(lo_b > 0 or hi_b < 0),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "window_start": str(p._date.min().date()),
        "split_date": str(cut.date()),
        "window_end": str(p._date.max().date()),
        "postings_early": int(len(early)),
        "postings_late": int(len(late)),
        "min_mentions": min_mentions,
        "tested": int(len(table)),
        "flagged_uncorrected": int(table.flag_uncorrected.sum()) if len(table) else 0,
        "flagged_bonferroni": int(table.flag_bonferroni.sum()) if len(table) else 0,
        "flagged": table[table.flag_bonferroni].to_dict("records") if len(table) else [],
        "table": table,
    }
