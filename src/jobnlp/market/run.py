"""Market analysis from the processed corpus: one command regenerates every output.

Skills are extracted from the requirements and tasks sections (the whole text when no such
section was found) of German, deduplicated postings, with the tuned dictionary matcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from jobnlp.market.aggregate import data_role, employer_concentration, experience_level, shares
from jobnlp.market.robustness import robustness, suppression_sensitivity, trend_test
from jobnlp.market.structure import binary_matrix, pmi_edges, role_profiles
from jobnlp.skills.extract import load_nlp
from jobnlp.skills.gazetteer import Gazetteer
from jobnlp.skills.pipeline import requirement_text

matplotlib.use("Agg")
MIN_CHARS = 200


def market_postings(corpus: pd.DataFrame, include_duplicates: bool = False) -> pd.DataFrame:
    # postings without a usable description (some JSON-LD lacks it) cannot show skills
    keep = corpus.in_germany & (corpus.n_chars >= MIN_CHARS)
    if not include_duplicates:
        keep &= ~corpus.is_dup
    p = corpus[keep].copy()
    p["posting_key"] = p.source + ":" + p.posting_id
    p["experience"] = [
        experience_level(t or "", x or "") for t, x in zip(p.title, p.text, strict=True)
    ]
    p["data_role"] = [data_role(t or "") for t in p.title]
    p["category"] = p.categories.apply(lambda c: next((t for t in c if t != "Remote"), "none"))
    p["published_week"] = pd.to_datetime(p.published, errors="coerce").dt.to_period("W").astype(str)
    p["state"] = p.state.fillna("unknown")
    p["region"] = p.region.fillna("other")
    return p


def extract_skills(postings: pd.DataFrame, gaz: Gazetteer) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, lang, sections, text in zip(
        postings.posting_key, postings.lang, postings.sections, postings.text, strict=True
    ):
        body = requirement_text(sections) or (text or "")[:3000]
        doc = load_nlp("en" if lang == "en" else "de")(body)
        seen: set[str] = set()
        for m in gaz.match(doc):
            if m.skill_id in seen:
                continue
            seen.add(m.skill_id)
            rows.append(
                {"posting_key": key, "skill_id": m.skill_id, "label": m.label, "source": m.source}
            )
    return pd.DataFrame(rows)


def run(corpus: pd.DataFrame, gaz: Gazetteer, out: Path, min_group: int = 30) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    everything = market_postings(corpus, include_duplicates=True)
    all_skills = extract_skills(everything, gaz)  # duplicates included, for robustness only
    p = everything[~everything.is_dup]
    skills = all_skills[all_skills.posting_key.isin(set(p.posting_key))]
    processed = out.parents[1] / "data" / "processed"
    if processed.exists():  # per-posting skills for the CV tool's demand ranking (not committed)
        skills.to_parquet(processed / "market_skills.parquet", index=False)
        p[["posting_key", "category", "source", "lang"]].to_parquet(
            processed / "market_postings.parquet", index=False
        )
    summary: dict[str, Any] = {
        "postings": int(len(p)),
        "postings_with_skills": int(skills.posting_key.nunique()) if len(skills) else 0,
        "skill_mentions": int(len(skills)),
        "distinct_skills": int(skills.skill_id.nunique()) if len(skills) else 0,
        "by_source": p.source.value_counts().to_dict(),
        "published_min": str(pd.to_datetime(p.published, errors="coerce").min().date()),
        "published_max": str(pd.to_datetime(p.published, errors="coerce").max().date()),
        "min_group_size": min_group,
        "employer_concentration": employer_concentration(p),
        "suppressed_groups": {},
    }
    for name, col in [
        ("overall", None),
        ("state", "state"),
        ("region", "region"),
        ("category", "category"),
        ("language", "lang"),
        ("experience", "experience"),
        ("remote", "remote"),
        ("data_role", "data_role"),
        ("week", "published_week"),
    ]:
        sub = p if col != "data_role" else p[p.data_role.notna()]
        table, suppressed = shares(sub, skills, col, min_group=min_group)
        table.to_csv(out / f"skill_shares_by_{name}.csv", index=False, encoding="utf-8")
        summary["suppressed_groups"][name] = suppressed
    # one posting per employer: sensitivity of the overall shares to employer concentration
    one_per = p.drop_duplicates("employer")
    table, _ = shares(one_per, skills, None, min_group=min_group)
    table.to_csv(out / "skill_shares_one_per_employer.csv", index=False, encoding="utf-8")
    edges = pmi_edges(skills, len(p), min_pair=8) if len(skills) else pd.DataFrame()
    edges.to_csv(out / "cooccurrence_pmi.csv", index=False, encoding="utf-8")
    m, _, top = binary_matrix(skills, top_n=120)
    labels = skills.drop_duplicates("skill_id").set_index("skill_id").label
    profiles = role_profiles(m, [str(labels[s]) for s in top], k=6)
    summary["role_profiles"] = profiles
    data = p[p.data_role.notna()]
    summary["data_roles"] = {
        "postings": int(len(data)),
        "by_role": data.data_role.value_counts().to_dict(),
        "by_region": data.region.value_counts().to_dict(),
        "employer_concentration": employer_concentration(data) if len(data) else None,
    }
    summary["regions"] = {
        r: {
            "postings": int((p.region == r).sum()),
            "data_role_postings": int(((p.region == r) & p.data_role.notna()).sum()),
        }
        for r in ("rhine_main", "rhine_neckar")
    }
    rob = robustness(everything, all_skills)
    rob["suppression_sensitivity"] = suppression_sensitivity(p, ["state", "category", "region"])
    (out / "robustness.json").write_text(json.dumps(rob, indent=1, ensure_ascii=False), "utf-8")
    trends: dict[str, Any] = {}
    for src in sorted(p.source.unique()):
        trend = trend_test(p[p.source == src], skills)
        table = trend.pop("table", pd.DataFrame())
        table.to_csv(out / f"trend_tests_{src}.csv", index=False, encoding="utf-8")
        trends[src] = trend
    (out / "trends.json").write_text(json.dumps(trends, indent=1, ensure_ascii=False), "utf-8")
    summary["trends"] = {
        src: {k: t[k] for k in ("tested", "flagged_uncorrected", "flagged_bonferroni") if k in t}
        for src, t in trends.items()
    }
    (out / "market_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    figures(out)
    return summary


def figures(out: Path) -> None:
    import matplotlib.pyplot as plt

    overall = pd.read_csv(out / "skill_shares_by_overall.csv").head(20).iloc[::-1]
    if overall.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    err = [overall.share - overall.ci_low, overall.ci_high - overall.share]
    ax.barh(overall.label.str.slice(0, 40), overall.share, xerr=err, color="#4C72B0", capsize=2)
    ax.set_xlabel("share of postings (95% Wilson interval)")
    n = int(overall.postings_in_group.iloc[0])
    ax.set_title(f"Most requested skills, German postings (n = {n})")
    fig.tight_layout()
    fig.savefig(out / "top_skills.png", dpi=130)
    plt.close(fig)
