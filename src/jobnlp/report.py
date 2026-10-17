"""Render result tables from results/ into docs/RESULTS.md and the README headline block.

Every number in these documents is read from a file in results/; nothing is typed by hand.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from jobnlp.config import ROOT

R = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(name: str) -> Any:
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def corpus_section() -> list[str]:
    c = _load("corpus_stats.json")
    m = _load("market/market_summary.json")
    if not c:
        return ["Corpus statistics not available."]
    out = [
        "## Corpus",
        "",
        "| measure | value |",
        "|---|---|",
        f"| stored postings (all sources and countries) | {c['n_total']} |",
        f"| near-duplicates removed | {c['n_duplicates']} |",
        f"| German postings after deduplication | {c['n_germany_dedup']} |",
        f"| excluded job-board hosts / postings | {c.get('excluded_aggregator_hosts', 0)} / "
        f"{c.get('excluded_aggregator_postings', 0)} |",
        f"| German postings by source | {c['germany_by_source']} |",
        f"| German postings by language | {c['germany_by_lang']} |",
        f"| employers (German postings) | {c['employers']} |",
        f"| share of the 10 largest employers | {_pct(c['employer_top10_share'])} |",
        f"| publication dates | {c['published_min']} to {c['published_max']} |",
    ]
    if m:
        out.append(
            f"| postings used in the market analysis (text of at least 200 characters) | "
            f"{m['postings']} |"
        )
    return out


def extraction_section() -> list[str]:
    e = _load("extraction_eval.json")
    t = _load("extraction_tuning.json")
    if not e:
        return ["Extraction evaluation not available."]
    out = ["## Skill extraction (gold set, model-annotated, see the caveat)", ""]
    for split, title in (
        ("test", "Held-out test half (100 postings, used once)"),
        ("dev", "Development half (100 postings, used for tuning)"),
    ):
        out += [
            f"### {title}",
            "",
            "| method | exact P | exact R | exact F1 | partial F1 | skill-set F1 |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in e[split].items():
            if name == "necessity":
                continue
            a = r["all"]
            out.append(
                f"| {name} | {_f(a['exact']['precision'])} | {_f(a['exact']['recall'])} | "
                f"{_f(a['exact']['f1'])} | {_f(a['partial']['f1'])} | "
                f"{_f(a['skill_set']['f1'])} |"
            )
        best = max(
            (k for k in e[split] if k != "necessity"),
            key=lambda k: e[split][k]["all"]["partial"]["f1"],
        )
        b = e[split][best]
        out += [
            "",
            f"By language and section, {best}, partial F1: "
            + ", ".join(
                f"{g} {_f(v['partial']['f1'])} (n gold {v['partial']['n_gold']})"
                for g, v in b.items()
                if g.startswith(("lang=", "section="))
            ),
        ]
        out += [
            f"Recall by annotated type, {best} (partial): "
            + ", ".join(f"{k} {_f(v)}" for k, v in b["recall_by_type_partial"].items())
        ]
        n = e[split]["necessity"]
        out += [
            f"Must-have versus nice-to-have rule on gold spans: accuracy {_f(n['accuracy'])} "
            f"(majority baseline {_f(n['majority_baseline_accuracy'])}), nice-to-have "
            f"precision {_f(n['nice_precision'])}, recall {_f(n['nice_recall'])}.",
            "",
        ]
    if t:
        out += [
            f"Tuning on the development half chose mining thresholds {t['chosen']['min_df']} "
            f"postings and cue ratio {t['chosen']['min_cue_ratio']} "
            f"({t['mined_selected']} mined terms from {t['candidates']} candidates) and "
            f"blocked {len(t['blocked_keys'])} dictionary keys.",
            "",
        ]
    return out


def classifier_section() -> list[str]:
    c = _load("classifier_results.json")
    if not c:
        return ["Classifier results not available."]
    out = [
        "## Occupational-field classifier (platform tags as labels)",
        "",
        f"Train / validation / test postings: {c['n']['train']} / {c['n']['val']} / "
        f"{c['n']['test']}; {len(c['classes'])} classes; splits grouped by employer and "
        "duplicate cluster.",
        "",
        "| model:input | macro-F1 | 95% CI | accuracy | top-3 accuracy | latency ms/posting |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in c["models"].items():
        ci = v["macro_f1_95ci"]
        out.append(
            f"| {k} | {_f(v['macro_f1'])} | {_f(ci[0])} to {_f(ci[1])} | "
            f"{_f(v['accuracy'])} | {_f(v['top3_accuracy'])} | "
            f"{_f(1000 * v['latency_seconds_per_posting'], 1)} |"
        )
    d = c.get("bert_vs_best_baseline")
    if d:
        out += [
            "",
            f"Paired bootstrap, BERT minus {d['baseline']}: mean macro-F1 difference "
            f"{_f(d['mean_diff'])}, 95% CI {_f(d['diff_95ci'][0])} to {_f(d['diff_95ci'][1])}.",
        ]
    for name in sorted(p.name for p in R.glob("classifier_*_eval.json")):
        v = _load(name)
        out.append(
            f"Colab-trained model `{name}`: macro-F1 {_f(v['macro_f1'])} (re-evaluated locally)."
        )
    return out + [""]


def market_section() -> list[str]:
    m = _load("market/market_summary.json")
    if not m:
        return ["Market results not available."]
    top = pd.read_csv(R / "market" / "skill_shares_by_overall.csv").head(15)
    out = [
        "## Market analysis (German postings)",
        "",
        f"{m['postings']} postings, published {m['published_min']} to {m['published_max']}; "
        f"groups under {m['min_group_size']} postings are suppressed.",
        "",
        "| skill | share | 95% Wilson interval |",
        "|---|---|---|",
    ]
    for label, share, lo, hi in zip(top.label, top.share, top.ci_low, top.ci_high, strict=True):
        out.append(f"| {label} | {_pct(float(share))} | {_pct(float(lo))} to {_pct(float(hi))} |")
    reg = m["regions"]
    out += [
        "",
        "Rhine-Main: {} postings ({} data or ML roles); Rhine-Neckar: {} postings ({} data "
        "or ML roles).".format(
            reg["rhine_main"]["postings"],
            reg["rhine_main"]["data_role_postings"],
            reg["rhine_neckar"]["postings"],
            reg["rhine_neckar"]["data_role_postings"],
        ),
        f"Data and ML role postings: {m['data_roles']['postings']} ({m['data_roles']['by_role']}).",
        f"Employer concentration: top employer {_pct(m['employer_concentration']['top1_share'])}"
        f", ten largest {_pct(m['employer_concentration']['top10_share'])}, "
        f"HHI {m['employer_concentration']['hhi']}.",
        f"Role profiles: {m['role_profiles']['k']} NMF profiles, stability across "
        f"{len(m['role_profiles']['seeds'])} seeds: mean adjusted Rand index "
        f"{m['role_profiles']['stability_mean_ari']}.",
        "",
    ]
    return out


def gap_section() -> list[str]:
    s, g, t = (
        _load("gap_eval_sanity.json"),
        _load("gap_eval_guards.json"),
        _load("related_threshold.json"),
    )
    out = ["## CV gap analyser evaluation (no CV data used)", ""]
    if s:
        out.append(
            f"Matching sanity on real postings: AUC same-label versus different-label pairs "
            f"{_f(s['auc_same_vs_different'])} (95% CI {_f(s['auc_95ci'][0])} to "
            f"{_f(s['auc_95ci'][1])}; {s['pairs_same_label']} and "
            f"{s['pairs_different_label']} pairs); self-match mean "
            f"{_f(s['self_match_mean'])} (minimum {_f(s['self_match_min'])}, "
            f"n {s['self_match_n']})."
        )
    if t:
        out.append(
            f"Related-skill threshold from dev pairs: {t['threshold']} (pair F1 "
            f"{_f(t.get('dev_pair_f1'))})."
        )
    if g:
        out.append(
            f"LLM guard evaluation ({g['model']}, {g['n_cases']} real requirement bullets as "
            f"stand-in CV bullets): first-attempt violation rate "
            f"{_pct(g['first_attempt_violation_rate'])} (unsupported skills "
            f"{_pct(g['first_attempt_unsupported_skill_rate'])}, facts "
            f"{_pct(g['first_attempt_fact_violation_rate'])}, invalid JSON "
            f"{_pct(g['first_attempt_invalid_json_rate'])}); final pass rate "
            f"{_pct(g['final_pass_rate'])}; fallback rate {_pct(g['fallback_rate'])}."
        )
    return out + [""]


def headline() -> list[str]:
    lines = []
    c, e, k = (
        _load("corpus_stats.json"),
        _load("extraction_eval.json"),
        _load("classifier_results.json"),
    )
    s = _load("gap_eval_sanity.json")
    if c:
        lines.append(
            f"- Corpus: {c['n_germany_dedup']} German postings after deduplication "
            f"(published {c['published_min']} to {c['published_max']})."
        )
    if e:
        best = max(
            (x for x in e["test"] if x != "necessity"),
            key=lambda x: e["test"][x]["all"]["partial"]["f1"],
        )
        a = e["test"][best]["all"]
        lines.append(
            f"- Skill extraction, held-out gold half ({best}): partial-match F1 "
            f"{_f(a['partial']['f1'])}, exact-span F1 {_f(a['exact']['f1'])} "
            "(gold set annotated by a model, see limitations)."
        )
    if k:
        best = max(k["models"], key=lambda x: k["models"][x]["macro_f1"])
        v = k["models"][best]
        lines.append(
            f"- Occupational-field classifier: best {best}, macro-F1 {_f(v['macro_f1'])} "
            f"(95% CI {_f(v['macro_f1_95ci'][0])} to {_f(v['macro_f1_95ci'][1])})."
        )
    if s:
        lines.append(
            f"- CV matching sanity: same-field versus different-field AUC "
            f"{_f(s['auc_same_vs_different'])}."
        )
    return lines


def write_all() -> None:
    body = ["# Results", "", "Generated by `jobnlp report` from the files in `results/`.", ""]
    for part in (
        corpus_section,
        extraction_section,
        classifier_section,
        market_section,
        gap_section,
    ):
        body += part() + [""]
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        block = "\n".join([START, *headline(), END])
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
