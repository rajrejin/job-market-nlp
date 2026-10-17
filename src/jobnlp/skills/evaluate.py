"""Evaluation of skill extraction against the gold set (annotation/, gitignored)."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobnlp.skills.necessity import necessity_at

WORDCHAR = re.compile(r"[\w]")

Span = tuple[int, int]


@dataclass
class GoldItem:
    gid: str
    lang: str
    split: str
    section: str
    text: str
    spans: list[tuple[int, int, str, str]] = field(default_factory=list)  # start, end, type, nec
    strings: list[dict[str, str]] = field(default_factory=list)


def _occurrences(text: str, s: str) -> list[Span]:
    out = []
    for m in re.finditer(re.escape(s), text):
        a, b = m.start(), m.end()
        before = text[a - 1] if a > 0 else " "
        after = text[b] if b < len(text) else " "
        if WORDCHAR.match(before) and WORDCHAR.match(s[0]):
            continue
        if WORDCHAR.match(after) and WORDCHAR.match(s[-1]):
            continue
        out.append((a, b))
    return out


def place_spans(text: str, strings: list[dict[str, str]]) -> list[tuple[int, int, str, str]]:
    """Every word-bounded occurrence of every annotated string, longest strings first,
    without overlaps."""
    placed: list[tuple[int, int, str, str]] = []
    for s in sorted(strings, key=lambda x: -len(x["text"])):
        for a, b in _occurrences(text, s["text"]):
            if not any(a < pb and pa < b for pa, pb, _, _ in placed):
                placed.append((a, b, s["type"], s["nec"]))
    return sorted(placed)


def load_gold(directory: Path) -> list[GoldItem]:
    sample = [
        json.loads(x) for x in (directory / "gold_sample.jsonl").read_text("utf-8").splitlines()
    ]
    ann = {
        a["gid"]: a["spans"]
        for a in map(
            json.loads, (directory / "gold_annotations.jsonl").read_text("utf-8").splitlines()
        )
    }
    items = []
    for r in sample:
        strings = ann.get(r["gid"], [])
        items.append(
            GoldItem(
                r["gid"],
                r["lang"],
                r["split"],
                r.get("section") or "unknown",
                r["gold_text"],
                place_spans(r["gold_text"], strings),
                strings,
            )
        )
    return items


def prf(tp: float, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "tp": round(tp, 2),
        "n_pred": n_pred,
        "n_gold": n_gold,
    }


def partial_tp(pred: list[Span], gold: list[Span]) -> int:
    """One-to-one greedy matching of overlapping spans."""
    used: set[int] = set()
    tp = 0
    for a, b in pred:
        for j, (ga, gb) in enumerate(gold):
            if j not in used and a < gb and ga < b:
                used.add(j)
                tp += 1
                break
    return tp


Predictor = Callable[[GoldItem], list[tuple[int, int, str]]]  # start, end, canonical key
Canon = Callable[[str, str], str]  # (surface, lang) -> canonical key


def evaluate(items: list[GoldItem], predict: Predictor, canon: Canon) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = {}
    fp_strings: Counter[str] = Counter()
    fn_strings: Counter[str] = Counter()

    def add(group: str, **kw: int) -> None:
        c = groups.setdefault(group, Counter())
        c.update(kw)

    type_recall: Counter[str] = Counter()
    type_total: Counter[str] = Counter()
    for it in items:
        preds = predict(it)
        pred_spans = [(a, b) for a, b, _ in preds]
        gold_spans = [(a, b) for a, b, _, _ in it.spans]
        exact = len(set(pred_spans) & set(gold_spans))
        part = partial_tp(pred_spans, gold_spans)
        pred_set = {k for _, _, k in preds}
        gold_set = {canon(it.text[a:b], it.lang) for a, b in gold_spans}
        set_tp = len(pred_set & gold_set)
        for g in ("all", f"lang={it.lang}", f"section={it.section}"):
            add(
                g,
                exact=exact,
                partial=part,
                n_pred=len(pred_spans),
                n_gold=len(gold_spans),
                set_tp=set_tp,
                set_pred=len(pred_set),
                set_gold=len(gold_set),
            )
        for a, b, typ, _ in it.spans:
            type_total[typ] += 1
            if any(a < pb and pa < b for pa, pb in pred_spans):
                type_recall[typ] += 1
        for a, b in set(pred_spans) - set(gold_spans):
            fp_strings[it.text[a:b].lower()] += 1
        for a, b in set(gold_spans) - set(pred_spans):
            fn_strings[it.text[a:b].lower()] += 1
    result: dict[str, Any] = {}
    for g, c in groups.items():
        result[g] = {
            "exact": prf(c["exact"], c["n_pred"], c["n_gold"]),
            "partial": prf(c["partial"], c["n_pred"], c["n_gold"]),
            "skill_set": prf(c["set_tp"], c["set_pred"], c["set_gold"]),
        }
    result["recall_by_type_partial"] = {
        t: round(type_recall[t] / type_total[t], 4) for t in sorted(type_total)
    }
    result["n_items"] = len(items)
    result["errors"] = {
        "false_positives_top30": fp_strings.most_common(30),
        "false_negatives_top30": fn_strings.most_common(30),
    }
    return result


def necessity_accuracy(items: list[GoldItem]) -> dict[str, Any]:
    """Rule accuracy on gold spans (the span positions are given; only the label is predicted)."""
    conf: Counter[str] = Counter()
    for it in items:
        for a, _, _, nec in it.spans:
            conf[f"gold={nec},pred={necessity_at(it.text, a)}"] += 1
    total = sum(conf.values())
    correct = conf["gold=must,pred=must"] + conf["gold=nice,pred=nice"]
    nice_gold = conf["gold=nice,pred=nice"] + conf["gold=nice,pred=must"]
    nice_pred = conf["gold=nice,pred=nice"] + conf["gold=must,pred=nice"]
    return {
        "n_spans": total,
        "accuracy": round(correct / total, 4) if total else None,
        "majority_baseline_accuracy": round(
            (conf["gold=must,pred=must"] + conf["gold=must,pred=nice"]) / total, 4
        )
        if total
        else None,
        "nice_precision": round(conf["gold=nice,pred=nice"] / nice_pred, 4) if nice_pred else None,
        "nice_recall": round(conf["gold=nice,pred=nice"] / nice_gold, 4) if nice_gold else None,
        "confusion": dict(conf),
    }
