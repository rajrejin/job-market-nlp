"""Evaluation of the section splitter against hand-labelled heading lines.

Annotation: for each sampled posting, every short line that is not a bullet (where headings
occur) is labelled with the section it opens (company, tasks, requirements, offer, other) or
'none' when it is ordinary text. Lines not shown to the annotator are content lines.
Metrics:
- heading accuracy: share of true headings that the splitter opens with the right label;
- boundary precision and recall: a predicted boundary is a line where the splitter opens a
  section; it is correct when the line is a true heading (any label);
- line accuracy: every line gets the section of the closest heading above it (true versus
  predicted, 'company' before the first heading); share of lines where both agree.
"""

from __future__ import annotations

from typing import Any

from jobnlp.text.sections import _heading_of, classify_heading

CANDIDATE_MAX_CHARS = 80


def candidate_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.split("\n")):
        s = line.strip()
        if s and len(s) <= CANDIDATE_MAX_CHARS and not s.startswith("- "):
            out.append((i, line))
    return out


def predicted_labels(text: str) -> dict[int, str]:
    """Line index -> label for every line where the splitter opens a section."""
    out = {}
    for i, line in enumerate(text.split("\n")):
        head = _heading_of(line)
        label = classify_heading(head) if head is not None else None
        if label is not None:
            out[i] = label
    return out


def line_sections(n_lines: int, openers: dict[int, str]) -> list[str]:
    current, out = "company", []
    for i in range(n_lines):
        current = openers.get(i, current)
        out.append(current)
    return out


def score(items: list[dict[str, Any]]) -> dict[str, Any]:
    """items: {'text': str, 'labels': {line_index: label or 'none'}}"""
    true_h = pred_h = tp_boundary = correct_label = 0
    lines_total = lines_ok = 0
    for it in items:
        text = it["text"]
        truth = {int(k): v for k, v in it["labels"].items() if v != "none"}
        pred = predicted_labels(text)
        true_h += len(truth)
        pred_h += len(pred)
        tp_boundary += len(set(pred) & set(truth))
        correct_label += sum(pred.get(i) == lab for i, lab in truth.items())
        n = len(text.split("\n"))
        t_sec, p_sec = line_sections(n, truth), line_sections(n, pred)
        lines_total += n
        lines_ok += sum(a == b for a, b in zip(t_sec, p_sec, strict=True))
    return {
        "postings": len(items),
        "true_headings": true_h,
        "predicted_boundaries": pred_h,
        "heading_accuracy": round(correct_label / true_h, 4) if true_h else None,
        "boundary_precision": round(tp_boundary / pred_h, 4) if pred_h else None,
        "boundary_recall": round(tp_boundary / true_h, 4) if true_h else None,
        "line_accuracy": round(lines_ok / lines_total, 4) if lines_total else None,
        "lines": lines_total,
    }
