"""Apply the current redaction to the splitter sample and the gold texts; print changed lines
(names appear only on this terminal) and confirm that no annotated gold span is lost."""

from __future__ import annotations

import json

from jobnlp.config import ROOT
from jobnlp.text.redact import redact, spacy_person_finder

pf = spacy_person_finder()
changed = 0
for name in ("splitter_sample.jsonl", "gold_sample.jsonl"):
    rows = [json.loads(x) for x in (ROOT / "annotation" / name).read_text("utf-8").splitlines()]
    key = "text" if name.startswith("splitter") else "gold_text"
    for r in rows:
        new = redact(r[key], pf, r.get("lang", "de"))
        for a, b in zip(r[key].split("\n"), new.split("\n"), strict=True):
            if a != b:
                changed += 1
                print(f"{name[:8]} | {a[:140]!r} -> {b[:140]!r}")
ann = {
    a["gid"]: a["spans"]
    for a in map(
        json.loads, (ROOT / "annotation" / "gold_annotations.jsonl").read_text("utf-8").splitlines()
    )
}
gold = [
    json.loads(x)
    for x in (ROOT / "annotation" / "gold_sample.jsonl").read_text("utf-8").splitlines()
]
lost = [
    (g["gid"], s["text"])
    for g in gold
    for s in ann[g["gid"]]
    if s["text"] not in redact(g["gold_text"], pf, g["lang"])
]
print("changed lines:", changed, "lost gold spans:", lost)
