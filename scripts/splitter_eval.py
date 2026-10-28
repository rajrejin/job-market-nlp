"""Score the section splitter on the 100 labelled postings (annotation/, not committed).

Postings s000-s049 are the development half (used to improve the cue lists), s050-s099 the
held-out half. Writes results/section_splitter_eval.json."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from jobnlp.config import ROOT  # noqa: E402
from jobnlp.text.sections_eval import score  # noqa: E402

rows = [
    json.loads(x)
    for x in (ROOT / "annotation" / "splitter_sample.jsonl").read_text("utf-8").splitlines()
]
labels = json.loads((ROOT / "annotation" / "splitter_labels.json").read_text("utf-8"))
items = [{"sid": r["sid"], "text": r["text"], "labels": labels[r["sid"]]} for r in rows]
dev = [i for i in items if int(i["sid"][1:]) < 50]
test = [i for i in items if int(i["sid"][1:]) >= 50]
result = {"dev_half": score(dev), "held_out_half": score(test), "all": score(items)}
tag = sys.argv[1] if len(sys.argv) > 1 else "current"
out = ROOT / "results" / "section_splitter_eval.json"
prev = json.loads(out.read_text("utf-8")) if out.exists() else {}
prev[tag] = result
out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
print(
    tag,
    json.dumps(
        {
            k: {
                m: v[m]
                for m in (
                    "heading_accuracy",
                    "boundary_precision",
                    "boundary_recall",
                    "line_accuracy",
                )
            }
            for k, v in result.items()
        },
        indent=1,
    ),
)
