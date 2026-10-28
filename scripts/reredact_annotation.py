"""Re-apply the current redaction to the stored splitter sample (annotation/, not committed).
Replacements happen inside lines, so the labelled line numbers stay valid."""

import json

from jobnlp.config import ROOT
from jobnlp.text.redact import redact, spacy_person_finder

pf = spacy_person_finder()
path = ROOT / "annotation" / "splitter_sample.jsonl"
rows = [json.loads(x) for x in path.read_text("utf-8").splitlines()]
for r in rows:
    new = redact(r["text"], pf, "de")
    assert new.count("\n") == r["text"].count("\n")
    r["text"] = new
path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
print("re-redacted", len(rows))
