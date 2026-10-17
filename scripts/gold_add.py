"""Append annotation batches to annotation/gold_annotations.jsonl (gitignored).

Usage: python scripts/gold_add.py BATCH_FILE
The batch file is JSON: {gid: [[span, type, nec], ...]}.
Every span must be an exact substring of the item's gold_text; misses are reported and not saved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "annotation" / "gold_sample.jsonl"
OUT = ROOT / "annotation" / "gold_annotations.jsonl"
TYPES = {"tool", "knowledge", "soft", "language", "certificate"}


def main(batch_path: str) -> int:
    texts = {
        r["gid"]: r["gold_text"]
        for r in map(json.loads, SAMPLE.read_text(encoding="utf-8").splitlines())
    }
    batch = json.loads(Path(batch_path).read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        done = {json.loads(line)["gid"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    bad = 0
    with OUT.open("a", encoding="utf-8") as f:
        for gid, spans in batch.items():
            if gid in done:
                print(f"{gid}: already annotated, skipped")
                continue
            ok = True
            for span, typ, nec in spans:
                if span not in texts[gid] or typ not in TYPES or nec not in {"must", "nice"}:
                    print(f"{gid}: invalid span {span!r} {typ} {nec}")
                    ok = False
            if ok:
                rec = {"gid": gid, "spans": [{"text": s, "type": t, "nec": n} for s, t, n in spans]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                bad += 1
    print(f"saved batch; items with errors: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
