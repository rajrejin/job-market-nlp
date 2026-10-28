"""Which surface forms produce the most frequent canonical skills? (diagnostic, prints counts)"""

from collections import Counter

import pandas as pd

from jobnlp.config import ROOT
from jobnlp.market.run import market_postings
from jobnlp.skills.extract import load_nlp
from jobnlp.skills.pipeline import requirement_text
from jobnlp.skills.runtime import load_gazetteer

gaz = load_gazetteer()
p = market_postings(pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet"))
p = p.sample(600, random_state=0)
pairs: Counter[tuple[str, str, str]] = Counter()
for lang, sections, text in zip(p.lang, p.sections, p.text, strict=True):
    body = requirement_text(sections) or text[:3000]
    for m in gaz.match(load_nlp("en" if lang == "en" else "de")(body)):
        pairs[(m.label, m.text.lower(), m.source)] += 1
by_label: Counter[str] = Counter()
for (label, _, _), n in pairs.items():
    by_label[label] += n
for label, n in by_label.most_common(45):
    forms = [(s, src, c) for (lab, s, src), c in pairs.most_common() if lab == label][:4]
    print(n, label, "<-", forms)
