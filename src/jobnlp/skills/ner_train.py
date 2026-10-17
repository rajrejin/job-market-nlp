"""Train a spaCy NER model (label SKILL) on weak labels from the gazetteer.

The training texts are requirement and task sections of postings that are not in the gold set
(nor in a gold item's duplicate cluster). Labels come from the dictionary matcher, so the model
can only generalise from patterns the dictionary already marks; its evaluation is on gold data.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.training import Example

from jobnlp.skills.extract import load_nlp
from jobnlp.skills.gazetteer import Gazetteer
from jobnlp.text.lang import detect


def weak_examples(
    nlp: Language, gaz: Gazetteer, texts: list[str], max_chars: int = 1500
) -> list[Example]:
    examples = []
    for text in texts:
        text = text[:max_chars]
        if not text.strip():
            continue
        lang = detect(text)
        matches = gaz.match(load_nlp("en" if lang == "en" else "de")(text))
        doc = nlp.make_doc(text)
        ents: list[Span] = []
        for m in matches:
            span = doc.char_span(m.start, m.end, label="SKILL", alignment_mode="contract")
            if span is not None and (not ents or span.start >= ents[-1].end):
                ents.append(span)
        examples.append(
            Example.from_dict(
                doc, {"entities": [(e.start_char, e.end_char, "SKILL") for e in ents]}
            )
        )
    return examples


def train(
    gaz: Gazetteer,
    texts: list[str],
    out_dir: Path,
    epochs: int = 8,
    seed: int = 0,
    dev_fraction: float = 0.1,
) -> dict[str, Any]:
    random.seed(seed)
    spacy.util.fix_random_seed(seed)
    nlp = spacy.blank("de")  # German tokenizer rules also split English text sensibly
    ner = nlp.add_pipe("ner")
    ner.add_label("SKILL")  # type: ignore[attr-defined]
    examples = weak_examples(nlp, gaz, texts)
    random.shuffle(examples)
    n_dev = max(1, int(len(examples) * dev_fraction))
    dev, train_ex = examples[:n_dev], examples[n_dev:]
    nlp.initialize(lambda: train_ex)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        random.shuffle(train_ex)
        losses: dict[str, float] = {}
        for batch in spacy.util.minibatch(train_ex, size=16):
            nlp.update(batch, drop=0.2, losses=losses)
        scores = nlp.evaluate(dev)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(losses.get("ner", 0.0), 2),
                "weak_dev_f": round(float(scores.get("ents_f") or 0.0), 4),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out_dir)
    return {
        "n_train": len(train_ex),
        "n_weak_dev": len(dev),
        "epochs": epochs,
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
