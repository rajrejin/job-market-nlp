"""Fine-tune a German BERT model for occupational-field classification (CPU or GPU).

Early stopping on validation macro-F1, best checkpoint kept, checkpoints saved every epoch so
an interrupted run (for example a Colab disconnect) resumes from the last one.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "distilbert/distilbert-base-german-cased"


def train_bert(
    x_tr: list[str],
    y_tr: list[str],
    x_va: list[str],
    y_va: list[str],
    classes: list[str],
    out_dir: Path,
    model_id: str = MODEL_ID,
    max_len: int = 256,
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 5e-5,
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    idx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(model_id)

    def encode(texts: list[str], labels: list[str]) -> Dataset:
        # "mixed truncation": keep the title and the start of the description (first tokens)
        enc = tok(texts, truncation=True, max_length=max_len)
        enc["labels"] = [idx[y] for y in labels]
        return Dataset.from_dict(dict(enc))

    ds_tr, ds_va = encode(x_tr, y_tr), encode(x_va, y_va)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=idx
    )

    def compute(pred: Any) -> dict[str, float]:
        yp = np.argmax(pred.predictions, axis=1)
        return {"macro_f1": float(f1_score(pred.label_ids, yp, average="macro", zero_division=0))}

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        # a short warm-up (10 percent of all optimiser steps) stabilises the new head
        warmup_steps=max(1, int(0.1 * epochs * math.ceil(len(x_tr) / (batch_size * grad_accum)))),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        processing_class=tok,
        compute_metrics=compute,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    resume = any((out_dir / "checkpoints").glob("checkpoint-*"))
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume or None)
    seconds = time.time() - t0
    final = out_dir / "model"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    return {
        "model_id": model_id,
        "train_seconds": round(seconds, 1),
        "epochs": epochs,
        "max_len": max_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "seed": seed,
        "n_train": len(x_tr),
        "resumed": resume,
        "log_history": trainer.state.log_history,
    }


def predict_proba(
    model_dir: Path, texts: list[str], max_len: int = 256
) -> tuple[np.ndarray, list[str], float]:
    """Probabilities, class order and mean latency per posting (seconds)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(
                texts[i : i + 16],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    latency = (time.time() - t0) / max(1, len(texts))
    return np.concatenate(out), classes, latency
