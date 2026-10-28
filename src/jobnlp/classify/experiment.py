"""The classifier experiment: dataset, grouped splits, baselines, BERT, metrics, breakdowns."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score

from jobnlp.classify.baselines import EmbeddingLR, TfidfLR
from jobnlp.classify.data import group_split, label_table, text_view
from jobnlp.classify.evaluate import bootstrap_ci, metrics, paired_bootstrap
from jobnlp.manifest import make_manifest, sha256_file, write_manifest

VIEWS = ("title", "description", "title+description")


def dataset(corpus: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, info = label_table(corpus)
    df["split"] = group_split(df)
    df["n_words"] = [len(str(t).split()) for t in df.description.fillna("")]
    return df, info


def split_manifest(df: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
    tr, te = df[df.split == "train"], df[df.split == "test"]
    return {
        "label_info": info,
        "split_counts": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "employers_per_split": df.groupby("split").employer.nunique().to_dict(),
        "employer_overlap_train_test": len(set(tr.employer) & set(te.employer) - {""}),
        "postings": {
            s: sorted(df[df.split == s].posting_id.tolist()) for s in ("train", "val", "test")
        },
    }


def breakdown(te: pd.DataFrame, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Macro-F1 by posting language and by description-length tercile (test split)."""
    idx = {c: i for i, c in enumerate(classes)}
    yt = np.array([idx[y] for y in te.label])
    yp = proba.argmax(axis=1)

    def f1(mask: np.ndarray) -> dict[str, Any]:
        if mask.sum() == 0:
            return {"n": 0, "macro_f1": None}
        return {
            "n": int(mask.sum()),
            "macro_f1": round(
                float(
                    f1_score(
                        yt[mask],
                        yp[mask],
                        average="macro",
                        labels=range(len(classes)),
                        zero_division=0,
                    )
                ),
                4,
            ),
            "accuracy": round(float(np.mean(yt[mask] == yp[mask])), 4),
        }

    lang = te.lang.to_numpy()
    words = te.n_words.to_numpy()
    q1, q2 = np.percentile(words, [100 / 3, 200 / 3])
    return {
        "by_language": {g: f1(lang == g) for g in sorted(set(lang))},
        "by_length_tercile": {
            f"short (<= {int(q1)} words)": f1(words <= q1),
            f"medium ({int(q1)} to {int(q2)} words)": f1((words > q1) & (words <= q2)),
            f"long (> {int(q2)} words)": f1(words > q2),
        },
    }


def _evaluate(te: pd.DataFrame, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    y = te.label.tolist()
    return {**metrics(y, proba, classes), **bootstrap_ci(y, proba, classes)}


def run_baselines(
    df: pd.DataFrame, classes: list[str], which: set[str], tfidf_dir: Path | None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    proc = psutil.Process(os.getpid())
    out: dict[str, Any] = {}
    probas: dict[str, np.ndarray] = {}
    for name, cls in (("tfidf", TfidfLR), ("e5", EmbeddingLR)):
        if name not in which:
            continue
        for view in VIEWS:
            model = cls()
            fit = model.fit(
                text_view(tr, view), tr.label.tolist(), text_view(va, view), va.label.tolist()
            )
            t0 = time.time()
            proba = model.predict_proba(text_view(te, view))
            latency = (time.time() - t0) / max(1, len(te))
            proba = proba[:, [model.classes.index(c) for c in classes]]
            key = f"{model.name}:{view}"
            probas[key] = proba
            out[key] = {
                **_evaluate(te, proba, classes),
                "fit": fit,
                "latency_seconds_per_posting": round(latency, 5),
                "rss_mb_after": int(proc.memory_info().rss / 2**20),
            }
            if view == "title+description":
                out[key]["breakdown"] = breakdown(te, proba, classes)
                if isinstance(model, TfidfLR) and tfidf_dir is not None:
                    save_tfidf(model, classes, tfidf_dir)
    return out, probas


def save_tfidf(model: TfidfLR, classes: list[str], d: Path) -> None:
    import pickle

    d.mkdir(parents=True, exist_ok=True)
    with (d / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    write_manifest(
        d,
        make_manifest(
            "classifier",
            0,
            {"model": "tfidf_lr", "classes": classes, "split_manifest_sha256": _split_sha(d)},
        ),
    )


def _split_sha(model_dir: Path) -> str | None:
    p = model_dir.parents[1] / "results" / "split_manifest.json"
    return sha256_file(p) if p.exists() else None


def run_bert(
    df: pd.DataFrame,
    classes: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
    model_id: str,
    epochs: int,
    max_train: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    from jobnlp.classify.bert import predict_proba, train_bert

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    tr_b = tr.sample(n=min(max_train, len(tr)), random_state=0)
    proc = psutil.Process(os.getpid())
    stats = train_bert(
        text_view(tr_b, "title+description"),
        tr_b.label.tolist(),
        text_view(va, "title+description"),
        va.label.tolist(),
        classes,
        out_dir,
        model_id=model_id,
        max_len=int(cfg["max_len"]),
        epochs=epochs,
        seed=seed,
    )
    proba, order, latency = predict_proba(
        out_dir / "model", text_view(te, "title+description"), max_len=int(cfg["max_len"])
    )
    proba = proba[:, [order.index(c) for c in classes]]
    history = stats.pop("log_history", [])
    write_manifest(
        out_dir / "model",
        make_manifest(
            "classifier",
            seed,
            {
                "model": "bert",
                "model_id": model_id,
                "classes": classes,
                "training": stats,
                "split_manifest_sha256": _split_sha(out_dir / "model"),
            },
        ),
    )
    res = {
        **_evaluate(te, proba, classes),
        "fit": stats,
        "validation_history": [h for h in history if "eval_macro_f1" in h],
        "latency_seconds_per_posting": round(latency, 5),
        "rss_mb_after": int(proc.memory_info().rss / 2**20),
        "breakdown": breakdown(te, proba, classes),
        "budget": f"{len(tr_b)} training examples, {epochs} epochs, seed {seed}",
    }
    return res, proba


def merge_results(path: Path, new: dict[str, Any]) -> dict[str, Any]:
    """Keep model entries from earlier runs (for example baselines when only BERT is rerun)."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = {**old, **{k: v for k, v in new.items() if k != "models"}}
    merged["models"] = {**old.get("models", {}), **new.get("models", {})}
    return merged


def save_probas(d: Path, probas: dict[str, np.ndarray], test_ids: list[str]) -> None:
    """Test-set probabilities per model (models/, not committed) for later paired comparisons."""
    d.mkdir(parents=True, exist_ok=True)
    for key, p in probas.items():
        np.save(d / f"{key.replace(':', '__').replace('+', '_')}.npy", p)
    (d / "test_ids.json").write_text(json.dumps(test_ids), encoding="utf-8")


def compare_with_best_baseline(
    d: Path, results: dict[str, Any], te: pd.DataFrame, bert_key: str, classes: list[str]
) -> dict[str, Any] | None:
    ids = (
        json.loads((d / "test_ids.json").read_text("utf-8"))
        if (d / "test_ids.json").exists()
        else None
    )
    if ids != te.posting_id.tolist():
        return None  # baseline probabilities belong to another split
    base = [
        k for k in results["models"] if k.endswith("title+description") and not k.startswith("bert")
    ]
    if not base:
        return None
    best = max(base, key=lambda k: results["models"][k]["macro_f1"])
    files = {k: d / f"{k.replace(':', '__').replace('+', '_')}.npy" for k in (best, bert_key)}
    if not all(f.exists() for f in files.values()):
        return None
    return {
        "baseline": best,
        **paired_bootstrap(
            te.label.tolist(), np.load(files[bert_key]), np.load(files[best]), classes
        ),
    }
