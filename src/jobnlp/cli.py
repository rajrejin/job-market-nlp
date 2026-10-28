"""jobnlp command line interface."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from jobnlp.config import ROOT, load_config

if TYPE_CHECKING:
    from jobnlp.harvest.client import PoliteClient
    from jobnlp.harvest.store import PostingStore

TIMED = {
    "harvest",
    "crawl",
    "esco",
    "build-corpus",
    "train-ner",
    "evaluate",
    "market",
    "train-classifier",
    "evaluate-gap",
}
_RUN: dict[str, object] = {}


def _record_runtime(*_: object, **__: object) -> None:
    """After a pipeline step succeeds, record its wall-clock time in results/runtime.json."""
    import time

    step = _RUN.get("step")
    if step not in TIMED:
        return
    path = ROOT / "results" / "runtime.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[str(step)] = {
        "seconds": round(time.time() - float(_RUN["t0"]), 1),  # type: ignore[arg-type]
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "machine": "Ryzen 5 laptop, 8 GB RAM, CPU only, Docker VM with 3.5 GB",
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


app = typer.Typer(
    no_args_is_help=True,
    help="German job market NLP analyser and CV gap tool.",
    result_callback=_record_runtime,
)


@app.callback()
def _start(ctx: typer.Context) -> None:
    import time

    _RUN.update(step=ctx.invoked_subcommand, t0=time.time())


@app.command()
def version() -> None:
    """Print the package version."""
    from jobnlp import __version__

    typer.echo(__version__)


def _client(min_interval: float | None = None, max_retries: int = 5) -> PoliteClient:
    from jobnlp.harvest.client import PoliteClient

    cfg = load_config()["harvest"]
    return PoliteClient(
        user_agent=cfg["user_agent"],
        cache_path=ROOT / "stores" / "http_cache.sqlite",
        min_interval_s=min_interval or cfg["min_interval_s"],
        max_retries=max_retries,
    )


def _store() -> PostingStore:
    from jobnlp.harvest.store import PostingStore

    return PostingStore(ROOT / "stores" / "postings.sqlite")


@app.command()
def harvest(snapshot: str = typer.Option("", help="Snapshot date, default today (UTC).")) -> None:
    """Take a snapshot of the Arbeitnow public API."""
    from jobnlp.harvest.arbeitnow import harvest_snapshot

    snap = snapshot or dt.datetime.now(dt.UTC).date().isoformat()
    store = _store()
    stats = harvest_snapshot(_client(), store, snap)
    out = ROOT / "results" / "harvest_progress.json"
    out.write_text(
        json.dumps({"snapshot": snap, **stats, "totals": store.counts()}, indent=1),
        encoding="utf-8",
    )
    typer.echo(json.dumps(stats))


@app.command()
def crawl(
    n_hosts: int = typer.Option(0, help="Random sample of hosts (seed 0); 0 means all."),
    max_pages: int = typer.Option(200, help="Job pages per host per snapshot."),
    workers: int = typer.Option(16, help="Hosts crawled in parallel."),
    snapshot: str = typer.Option("", help="Snapshot date, default today (UTC)."),
    out: str = typer.Option("results/crawl_summary.json", help="Summary output path."),
) -> None:
    """Crawl employer career sites with JobPosting markup (robots.txt and TDM opt-outs honoured)."""
    import io
    import random
    import zipfile

    from jobnlp.harvest.client import PoliteClient
    from jobnlp.harvest.crawl import candidate_hosts, run_crawl, summarise

    cfg = load_config()["harvest"]
    stats_zip = ROOT / "data" / "raw" / "wdc" / "stats.zip"
    csv_path = ROOT / "data" / "interim" / "wdc_jobposting_hosts.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(stats_zip) as z:
            raw = z.read("table_statistics/JobPosting_September2020_statistics.csv")
        csv_path.write_text(io.BytesIO(raw).read().decode("utf-8"), encoding="utf-8")
    hosts = candidate_hosts(csv_path)
    if n_hosts:
        hosts = random.Random(0).sample(hosts, min(n_hosts, len(hosts)))
    snap = snapshot or dt.datetime.now(dt.UTC).date().isoformat()
    client = PoliteClient(
        user_agent=cfg["user_agent"],
        cache_path=ROOT / "stores" / "http_cache.sqlite",
        min_interval_s=cfg["crawl_min_interval_s"],
        max_retries=2,
        timeout_s=20.0,
    )
    store = _store()
    reports = run_crawl(client, store, hosts, cfg["user_agent"], snap, max_pages, workers)
    summary = {"snapshot": snap, "candidate_hosts": len(hosts), **summarise(reports)}
    (ROOT / out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    hosts_out = ROOT / "data" / "interim" / f"crawl_hosts_{snap}.json"
    hosts_out.write_text(json.dumps([r.__dict__ for r in reports], indent=1), encoding="utf-8")
    typer.echo(json.dumps(summary))


@app.command()
def esco() -> None:
    """Fetch ESCO skills (German and English labels) from the public ESCO API."""
    from jobnlp.skills.esco import fetch_skills

    df = fetch_skills(_client(max_retries=1), ROOT / "data" / "raw" / "esco")
    out = ROOT / "data" / "interim" / "esco_skills.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summary = {
        "skills": int(len(df)),
        "api_total": None,
        "with_german_label": int(df.pref_de.notna().sum()),
        "digital": int(df.is_digital.sum()),
        "by_type": df.skill_type.value_counts().to_dict(),
        "accessed": dt.datetime.now(dt.UTC).date().isoformat(),
        "source": "ESCO web service API https://ec.europa.eu/esco/api",
    }
    (ROOT / "results" / "esco_summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    typer.echo(json.dumps(summary))


def _gold_ids() -> set[tuple[str, str]]:
    rows = (ROOT / "annotation" / "gold_sample.jsonl").read_text(encoding="utf-8").splitlines()
    return {(r["source"], r["posting_id"]) for r in map(json.loads, rows)}


def _jsonable(x: object) -> object:
    """numpy scalars and arrays (from spaCy, scikit-learn) to plain Python values."""
    if hasattr(x, "item") and callable(x.item) and getattr(x, "ndim", 1) == 0:
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def _write_result(name: str, data: object) -> None:
    (ROOT / "results" / name).write_text(
        json.dumps(data, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8"
    )


@app.command("train-ner")
def train_ner(
    max_docs: int = typer.Option(3000, help="Weakly labelled training documents."),
    epochs: int = typer.Option(8),
) -> None:
    """Tune the dictionary matcher on gold dev data, then train NER on its weak labels."""
    import pandas as pd

    from jobnlp.skills.evaluate import load_gold
    from jobnlp.skills.ner_train import train
    from jobnlp.skills.pipeline import mined_table, requirement_text, tune, unseen_postings

    esco = pd.read_parquet(ROOT / "data" / "interim" / "esco_skills.parquet")
    corpus = pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet")
    gold = load_gold(ROOT / "annotation")
    dev = [g for g in gold if g.split == "dev"]
    base, mined_gaz, info, selected = tune(esco, corpus, dev, _gold_ids())
    mined_table(selected).to_csv(ROOT / "results" / "mined_skills.csv", index=False)
    cfg = {"mined_terms": [t.term for t in selected], "blocked_keys": info["blocked_keys"]}
    (ROOT / "stores").mkdir(exist_ok=True)
    (ROOT / "stores" / "skills_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    pool = unseen_postings(corpus, _gold_ids()).sample(frac=1, random_state=0).head(max_docs)
    texts = [requirement_text(s) for s in pool.sections]
    stats = train(mined_gaz, texts, ROOT / "models" / "ner_weak", epochs=epochs)
    from jobnlp.manifest import make_manifest, write_manifest

    write_manifest(ROOT / "models" / "ner_weak", make_manifest("ner", 0, {"training": stats}))
    _write_result("extraction_tuning.json", info)
    _write_result("ner_training.json", stats)
    typer.echo(
        json.dumps(
            {
                "mined": len(selected),
                "blocked": len(info["blocked_keys"]),
                "ner_seconds": stats["seconds"],
            }
        )
    )


@app.command()
def evaluate(what: str = typer.Option("extraction", help="extraction")) -> None:
    """Evaluate skill extraction methods on the gold dev and held-out test halves."""
    import pandas as pd
    import spacy

    from jobnlp.skills.evaluate import load_gold
    from jobnlp.skills.extract import build_gazetteer
    from jobnlp.skills.mine import MinedTerm
    from jobnlp.skills.pipeline import evaluate_all, with_mined

    if what != "extraction":
        raise typer.BadParameter(
            "only 'extraction' is evaluated here; classifiers by train-classifier"
        )
    esco = pd.read_parquet(ROOT / "data" / "interim" / "esco_skills.parquet")
    cfg = json.loads((ROOT / "stores" / "skills_config.json").read_text(encoding="utf-8"))
    base = build_gazetteer(esco, blocked=cfg["blocked_keys"])
    mined_gaz = with_mined(base, [MinedTerm(t, 0, 0, 0) for t in cfg["mined_terms"]])
    ner_dir = ROOT / "models" / "ner_weak"
    ner = spacy.load(ner_dir) if ner_dir.exists() else None
    gold = load_gold(ROOT / "annotation")
    results = evaluate_all(base, mined_gaz, ner, gold)
    _write_result("extraction_eval.json", results)
    for split in ("dev", "test"):
        for m, r in results[split].items():
            if m != "necessity":
                typer.echo(
                    f"{split} {m}: partial F1 {r['all']['partial']['f1']}"
                    f" exact F1 {r['all']['exact']['f1']}"
                )
        typer.echo(f"{split} necessity accuracy {results[split]['necessity']['accuracy']}")


@app.command("train-classifier")
def train_classifier(
    models: str = typer.Option("tfidf,e5,bert", help="Comma list of tfidf, e5, bert."),
    bert_epochs: int = typer.Option(0, help="0 = value from config."),
    bert_model: str = typer.Option("", help="Hugging Face id; default from config."),
    max_train: int = typer.Option(0, help="Cap on BERT training examples; 0 = config."),
    seed: int = typer.Option(0, help="Seed for BERT fine-tuning."),
    out_dir: str = typer.Option("", help="BERT output folder; default models/classifier_bert."),
) -> None:
    """Occupational-field classifier: baselines and BERT on grouped splits, with intervals."""
    import pandas as pd

    from jobnlp.classify import experiment as ex

    cfg = load_config()["classifier"]
    df, info = ex.dataset(pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet"))
    _write_result("split_manifest.json", ex.split_manifest(df, info))
    classes = sorted(info["classes"])
    te = df[df.split == "test"]
    which = set(models.split(","))
    proba_dir = ROOT / "models" / "classifier_probas"
    new: dict[str, object] = {
        "n": df.split.value_counts().to_dict(),
        "classes": classes,
        "models": {},
    }
    base_res, probas = ex.run_baselines(df, classes, which, ROOT / "models" / "classifier_tfidf")
    new["models"] = base_res
    for key, r in base_res.items():
        typer.echo(f"{key}: macro-F1 {r['macro_f1']}")
    if probas:
        ex.save_probas(proba_dir, probas, te.posting_id.tolist())
    if "bert" in which:
        bert_dir = Path(out_dir) if out_dir else ROOT / "models" / "classifier_bert"
        res, proba = ex.run_bert(
            df,
            classes,
            cfg,
            bert_dir,
            bert_model or cfg["model"],
            bert_epochs or int(cfg["epochs"]),
            max_train or int(cfg["local_max_train"]),
            seed,
        )
        key = f"bert:title+description:seed{seed}" if out_dir else "bert:title+description"
        new["models"][key] = res  # type: ignore[index]
        typer.echo(f"{key}: macro-F1 {res['macro_f1']}")
        ex.save_probas(proba_dir, {key: proba}, te.posting_id.tolist())
    results = ex.merge_results(ROOT / "results" / "classifier_results.json", new)
    bert_keys = [k for k in results["models"] if k.startswith("bert")]
    if bert_keys:
        cmp = ex.compare_with_best_baseline(proba_dir, results, te, bert_keys[0], classes)
        if cmp:
            results["bert_vs_best_baseline"] = cmp
    _write_result("classifier_results.json", results)


def _sha(path: object) -> str:
    from pathlib import Path

    from jobnlp.manifest import sha256_file

    return sha256_file(Path(str(path)))


@app.command()
def market(min_group: int = typer.Option(0, help="0 = config value.")) -> None:
    """Regenerate every market analysis output from the processed corpus."""
    import pandas as pd

    from jobnlp.market.run import run
    from jobnlp.skills.runtime import load_gazetteer

    corpus = pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet")
    n = min_group or int(load_config()["market"]["min_group_size"])
    summary = run(corpus, load_gazetteer(), ROOT / "results" / "market", n)
    typer.echo(
        json.dumps({k: summary[k] for k in ("postings", "postings_with_skills", "distinct_skills")})
    )


@app.command()
def gap(
    job: str = typer.Argument(..., help="Path to the job description (.txt)."),
    cv: str = typer.Argument(..., help="Path to the CV (.txt, .pdf, .docx)."),
    no_llm: bool = typer.Option(False, help="Skip LLM suggestions."),
) -> None:
    """CV gap analysis on your machine. Output goes to the terminal only, never to a file."""
    from pathlib import Path

    from jobnlp.gap.llm import LocalLLM
    from jobnlp.gap.parse_cv import read_cv
    from jobnlp.gap.report import run_gap
    from jobnlp.skills.runtime import get_extractor

    job_text = Path(job).read_text(encoding="utf-8")
    cv_text = read_cv(Path(cv).read_bytes(), Path(cv).name)
    thr = _related_threshold()
    res = run_gap(
        job_text,
        cv_text,
        get_extractor(),
        related_threshold=thr,
        llm=None if no_llm else LocalLLM(),
    )
    typer.echo(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def _related_threshold() -> float:
    p = ROOT / "results" / "related_threshold.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return 0.9


@app.command("evaluate-gap")
def evaluate_gap(
    n_pairs: int = typer.Option(300),
    n_guard: int = typer.Option(40),
    skip_llm: bool = typer.Option(False),
) -> None:
    """Sanity AUC on real posting pairs, related-skill threshold, and LLM guard rates."""
    import pandas as pd

    from jobnlp.classify.data import label_table
    from jobnlp.gap.embed import get_embedder
    from jobnlp.gap.evaluate import guard_eval, sanity_auc
    from jobnlp.gap.llm import LocalLLM
    from jobnlp.gap.related import tune_related_threshold
    from jobnlp.skills.evaluate import load_gold
    from jobnlp.skills.pipeline import canon_for, requirement_text, unseen_postings
    from jobnlp.skills.runtime import get_extractor, load_gazetteer

    extractor = get_extractor()
    embedder = get_embedder()
    gold = [g for g in load_gold(ROOT / "annotation") if g.split == "dev"]
    thr = tune_related_threshold(
        gold,
        canon_for(load_gazetteer()),
        embedder,
        pd.read_parquet(ROOT / "data" / "interim" / "esco_skills.parquet"),
    )
    _write_result("related_threshold.json", thr)
    corpus = pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet")
    labelled, _ = label_table(corpus)
    res = {
        "sanity": sanity_auc(
            labelled, extractor, embedder, n_pairs=n_pairs, related_threshold=thr["threshold"]
        )
    }
    _write_result("gap_eval_sanity.json", res["sanity"])
    typer.echo(json.dumps(res["sanity"]))
    llm = LocalLLM()
    if skip_llm or not llm.available():
        typer.echo("LLM unavailable or skipped: guard evaluation not run")
        return
    pool = unseen_postings(corpus, _gold_ids())
    pool = pool.assign(req=[requirement_text(s) for s in pool.sections])
    g = guard_eval(pool, extractor, llm, n=n_guard)
    _write_result("gap_eval_guards.json", g)
    typer.echo(json.dumps(g))


@app.command("annotate-export")
def annotate_export() -> None:
    """Write the gold set as editable JSONL and a CSV review sheet (annotation/, gitignored)."""
    import csv

    from jobnlp.skills.evaluate import load_gold

    gold = load_gold(ROOT / "annotation")
    out = ROOT / "annotation" / "gold_review.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "split", "lang", "span", "type", "necessity", "context"])
        for it in gold:
            for a, b, typ, nec in it.spans:
                w.writerow(
                    [
                        it.gid,
                        it.split,
                        it.lang,
                        it.text[a:b],
                        typ,
                        nec,
                        it.text[max(0, a - 60) : b + 60].replace("\n", " "),
                    ]
                )
    typer.echo(
        f"wrote {out}; edit annotation/gold_annotations.jsonl or this sheet, then rerun "
        "`jobnlp evaluate`"
    )


@app.command("package-colab")
def package_colab() -> None:
    """Write dist/jobnlp_code.zip and dist/colab_inputs/ for the optional Colab run."""
    from jobnlp.colab import package

    typer.echo(json.dumps(package(ROOT / "dist"), indent=1))


@app.command("import-colab")
def import_colab(
    folder: str = typer.Argument(..., help="Downloaded model folder (contains manifest.json)."),
    name: str = typer.Option("colab", help="Name for the imported model."),
) -> None:
    """Validate a Colab-trained BERT model, copy it in, and re-evaluate it locally."""
    import shutil

    import pandas as pd

    from jobnlp.classify.bert import predict_proba
    from jobnlp.classify.data import group_split, label_table, text_view
    from jobnlp.classify.evaluate import bootstrap_ci, metrics
    from jobnlp.colab import validate_import

    src = Path(folder)
    manifest = validate_import(src)
    dest = ROOT / "models" / f"classifier_bert_{name}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    df, info = label_table(pd.read_parquet(ROOT / "data" / "processed" / "postings.parquet"))
    df["split"] = group_split(df)
    te = df[df.split == "test"]
    classes = sorted(info["classes"])
    proba, order, latency = predict_proba(dest, text_view(te, "title+description"))
    proba = proba[:, [order.index(c) for c in classes]]
    res = {
        "source": "Colab-trained model, evaluated locally on the local test split",
        "manifest": {k: manifest.get(k) for k in ("model_id", "seed", "git_commit", "created")},
        **metrics(te.label.tolist(), proba, classes),
        **bootstrap_ci(te.label.tolist(), proba, classes),
        "latency_seconds_per_posting_local_cpu": round(latency, 5),
    }
    _write_result(f"classifier_{name}_eval.json", res)
    typer.echo(f"imported to {dest}; macro-F1 {res['macro_f1']}")


@app.command()
def report() -> None:
    """Regenerate docs/RESULTS.md and the README headline block from results/."""
    from jobnlp.report import write_all

    write_all()
    typer.echo("wrote docs/RESULTS.md and README headline")


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port)


@app.command("build-corpus")
def build_corpus(no_ner: bool = typer.Option(False, help="Skip spaCy name redaction.")) -> None:
    """Clean, redact, locate and deduplicate all stored postings into Parquet."""
    from jobnlp.text.corpus import build, corpus_stats
    from jobnlp.text.geo import GermanPlaces
    from jobnlp.text.redact import spacy_person_finder

    cfg = load_config()
    places = GermanPlaces(ROOT / "data" / "raw" / "geonames" / "DE.zip")
    finder = None if no_ner else spacy_person_finder()
    df = build(_store().iter_rows(), places, cfg["regions"], finder)
    out = ROOT / "data" / "processed" / "postings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    stats = corpus_stats(df)
    (ROOT / "results" / "corpus_stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps({k: stats[k] for k in ("n_total", "n_duplicates", "n_germany_dedup")}))


if __name__ == "__main__":
    app()
