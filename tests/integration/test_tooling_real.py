"""Report rendering, Colab packaging and validation, CLI wiring, logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jobnlp import report
from jobnlp.cli import app
from jobnlp.config import ROOT
from jobnlp.logging_utils import get_logger


def test_report_renders_from_results(tmp_path: Path) -> None:
    if not (ROOT / "results" / "corpus_stats.json").exists():
        pytest.skip("results absent")
    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{report.START}\nold\n{report.END}\ny\n", encoding="utf-8")
    report.write_all(docs_dir=tmp_path, readme=readme)
    body = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert body.startswith("# Results") and "## Corpus" in body
    text = readme.read_text(encoding="utf-8")
    assert "PLACEHOLDER_QQ" not in text and text.startswith("x\n") and text.endswith("y\n")
    assert "—" not in body and "–" not in body  # no em or en dashes


def test_report_handles_missing_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(report, "R", tmp_path)
    assert report.extraction_section() == ["Extraction evaluation not available."]
    assert report.headline() == []


def test_colab_package_and_refuse_foreign_model(tmp_path: Path) -> None:
    from jobnlp.colab import package, validate_import
    from jobnlp.manifest import make_manifest, write_manifest

    if not (ROOT / "results" / "split_manifest.json").exists():
        pytest.skip("split manifest absent: run train-classifier")
    checks = package(tmp_path / "dist")
    assert (tmp_path / "dist" / "jobnlp_code.zip").exists()
    assert json.loads((tmp_path / "dist" / "colab_inputs" / "expected.json").read_text())
    assert checks["corpus_sha256"]
    model = tmp_path / "model"
    write_manifest(
        model, make_manifest("classifier", 0, {"model": "bert", "split_manifest_sha256": "other"})
    )
    with pytest.raises(ValueError, match="split_manifest_sha256 differs"):
        validate_import(model)


def test_cli_version_and_help() -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["version"]).output.strip() == "0.1.0"
    out = runner.invoke(app, ["--help"]).output
    for cmd in ("harvest", "crawl", "build-corpus", "train-ner", "evaluate", "market", "gap"):
        assert cmd in out


def test_cli_gap_runs_on_real_postings(tmp_path: Path, corpus_df) -> None:  # type: ignore[no-untyped-def]
    p = corpus_df[corpus_df.in_germany & (corpus_df.n_chars > 1000)].sample(2, random_state=3)
    job, cv = tmp_path / "job.txt", tmp_path / "cv.txt"
    job.write_text(p.text.iloc[0], encoding="utf-8")
    cv.write_text(p.text.iloc[1], encoding="utf-8")
    res = CliRunner().invoke(app, ["gap", str(job), str(cv), "--no-llm"])
    assert res.exit_code == 0, res.output
    assert "disclaimer" in json.loads(res.output)


def test_logger_writes_file_without_duplicate_handlers() -> None:
    log = get_logger("jobnlp.test", "test.log")
    again = get_logger("jobnlp.test", "test.log")
    assert log is again and len(log.handlers) == 2 and log.level == logging.INFO
