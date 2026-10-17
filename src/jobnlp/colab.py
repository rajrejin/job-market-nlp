"""Colab offload: package code and inputs, then validate and re-evaluate returned models.

Colab is optional. Nothing here uploads anything: `package` writes local files that the user
copies to their own private Drive, and `validate_import` refuses returned models whose manifest
does not match this repository, config, corpus and split manifest.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from jobnlp.config import ROOT
from jobnlp.manifest import git_commit, sha256_file, verify_manifest

CODE_PARTS = ["pyproject.toml", "README.md", "requirements.txt", "configs", "src", "notebooks"]


def package(dist: Path) -> dict[str, Any]:
    """dist/jobnlp_code.zip (code only) and dist/colab_inputs/ (redacted corpus, split manifest,
    checksums). The corpus contains third-party posting text (redacted): keep it private."""
    dist.mkdir(parents=True, exist_ok=True)
    code = dist / "jobnlp_code.zip"
    with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as z:
        for part in CODE_PARTS:
            p = ROOT / part
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
    inputs = dist / "colab_inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data" / "processed" / "postings.parquet", inputs / "postings.parquet")
    shutil.copy2(ROOT / "results" / "split_manifest.json", inputs / "split_manifest.json")
    checks = {
        "git_commit": git_commit(),
        "config_sha256": sha256_file(ROOT / "configs" / "default.yaml"),
        "corpus_sha256": sha256_file(inputs / "postings.parquet"),
        "split_manifest_sha256": sha256_file(inputs / "split_manifest.json"),
        "code_zip_sha256": sha256_file(code),
    }
    (inputs / "expected.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return checks


def validate_import(model_dir: Path) -> dict[str, Any]:
    """Check a returned model folder against the local state; raise on any mismatch."""
    m = verify_manifest(model_dir, "classifier")
    problems = []
    local = {
        "git_commit": git_commit(),
        "config_sha256": sha256_file(ROOT / "configs" / "default.yaml"),
        "corpus_sha256": sha256_file(ROOT / "data" / "processed" / "postings.parquet"),
        "split_manifest_sha256": sha256_file(ROOT / "results" / "split_manifest.json"),
    }
    for key, value in local.items():
        if m.get(key) != value:
            problems.append(f"{key} differs (model {str(m.get(key))[:12]}, local {value[:12]})")
    if m.get("model") != "bert":
        problems.append("manifest model is not 'bert'")
    for f in ("config.json", "model.safetensors"):
        if not (model_dir / f).exists():
            problems.append(f"missing {f}")
    if problems:
        raise ValueError("refusing import: " + "; ".join(problems))
    return m
