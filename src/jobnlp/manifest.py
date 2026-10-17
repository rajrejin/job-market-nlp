"""Reproducibility manifests for trained models and corpus snapshots."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from jobnlp.config import ROOT

LIBS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "spacy",
    "torch",
    "transformers",
    "sentence-transformers",
    "duckdb",
)
REQUIRED = ("kind", "created", "config_sha256", "corpus_sha256", "git_commit", "seed", "libraries")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """HEAD of the repository; JOBNLP_GIT_COMMIT overrides it (set by the Colab notebook,
    where the code arrives as a zip without git history)."""
    if os.environ.get("JOBNLP_GIT_COMMIT"):
        return os.environ["JOBNLP_GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    return out


def make_manifest(kind: str, seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    corpus = ROOT / "data" / "processed" / "postings.parquet"
    return {
        "kind": kind,
        "created": dt.datetime.now(dt.UTC).isoformat(),
        "config_sha256": sha256_file(ROOT / "configs" / "default.yaml"),
        "corpus_sha256": sha256_file(corpus) if corpus.exists() else None,
        "git_commit": git_commit(),
        "seed": seed,
        "libraries": library_versions(),
        **(extra or {}),
    }


def write_manifest(model_dir: Path, manifest: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


class ManifestError(RuntimeError):
    pass


def verify_manifest(model_dir: Path, kind: str) -> dict[str, Any]:
    """Refuse a model whose manifest is missing, incomplete or of another kind."""
    path = model_dir / "manifest.json"
    if not path.exists():
        raise ManifestError(f"no manifest in {model_dir.name}")
    m = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise ManifestError(f"manifest of {model_dir.name} lacks {missing}")
    if m["kind"] != kind:
        raise ManifestError(f"{model_dir.name} is a {m['kind']} model, expected {kind}")
    return dict(m)
