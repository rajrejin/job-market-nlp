"""Configuration loading and the project root."""

import os
from pathlib import Path
from typing import Any

import yaml


def _root() -> Path:
    """JOBNLP_ROOT if set (containers); the source checkout for an editable install;
    otherwise the current working directory."""
    env = os.environ.get("JOBNLP_ROOT")
    if env:
        return Path(env)
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "configs" / "default.yaml").exists():
        return checkout
    return Path.cwd()


ROOT = _root()


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or ROOT / "configs" / "default.yaml"
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data
