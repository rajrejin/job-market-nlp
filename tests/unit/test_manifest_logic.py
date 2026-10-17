import json
from pathlib import Path

import pytest

from jobnlp.manifest import ManifestError, make_manifest, verify_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest("classifier", seed=0))
    assert verify_manifest(tmp_path, "classifier")["seed"] == 0


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        verify_manifest(tmp_path, "classifier")


def test_wrong_kind_and_incomplete_are_refused(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest("ner", seed=0))
    with pytest.raises(ManifestError):
        verify_manifest(tmp_path, "classifier")
    (tmp_path / "manifest.json").write_text(json.dumps({"kind": "classifier"}), encoding="utf-8")
    with pytest.raises(ManifestError):
        verify_manifest(tmp_path, "classifier")
