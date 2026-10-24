from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load_freeze_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts/freeze_publication_results.py"
    spec = importlib.util.spec_from_file_location("coact_publication_freeze", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load publication freeze script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_freeze_module()


def test_canonical_hash_is_order_independent() -> None:
    assert FREEZE.canonical_sha256({"a": 1, "b": 2}) == FREEZE.canonical_sha256(
        {"b": 2, "a": 1}
    )


def test_file_hash_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc")
    assert (
        FREEZE.sha256_file(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_json_object_reader_rejects_array(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        FREEZE.read_json(path)
