"""Canonical roots must accept aliases while retaining confinement checks."""

import importlib.util
from pathlib import Path

import pytest
from pilot_evidence_fixture import fixture

BASE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def alias_root(tmp_path):
    # Lexically distinct paths for the same directory, without symlink privileges.
    (tmp_path / "parent").mkdir()
    root = tmp_path / "parent" / ".." / "project"
    root.mkdir()
    assert root != root.resolve()
    return root


def test_imported_source_accepts_canonical_root_alias(alias_root):
    runner = load("run_vehicle_sumo_pilot")
    source = alias_root / "src/coact_cps/types.py"
    source.parent.mkdir(parents=True)
    source.write_text("example")
    runner.check_source_root(alias_root, {"src/coact_cps/types.py": source})
    with pytest.raises(ValueError):
        runner.check_source_root(alias_root, {"src/coact_cps/types.py": source.parent / "other.py"})


def test_latest_write_accepts_alias_and_rejects_escape(alias_root):
    runner = load("run_vehicle_sumo_pilot")
    latest = alias_root / "latest.json"
    runner.write_latest(latest, "ok", alias_root)
    assert latest.read_text() == "ok"
    with pytest.raises(ValueError):
        runner.write_latest(alias_root / ".." / "outside.json", "bad", alias_root)
    assert not (alias_root / ".." / "outside.json").exists()


def test_evidence_accepts_alias_and_still_detects_source_changes(alias_root):
    analysis = load("analyze_vehicle_sumo_pilot")
    path, _, _ = fixture(alias_root, analysis)
    metrics, rows, _ = analysis.verify_evidence(alias_root, path)
    assert len(rows) == metrics["policy_request_rows"]
    (alias_root / analysis.SOURCES[0]).write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.verify_evidence(alias_root, path)


def test_native_absolute_source_alias_resolves_and_outside_is_rejected(alias_root):
    audit = load("audit_vehicle_stage2_readiness")
    source = alias_root / "scripts/example.py"
    source.parent.mkdir()
    source.write_text("example")
    assert audit.source_path(alias_root, str(source)) == source.resolve()
    outside = alias_root.resolve().parent / "scripts/other.py"
    with pytest.raises(ValueError, match="outside_project"):
        audit.source_path(alias_root, str(outside))
