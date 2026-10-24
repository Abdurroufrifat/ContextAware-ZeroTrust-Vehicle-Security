import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stage20_runner",
    Path(__file__).resolve().parents[1] / "scripts/audit_vehicle_loss_stage20.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_changed_stage19_manifest_is_rejected(tmp_path):
    (tmp_path / "file_hashes.json").write_text("{}")
    with pytest.raises(ValueError, match="binding"):
        runner.verify_results(tmp_path)


def test_wrong_archive_does_not_create_output(tmp_path):
    source = tmp_path / "bad.zip"
    source.write_bytes(b"invalid source")
    with pytest.raises(ValueError, match="hash"):
        runner.run(source, tmp_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_diagnostic_cannot_change_policy_or_unlock_tests():
    status = runner.status()
    assert status["eligible_for_test"] is False
    assert status["policy_changed"] is False
    assert status["model_training_performed"] is False
