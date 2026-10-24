import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "calibrated19_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_vehicle_calibrated_stage19.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_false_alarm_gain_cannot_hide_detection_loss():
    result = runner.screen(0.1, 0.08, 0.03, 0.04, 0.85, 0.9, 0.12, 0.01)
    assert result["passed"] is False
    assert result["each_route_case_detection_loss_at_most_5pp"] is False


def test_screen_is_development_only_even_when_passed():
    result = runner.screen(0.1, 0.08, 0.03, 0.04, 0.85, 0.9, 0.01, 0.01)
    assert result["passed"] is True
    assert result["eligible_for_test"] is False


def test_changed_archive_rejected_before_creating_output(tmp_path):
    source = tmp_path / "bad.zip"
    source.write_bytes(b"wrong saved data")
    with pytest.raises(ValueError, match="hash"):
        runner.run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
