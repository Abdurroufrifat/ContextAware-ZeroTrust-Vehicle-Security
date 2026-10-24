import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "imu_audit18_runner",
    Path(__file__).resolve().parents[1] / "scripts/audit_vehicle_imu_stage18.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_attack_labels_cannot_enter_diagnostics():
    with pytest.raises(ValueError, match="normal"):
        runner.require_clean({"labels": np.array([False, True])})


def test_unknown_archive_rejected_before_writing(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"changed source archive")
    with pytest.raises(ValueError, match="hash"):
        runner.run(p, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_no_final_test_or_production_eligibility():
    status = runner.status()
    assert status["eligible_for_test"] is False
    assert status["policy_changed"] is False
    assert runner.ROLES == {"civic13_07": "fit", "civic13_08": "fit", "civic13_09": "check"}
