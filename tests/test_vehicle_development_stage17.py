import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "stage17_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_vehicle_development_stage17.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_tampered_archive_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not the frozen source")
    with pytest.raises(ValueError, match="hash"):
        runner.read_source(path)


def test_no_final_test_eligibility():
    status = runner.status_receipt()
    assert status["eligible_for_test"] is False
    assert status["fresh_validation_required"] is True
    assert status["status"] == "DEVELOPMENT_ONLY_NOT_PROMOTED"
    assert runner.EVALUATION_ROUTES == ("civic13_10", "civic13_11", "civic13_12")


def test_active_endpoint_and_grant_denominators():
    c = {
        "labels": np.array([False, True, True, True]),
        "positions": np.array([1, 3]),
        "base_grant": np.ones(4, dtype=bool),
    }
    result = runner.metrics(c, np.array([True, False]))
    assert result["active_endpoint_tpr"] == 0.5
    assert result["active_grant_fraction"] == 1 / 3
    assert result["grant_fraction_all_epochs"] == 0.25
    assert result["active_endpoint_count"] == 2


def test_all_normal_tpr_is_undefined():
    c = {
        "labels": np.zeros(4, dtype=bool),
        "positions": np.array([1, 3]),
        "base_grant": np.ones(4, dtype=bool),
    }
    result = runner.metrics(c, np.array([True, False]))
    assert result["active_endpoint_tpr"] is None
    assert result["normal_endpoint_fpr"] == 0.5
