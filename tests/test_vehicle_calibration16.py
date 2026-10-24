import numpy as np
import pytest

from coact_cps.vehicle import calibration16 as c


def test_threshold_uses_each_clean_route_and_never_lowers_original_threshold():
    routes = {"a": np.arange(100.0), "b": np.arange(100.0) * 2, "c": np.arange(100.0) / 2}
    result = c.fit_threshold(routes, 100.0)
    assert result["threshold"] == np.quantile(routes["b"], 0.95)
    assert result["route_windows"] == {"a": 100, "b": 100, "c": 100}
    assert c.fit_threshold(routes, 1000.0)["threshold"] == 1000.0


@pytest.mark.parametrize("value", [[], [1.0] * 31, [float("nan")] * 40, [[1.0]] * 40])
def test_invalid_calibration_scores_fail_closed(value):
    with pytest.raises(ValueError):
        c.fit_threshold({"a": np.zeros(40), "b": np.zeros(40), "c": value}, 1.0)


def test_requires_three_routes_and_finite_baseline():
    with pytest.raises(ValueError):
        c.fit_threshold({"a": np.zeros(40)}, 1.0)
    with pytest.raises(ValueError):
        c.fit_threshold({x: np.zeros(40) for x in "abc"}, float("nan"))


def test_validation_rejects_normal_gain_that_loses_attack_detection():
    def rows(drop):
        return [
            dict(
                trip=t,
                seed=s,
                case=case,
                baseline=0.2 if case == "normal" else 0.9,
                candidate=0.02 if case == "normal" else 0.9 - drop,
                base_grant=0.98,
                baseline_grant=0.7,
                candidate_grant=0.8,
            )
            for t in ["v1", "v2"]
            for s in [7, 19, 43]
            for case in ["normal", "attack"]
        ]

    assert c.validation_gate(rows(0.01), ["normal", "attack"])["passed"]
    result = c.validation_gate(rows(0.06), ["normal", "attack"])
    assert not result["passed"]
    assert result["attack_tpr_drop"]["attack"] > 0.05
    with pytest.raises(ValueError):
        c.validation_gate(rows(0.01)[:-1], ["normal", "attack"])


def test_reservations_require_ten_new_dates_and_fixed_split():
    trips = [
        dict(
            id=str(i),
            day=f"2018-12-{i + 1:02d}",
            route_id=f"route{i}",
            phase="calibration" if i < 3 else "validation" if i < 5 else "test",
        )
        for i in range(10)
    ]
    c.check_reservations(trips, [])
    with pytest.raises(ValueError):
        c.check_reservations(trips, [trips[0]])
    trips[-1]["day"] = trips[0]["day"]
    with pytest.raises(ValueError):
        c.check_reservations(trips, [])


def test_failed_validation_and_modified_receipt_cannot_open_test(tmp_path):
    from run_vehicle_calibration_stage16 import OUT, authorize_phase, hash_outputs, sha, write

    write(tmp_path / OUT / "protocol.json", {})
    ph = sha((tmp_path / OUT / "protocol.json").read_bytes())
    for phase in ["calibration", "validation"]:
        folder = tmp_path / OUT / phase
        if phase == "calibration":
            write(folder / "threshold_lock.json", {})
        summary = dict(protocol_sha256=ph, raw_sha256={}, gate={"passed": False})
        if phase == "validation":
            summary["threshold_lock_sha256"] = sha(
                (tmp_path / OUT / "calibration/threshold_lock.json").read_bytes()
            )
        write(folder / "summary.json", summary)
        write(folder / "file_hashes.json", hash_outputs(folder))
        write(
            folder / "independent_audit.json",
            dict(
                status="PASS", phase_hashes_sha256=sha((folder / "file_hashes.json").read_bytes())
            ),
        )
    with pytest.raises(ValueError, match="remain_locked"):
        authorize_phase(tmp_path, "test")
    summary["gate"]["passed"] = True
    write(tmp_path / OUT / "validation/summary.json", summary)
    with pytest.raises(ValueError, match="output_changed"):
        authorize_phase(tmp_path, "test")


def test_source_binding_detects_changes_before_model_or_data_access(tmp_path):
    from run_vehicle_calibration_stage16 import OUT, verify, write

    write(tmp_path / OUT / "protocol.json", {"source_sha256": {"critical.py": "0" * 64}})
    (tmp_path / "critical.py").write_text("changed")
    with pytest.raises(ValueError, match="frozen_source_changed"):
        verify(tmp_path)


def test_unknown_phase_rejected_before_accessing_data(tmp_path):
    from run_vehicle_calibration_stage16 import authorize_phase

    with pytest.raises(ValueError, match="invalid_phase"):
        authorize_phase(tmp_path, "training")


def test_validation_cannot_average_away_one_bad_route():
    rows = [
        dict(
            trip=t,
            seed=s,
            case=case,
            baseline=0.2 if case == "normal" else 0.9,
            candidate=(0.08 if t == "v1" else 0.0) if case == "normal" else 0.9,
            base_grant=0.98,
            baseline_grant=0.7,
            candidate_grant=0.8,
        )
        for t in ["v1", "v2"]
        for s in [7, 19, 43]
        for case in ["normal", "attack"]
    ]
    assert not c.validation_gate(rows, ["normal", "attack"])["passed"]
