"""Independent saved-output checks; no policy execution or fitting."""

import math
from collections import defaultdict, deque


class AccessAudit:
    def __init__(self):
        self.pending = {}
        self.rows = self.revoked_rows = self.postissue_rows = self.releases = 0

    @property
    def passed(self):
        return self.rows > 0

    def observe(self, row):
        key = row["trip"], row["case"], row["method"]
        now = int(row["time_ms"])
        issue, use = int(row["issued_motion_quorum"]), int(row["consumed_motion_quorum"])
        assert 0 <= issue <= 3 and 0 <= use <= 3
        allowed = row["issued_outcome"] == "ALLOW"
        authorized = row["capability_authorized"] == "True"
        granted = row["final_grant"] == "True"
        assert not authorized or allowed
        assert not granted or authorized
        if allowed:
            assert issue >= 3
        if authorized:
            assert use >= 3
        if row["method"] == "protected_recovery":
            assert int(row["primary_quorum"]) == issue
        active = row["perturbed"] == "True"
        if active and row["case"] == "identity_revoked":
            assert row["issued_outcome"] == "DENY" and not authorized and not granted
            assert issue == use == 0
            self.revoked_rows += 1
        if active and row["case"] == "postissue_revoked":
            assert not authorized and not granted
            if allowed:
                assert use == 0
            self.postissue_rows += 1
        if key not in self.pending:
            self.pending[key] = 3, -1
        if key in self.pending:
            remaining, previous = self.pending[key]
            assert now >= previous
            if row["motion_scored"] == "True" and now > previous:
                remaining = max(0, remaining - 1)
            if remaining:
                assert not allowed and not authorized and not granted
            self.pending[key] = remaining, now
        if row["recovery_release"] == "True":
            assert row["issued_outcome"] == "CHALLENGE" and not authorized and not granted
            assert issue == use == 0
            self.releases += 1
        if use == 0:
            self.pending[key] = 3, now
        self.rows += 1


def audit_protected_traces(rows, contexts, frozen):
    """Reconstruct accepted IMU integrals, lagged quiet history and loss latches."""
    assert rows.keys() == contexts.keys()
    lag = frozen["protected_parameters"]["lag_ms"]
    maximum_age = frozen["protected_parameters"]["maximum_age_ms"]
    assert lag == 20000 and maximum_age == 90000
    anchor_config = frozen["session_config"]["anchor"]
    bound = max(1.0, anchor_config["scale"] * anchor_config["threshold"])
    offset = frozen["model"]["gnss_offset"]
    grouped = defaultdict(list)
    for key in contexts:
        grouped[key[:2]].append(key[2])
    checked = lost_rows = expired_rows = protected_releases = 0
    for group, times in grouped.items():
        times.sort()
        begin = times[0] + (times[-1] - times[0]) / 3
        history = deque(maxlen=512)
        active = False
        anchor = None
        previous_imu = None
        integral = 0.0
        gaps = 0
        freeze_gaps = 0
        start = -1
        for now in times:
            key = group + (now,)
            row, context = rows[key], contexts[key]["observed"]
            present = not (
                (group[1] == "imu_dropout" and begin <= now < begin + 2000)
                or (group[1] == "gap_reenrollment" and begin + 500 <= now < begin + 1500)
            )
            if present:
                if previous_imu is not None and context["interval_start_ms"] != previous_imu:
                    gaps += 1
                    history.clear()
                integral += context["imu_accel"] * (now - context["interval_start_ms"]) / 1000
                previous_imu = now
            assert math.isclose(
                float(row["protected_error_bound"]), bound, rel_tol=0, abs_tol=1e-12
            )
            scored = row["motion_scored"] == "True"
            if row["quarantined"] == "True" and not active:
                assert scored
                active = True
                anchor = next(
                    (snapshot for snapshot in reversed(history) if snapshot[0] <= now - lag), None
                )
                start, freeze_gaps = now, gaps
            expected_present = active and anchor is not None
            assert (row["protected_present"] == "True") == expected_present
            if scored:
                lost = active and gaps > freeze_gaps
                assert (row["protected_lost"] == "True") == lost
                if expected_present:
                    time, speed, bias, old_integral = anchor
                    age = now - time
                    prediction = speed + integral - old_integral - bias * age / 1000
                    eg = abs(context["gnss_speed"] - offset - prediction)
                    ew = abs(context["wheel_speed"] - prediction)
                    expired = age > maximum_age
                    eligible = not lost and not expired and max(eg, ew) <= bound + 1e-12
                    assert int(row["protected_anchor_time_ms"]) == time
                    assert int(row["protected_quarantine_start_ms"]) == start
                    assert int(row["protected_age_ms"]) == age
                    assert float(row["protected_snapshot_speed"]) == speed
                    assert float(row["protected_snapshot_bias"]) == bias
                    assert (row["protected_expired"] == "True") == expired
                    assert (row["protected_guard_pass"] == "True") == eligible
                    assert math.isclose(
                        float(row["protected_prediction"]), prediction, rel_tol=1e-10, abs_tol=1e-8
                    )
                    assert abs(float(row["protected_gnss_error"]) - eg) < 1e-8
                    assert abs(float(row["protected_wheel_error"]) - ew) < 1e-8
                    checked += 1
                    lost_rows += lost
                    expired_rows += expired
                else:
                    assert row["protected_guard_pass"] == row["protected_expired"] == "False"
            released = row["recovery_release"] == "True"
            if row["protected_window_release"] == "True":
                assert released and expected_present and scored
                assert row["protected_guard_pass"] == "True"
                protected_releases += 1
            if released:
                active = False
                anchor = None
                history.clear()
            quiet = (
                scored
                and row["quarantined"] == row["primary_alarm"] == row["horizon_alarm"] == "False"
                and row["horizon_ready"] == "True"
                and int(row["primary_quorum"]) >= 3
                and not released
            )
            if quiet:
                assert not active
                snapshot = (
                    now,
                    float(row["primary_estimated_speed"]),
                    float(row["primary_estimated_bias"]),
                    integral,
                )
                history.append(snapshot)
    return dict(
        snapshot_rows=checked,
        lost_rows=lost_rows,
        expired_rows=expired_rows,
        protected_releases=protected_releases,
        continuity_bound_expiry_and_latest_snapshot_verified=True,
    )


def check_native_context(saved, expected):
    assert saved.keys() == expected.keys()
    for key, value in expected.items():
        if type(value) is float:
            assert type(saved[key]) in (float, int) and math.isfinite(saved[key])
            assert math.isclose(saved[key], value, rel_tol=1e-10, abs_tol=1e-12)
        else:
            assert type(saved[key]) is type(value) and saved[key] == value
