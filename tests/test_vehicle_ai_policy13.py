from dataclasses import replace

import pytest
import test_vehicle_window_recovery as legacy
from run_vehicle_native_stage6 import packets

from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest


def engine(scorer):
    from coact_cps.vehicle.ai_policy13 import LearnedRiskPolicyEngine

    c = legacy.WindowRecoveryTests().config()
    keys = {s: bytes([i + 1]) * 32 for i, s in enumerate(SOURCES)}
    args = (
        keys,
        b"capabilitykey" * 4,
        {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")},
        ModelConfig(**c["model"]),
        c["thresholds"]["candidate"],
    )
    return (
        LearnedRiskPolicyEngine(
            *args, c["session_config"]["stage7_horizon"], scorer, 1.0, "frozen-model", window=8
        ),
        IntegrityPolicyEngine(*args, "horizon_dual", c["session_config"]["stage7_horizon"]),
        keys,
    )


def request(f):
    return VehicleRequest(
        "r" + str(f.frame),
        "n" + str(f.frame),
        "native-reader",
        "native-ego",
        "vehicle.telemetry.longitudinal",
        "read",
    )


def test_gate_only_denies_and_duplicate_time_never_rescores():
    calls = []
    gate, base, keys = engine(lambda x: calls.append(x.copy()) or 2.0)
    for f in legacy.WindowRecoveryTests().frames()[:90]:
        q = request(f)
        d = gate.evaluate(q, packets(f, 3 * f.frame + 1, keys), f.time_ms)
        b = base.evaluate(q, packets(f, 3 * f.frame + 1, keys), f.time_ms)
        assert d.outcome != "ALLOW" or b.outcome == "ALLOW"
        n = len(calls)
        gate.evaluate(q, packets(f, 3 * f.frame + 2, keys), f.time_ms)
        assert len(calls) == n
    assert calls and all(x.shape == (8, 3) for x in calls)


@pytest.mark.parametrize("failure", ["nonfinite", "exception"])
def test_failed_scorer_is_denied_including_duplicate(failure):
    def scorer(x):
        if failure == "exception":
            raise RuntimeError("unavailable")
        return float("nan")

    gate, _, keys = engine(scorer)
    for f in legacy.WindowRecoveryTests().frames()[:20]:
        d = gate.evaluate(request(f), packets(f, 3 * f.frame + 1, keys), f.time_ms)
    assert d.outcome == "DENY"
    assert gate.evaluate(request(f), packets(f, 3 * f.frame + 2, keys), f.time_ms).outcome == "DENY"


def test_identity_use_time_revocation_and_one_use_replay_preserved():
    for revoke in [False, True]:
        gate, _, keys = engine(lambda x: 0.0)
        for f in legacy.WindowRecoveryTests().frames()[:90]:
            d = gate.evaluate(request(f), packets(f, 4 * f.frame + 1, keys), f.time_ms)
        assert d.outcome == "ALLOW"
        q = request(f)
        token = gate.issue_capability(d)
        assert gate.consume_capability(
            token, q, packets(f, 4 * f.frame + 2, keys, revoked=revoke), f.time_ms
        ).authorized == (not revoke)
        assert not gate.consume_capability(
            token, q, packets(f, 4 * f.frame + 3, keys), f.time_ms
        ).authorized


def test_window_gap_resets_AI_warmup():
    calls = []
    gate, _, keys = engine(lambda x: calls.append(x.copy()) or 0.0)
    for f in legacy.WindowRecoveryTests().frames()[:90]:
        gate.evaluate(request(f), packets(f, 3 * f.frame + 1, keys), f.time_ms)
    t = f.time_ms + 400
    f = replace(
        f,
        frame=100,
        time_ms=t,
        interval_start_ms=t - 200,
        gnss_time_ms=t,
        imu_time_ms=t,
        wheel_time_ms=t,
    )
    n = len(calls)
    d = gate.evaluate(request(f), packets(f, 1000, keys), f.time_ms)
    assert (
        d.outcome != "ALLOW"
        and len(calls) == n
        and gate.diagnostics("native-ego")["ai_window_epochs"] == 1
    )
