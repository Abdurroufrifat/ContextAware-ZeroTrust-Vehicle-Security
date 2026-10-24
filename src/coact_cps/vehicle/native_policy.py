"""Signed longitudinal telemetry gate for a native, three-motion-source study."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .native_detectors import ModelConfig, Observer, Statistic, innovation_energy
from .policy import (
    SOURCE_DOMAIN,
    PolicyDecision,
    VehiclePolicyEngine,
    VehicleRequest,
    _integer,
    _mac,
    _number,
    _text,
    digest,
    strict_loads,
)

NATIVE_SOURCES = ("identity", "gnss", "imu", "wheel")
METHODS = ("auth", "chi", "window", "cusum", "dual_ungated", "gated_dual")


@dataclass
class _State:
    observer: Observer
    statistics: dict
    last_imu_key: str = ""
    last_imu_end: int = -1
    cache_key: str = ""
    cache_end: int = -1
    cache_result: tuple = ()
    quorum: int = 0
    updates: int = 0
    resets: int = 0
    z: float = 0.0
    statistic: float = 0.0
    alarm: bool = False


class NativePolicyEngine(VehiclePolicyEngine):
    def __init__(self, source_keys, capability_key, permissions, model_config, method, thresholds):
        super().__init__(source_keys, capability_key, permissions)
        if not isinstance(model_config, ModelConfig) or method not in METHODS:
            raise ValueError("typed native model and method required")
        if set(thresholds) != {"chi", "window", "cusum", "dual_chi", "dual_cusum"} or any(
            not _number(v, 1e-9, 1e12) for v in thresholds.values()
        ):
            raise ValueError("complete bounded frozen native thresholds required")
        self._native_model = model_config
        self._method = method
        self._thresholds = MappingProxyType(dict(thresholds))
        self._states = {}
        self._imu_watermarks = {}
        self._motion_watermarks = {}
        self._policy_digest = digest(
            dict(
                version="coact-native-longitudinal-v1",
                base_digest=self.policy_digest,
                sources=NATIVE_SOURCES,
                model=asdict(model_config),
                method=method,
                thresholds=dict(self._thresholds),
                window=10,
                cusum_bias=3,
                quorum=3,
                implementation_sha256=hashlib.sha256(
                    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
                detector_implementation_sha256=hashlib.sha256(
                    Path(__file__)
                    .with_name("native_detectors.py")
                    .read_bytes()
                    .replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
        )

    @property
    def method(self):
        return self._method

    def _new_state(self):
        stats = {}
        if self.method in ("chi", "window", "cusum"):
            stats[self.method] = Statistic(self.method, self._thresholds[self.method])
        elif self.method in ("gated_dual", "dual_ungated"):
            stats = {
                "chi": Statistic("chi", self._thresholds["dual_chi"]),
                "cusum": Statistic("cusum", self._thresholds["dual_cusum"]),
            }
        return _State(Observer(self._native_model), stats)

    def _validate_signals(self, source, s):
        if source == "identity":
            return super()._validate_signals(source, s)
        if source in ("gnss", "wheel"):
            if (
                type(s) is not dict
                or set(s) != {"speed_mps"}
                or not _number(s["speed_mps"], 0, 100)
            ):
                raise ValueError(f"{source}:invalid_native_speed_schema")
            return
        if source == "imu":
            keys = {"acceleration_mps2", "interval_start_ms", "interval_end_ms", "maximum_gap_ms"}
            if (
                type(s) is not dict
                or set(s) != keys
                or not _number(s["acceleration_mps2"], -30, 30)
                or not all(_integer(s[k]) for k in keys - {"acceleration_mps2"})
                or not 0 < s["interval_end_ms"] - s["interval_start_ms"] <= 2000
                or not 0 < s["maximum_gap_ms"] <= 50
            ):
                raise ValueError("imu:invalid_native_interval_schema")
            return
        raise ValueError("source_not_in_native_schema")

    def _read_source(self, source, request, wire, now):
        packet = strict_loads(wire)
        if set(packet) != {"body", "mac"} or type(packet["body"]) is not dict:
            raise ValueError(f"{source}:invalid_envelope")
        body, mac = packet["body"], packet["mac"]
        if (
            type(mac) is not str
            or len(mac) != 64
            or not hmac.compare_digest(mac, _mac(body, self._keys[source], SOURCE_DOMAIN))
        ):
            raise ValueError(f"{source}:authentication_failed")
        if set(body) != {"source", "vehicle_id", "frame_id", "timestamp_ms", "sequence", "signals"}:
            raise ValueError(f"{source}:invalid_source_schema")
        if (
            body["source"] != source
            or body["vehicle_id"] != request.vehicle_id
            or not _text(body["frame_id"])
            or not _integer(body["sequence"])
            or not _integer(body["timestamp_ms"])
        ):
            raise ValueError(f"{source}:identity_or_metadata_mismatch")
        if not 0 <= now - body["timestamp_ms"] <= self.config.source_ttl_ms:
            raise ValueError(f"{source}:stale_or_future")
        self._validate_signals(source, body["signals"])
        if body["sequence"] <= self._watermarks.get((request.vehicle_id, source), -1):
            raise ValueError(f"{source}:sequence_replay")
        if source in ("gnss", "wheel", "imu"):
            previous = self._motion_watermarks.get((request.vehicle_id, source))
            sample = (body["timestamp_ms"], digest(body["signals"]))
            if previous is not None and (
                sample[0] < previous[0] or (sample[0] == previous[0] and sample[1] != previous[1])
            ):
                raise ValueError(f"{source}:source_time_rollback_or_conflicting_sample")
        if source == "imu":
            s = body["signals"]
            if (
                s["interval_end_ms"] > now
                or not s["interval_start_ms"] <= body["timestamp_ms"] <= s["interval_end_ms"]
                or s["interval_end_ms"] - body["timestamp_ms"] > s["maximum_gap_ms"]
            ):
                raise ValueError("imu:interval_receipt_binding_failed")
            previous = self._imu_watermarks.get(request.vehicle_id)
            if previous is not None and (
                body["sequence"] < previous[0]
                or (body["sequence"] == previous[0] and digest(body) != previous[1])
            ):
                raise ValueError("imu:independent_sequence_replay_or_conflict")
        return body

    def _ingest_imu(self, request, frames, now):
        if type(frames) is not dict or "imu" not in frames:
            return
        body = self._read_source("imu", request, frames["imu"], now)
        self._imu_watermarks[request.vehicle_id] = (body["sequence"], digest(body))
        if request.vehicle_id not in self._states:
            self._states[request.vehicle_id] = self._new_state()
        state = self._states[request.vehicle_id]
        s = body["signals"]
        key = digest(dict(timestamp_ms=body["timestamp_ms"], signals=s))
        end = s["interval_end_ms"]
        if end == state.last_imu_end:
            if key != state.last_imu_key:
                raise ValueError("imu:conflicting_same_time_motion")
            return
        if end < state.last_imu_end:
            raise ValueError("imu:input_time_rollback")
        had_reference = state.observer.x is not None
        covered = state.observer.predict(s["acceleration_mps2"], s["interval_start_ms"], end)
        if had_reference and not covered:
            state.quorum = 0
            state.cache_key = ""
            state.resets += 1
            for statistic in state.statistics.values():
                statistic.reset()
        state.last_imu_key = key
        state.last_imu_end = end
        self._motion_watermarks[request.vehicle_id, "imu"] = (
            body["timestamp_ms"],
            digest(body["signals"]),
        )

    def _context(self, request, frames, now):
        if type(frames) is not dict or set(frames) != set(NATIVE_SOURCES):
            raise ValueError("native_source_set_missing_or_extra")
        bodies = {s: self._read_source(s, request, frames[s], now) for s in NATIVE_SOURCES}
        times = [b["timestamp_ms"] for b in bodies.values()]
        if len({b["frame_id"] for b in bodies.values()}) != 1:
            raise ValueError("frame_mismatch")
        if max(times) - min(times) > self.config.maximum_skew_ms:
            raise ValueError("source_time_skew")
        for source, body in bodies.items():
            self._watermarks[request.vehicle_id, source] = body["sequence"]
            if source in ("gnss", "wheel", "imu"):
                self._motion_watermarks[request.vehicle_id, source] = (
                    body["timestamp_ms"],
                    digest(body["signals"]),
                )
        return bodies

    def _motion(self, vehicle, bodies):
        state = self._states[vehicle]
        key = digest(
            {
                s: dict(timestamp_ms=bodies[s]["timestamp_ms"], signals=bodies[s]["signals"])
                for s in ("gnss", "imu", "wheel")
            }
        )
        end = bodies["imu"]["signals"]["interval_end_ms"]
        if end == state.cache_end:
            if key != state.cache_key:
                raise ValueError("conflicting_same_time_motion")
            return state.cache_result
        if end < state.cache_end:
            raise ValueError("motion_time_rollback")
        g = float(bodies["gnss"]["signals"]["speed_mps"])
        w = float(bodies["wheel"]["signals"]["speed_mps"])
        if state.observer.x is None:
            state.observer.initialize(g, w, end)
        residual, covariance = state.observer.innovation(g, w)
        z = innovation_energy(residual, covariance)
        result = [s.step(z) for s in state.statistics.values()]
        alarm = any(r.alarm for r in result)
        statistic = max((r.statistic for r in result), default=z)
        if self.method != "gated_dual" or not alarm:
            state.observer.update(g, w)
            state.updates += 1
        state.quorum = min(3, state.quorum + 1)
        state.z = z
        state.statistic = statistic
        state.alarm = alarm
        output = (alarm, state.quorum < 3, z, statistic)
        state.cache_key = key
        state.cache_end = end
        state.cache_result = output
        return output

    def diagnostics(self, vehicle):
        with self._lock:
            s = self._states.get(vehicle)
            if s is None:
                return dict(measurement_updates=0, resets=0, z=0.0, statistic=0.0, alarm=False)
            return dict(
                measurement_updates=s.updates,
                resets=s.resets,
                z=s.z,
                statistic=s.statistic,
                alarm=s.alarm,
            )

    def evaluate(self, request, frames, now_ms):
        if not isinstance(request, VehicleRequest) or not _integer(now_ms):
            raise ValueError("typed_request_and_integer_clock_required")
        with self._lock:

            def decision(outcome, risk, reasons, context=""):
                if outcome == "DENY" and "native_motion_statistic_alarm" not in reasons:
                    state = self._states.get(request.vehicle_id)
                    if state is not None:
                        state.quorum = 0
                        if state.cache_result:
                            alarm, _, z, statistic = state.cache_result
                            state.cache_result = (alarm, True, z, statistic)
                return self._remember(
                    PolicyDecision(
                        request,
                        outcome,
                        "stream" if outcome == "ALLOW" else "none",
                        risk,
                        tuple(reasons),
                        (("native_gate_failure", risk),),
                        context,
                        self.policy_digest,
                        now_ms,
                        now_ms + self.config.capability_ttl_ms if context else now_ms,
                    )
                )

            if now_ms < self._clock_watermark:
                return decision("DENY", 1.0, ["clock_rollback"])
            self._clock_watermark = now_ms
            try:
                self._ingest_imu(request, frames, now_ms)
                bodies = self._context(request, frames, now_ms)
                identity = bodies["identity"]["signals"]
                reasons = []
                for failed, name in [
                    (identity["subject_id"] != request.subject_id, "subject_mismatch"),
                    (not identity["credential_valid"], "credential_invalid"),
                    (identity["revoked"], "credential_revoked"),
                    (not identity["attestation_ok"], "attestation_failed"),
                    (
                        (
                            request.subject_id,
                            request.vehicle_id,
                            request.resource,
                            request.operation,
                        )
                        not in self._permissions,
                        "scope_not_authorized",
                    ),
                    (
                        (request.resource, request.operation)
                        != ("vehicle.telemetry.longitudinal", "read"),
                        "native_scope_not_authorized",
                    ),
                ]:
                    if failed:
                        reasons.append(name)
                if reasons:
                    return decision("DENY", 1.0, reasons, digest(bodies))
                alarm, cold, z, statistic = self._motion(request.vehicle_id, bodies)
                context = digest(
                    dict(sources=bodies, preupdate_z=z, statistic=statistic, alarm=alarm, cold=cold)
                )
                if alarm:
                    return decision("DENY", 1.0, ["native_motion_statistic_alarm"], context)
                if cold:
                    return decision(
                        "CHALLENGE", 0.0, ["native_startup_or_recovery_quorum"], context
                    )
                reason = (
                    "authenticated_source_contract_only"
                    if self.method == "auth"
                    else "native_model_consistency_satisfied"
                )
                return decision("ALLOW", 0.0, [reason], context)
            except (
                ValueError,
                TypeError,
                KeyError,
                OverflowError,
                RecursionError,
                np.linalg.LinAlgError,
            ) as exc:
                return decision("DENY", 1.0, [str(exc)])
