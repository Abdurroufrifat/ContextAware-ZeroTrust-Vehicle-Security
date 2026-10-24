"""Development temporal guard; physical truth and safe enrollment are assumptions."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass
from pathlib import Path

from .policy import (
    SOURCE_DOMAIN,
    SOURCES,
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


@dataclass(frozen=True)
class TemporalConfig:
    enable_temporal: bool = True
    allow_degraded_v2x: bool = True
    residual_tolerance_mps: float = 0.4
    residual_scale_mps: float = 2.0
    acceleration_bias_allowance_mps2: float = 0.02
    maximum_imu_gap_ms: int = 1000
    maximum_reference_age_ms: int = 60000

    def __post_init__(self):
        if type(self.enable_temporal) is not bool or type(self.allow_degraded_v2x) is not bool:
            raise ValueError("typed_feature_flags_required")
        for value in (self.residual_tolerance_mps, self.acceleration_bias_allowance_mps2):
            if not _number(value, 0, 10):
                raise ValueError("finite_nonnegative_tolerances_required")
        if (
            not _number(self.residual_scale_mps, 1e-6, 100)
            or not _integer(self.maximum_imu_gap_ms)
            or not 0 < self.maximum_imu_gap_ms <= 10000
        ):
            raise ValueError("positive_scale_and_imu_gap_required")
        if (
            not _integer(self.maximum_reference_age_ms)
            or not 0 < self.maximum_reference_age_ms <= 3600000
        ):
            raise ValueError("positive_bounded_reference_age_required")


class TemporalVehiclePolicyEngine(VehiclePolicyEngine):
    def __init__(self, source_keys, capability_key, permissions, config=None, temporal_config=None):
        super().__init__(source_keys, capability_key, permissions, config)
        self._temporal_config = temporal_config or TemporalConfig()
        if not isinstance(self._temporal_config, TemporalConfig):
            raise ValueError("typed_temporal_config_required")
        self._policy_digest = digest(
            dict(
                version="coact-vehicle-temporal-development-v1",
                base_policy_digest=self.policy_digest,
                temporal_config=asdict(self.temporal_config),
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
        )
        self._motion = {}
        self._imu_sequences = {}

    @property
    def temporal_config(self):
        return self._temporal_config

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
        if source == "imu" and self.temporal_config.enable_temporal:
            previous = self._imu_sequences.get(request.vehicle_id)
            if previous is not None and (
                body["sequence"] < previous[0]
                or (body["sequence"] == previous[0] and digest(body) != previous[1])
            ):
                raise ValueError("imu:independent_sequence_replay_or_conflict")
        return body

    def _context(self, request, frames, now):
        if self.temporal_config.enable_temporal:
            self._read_source("imu", request, frames["imu"], now)
        if type(frames) is dict and set(frames) == set(SOURCES):
            return super()._context(request, frames, now)
        if (
            not self.temporal_config.allow_degraded_v2x
            or type(frames) is not dict
            or set(frames) != set(SOURCES) - {"v2x"}
        ):
            raise ValueError("required_source_set_missing_or_extra")
        bodies = {s: self._read_source(s, request, frames[s], now) for s in SOURCES if s != "v2x"}
        times = [b["timestamp_ms"] for b in bodies.values()]
        if len({b["frame_id"] for b in bodies.values()}) != 1:
            raise ValueError("frame_mismatch")
        if max(times) - min(times) > self.config.maximum_skew_ms:
            raise ValueError("source_time_skew")
        for source, body in bodies.items():
            self._watermarks[request.vehicle_id, source] = body["sequence"]
        return bodies

    def _advance_imu(self, request, frames, now):
        if not self.temporal_config.enable_temporal:
            return
        try:
            body = self._read_source("imu", request, frames["imu"], now)
            if body["sequence"] <= self._imu_sequences.get(request.vehicle_id, (-1, ""))[0]:
                return
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            return
        self._imu_sequences[request.vehicle_id] = (body["sequence"], digest(body))
        state = self._motion.get(request.vehicle_id)
        if state is None:
            return
        dt = body["timestamp_ms"] - state["last_ms"]
        if dt < 0 or dt > self.temporal_config.maximum_imu_gap_ms:
            state["coverage_lost"] = True
        elif not state["coverage_lost"]:
            state["predicted_speed_mps"] = max(
                0.0,
                state["predicted_speed_mps"] + body["signals"]["acceleration_mps2"] * dt / 1000.0,
            )
        state["last_ms"] = max(state["last_ms"], body["timestamp_ms"])

    def _temporal_risk(self, request, bodies, base_risk, now_ms):
        if not self.temporal_config.enable_temporal:
            return 0.0, False, {}
        state = self._motion.get(request.vehicle_id)
        if state is None:
            if base_risk >= self.config.allow_threshold:
                return 1.0, False, {"state": "not_enrolled"}
            now = bodies["imu"]["timestamp_ms"]
            state = dict(
                anchor_ms=now,
                last_ms=now,
                predicted_speed_mps=float(bodies["wheel"]["signals"]["speed_mps"]),
                coverage_lost=False,
            )
            self._motion[request.vehicle_id] = state
            return 0.0, True, dict(state, state="enrollment_requires_next_context")
        if state["coverage_lost"]:
            return 1.0, False, dict(state, state="imu_coverage_lost")
        if now_ms - state["anchor_ms"] > self.temporal_config.maximum_reference_age_ms:
            return 1.0, False, dict(state, state="reference_expired")
        elapsed = max(0.0, (state["last_ms"] - state["anchor_ms"]) / 1000.0)
        tolerance = (
            self.temporal_config.residual_tolerance_mps
            + self.temporal_config.acceleration_bias_allowance_mps2 * elapsed
        )
        residual = max(
            abs(bodies[s]["signals"]["speed_mps"] - state["predicted_speed_mps"])
            for s in ("gnss", "wheel")
        )
        risk = min(1.0, max(0.0, residual - tolerance) / self.temporal_config.residual_scale_mps)
        return (
            risk,
            False,
            dict(state, residual_mps=residual, tolerance_mps=tolerance, state="tracking"),
        )

    def evaluate(self, request, frames, now_ms):
        if not isinstance(request, VehicleRequest) or not _integer(now_ms):
            raise ValueError("typed_request_and_integer_clock_required")
        with self._lock:

            def decision(outcome, mode, risk, reasons, components=(), context=""):
                return self._remember(
                    PolicyDecision(
                        request,
                        outcome,
                        mode,
                        risk,
                        tuple(reasons),
                        tuple(components),
                        context,
                        self.policy_digest,
                        now_ms,
                        now_ms + self.config.capability_ttl_ms if context else now_ms,
                    )
                )

            if now_ms < self._clock_watermark:
                return decision("DENY", "none", 1.0, ["clock_rollback"])
            self._clock_watermark = now_ms
            self._advance_imu(request, frames, now_ms)
            try:
                bodies = self._context(request, frames, now_ms)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                return decision("DENY", "none", 1.0, [str(exc)])
            s = {name: body["signals"] for name, body in bodies.items()}
            cfg = self.config
            components = [
                (
                    "speed_disagreement",
                    min(
                        1.0,
                        abs(s["gnss"]["speed_mps"] - s["wheel"]["speed_mps"]) / cfg.speed_scale_mps,
                    ),
                ),
                (
                    "acceleration_disagreement",
                    min(
                        1.0,
                        abs(s["imu"]["acceleration_mps2"] - s["wheel"]["acceleration_mps2"])
                        / cfg.acceleration_scale_mps2,
                    ),
                ),
                (
                    "yaw_disagreement",
                    min(
                        1.0,
                        abs(s["imu"]["yaw_rate_rps"] - s["wheel"]["yaw_rate_rps"])
                        / cfg.yaw_scale_rps,
                    ),
                ),
                ("can_anomaly", float(s["can"]["anomaly_score"])),
            ]
            missing = "v2x" not in s
            if not missing:
                components.extend(
                    [
                        ("v2x_inconsistency", float(s["v2x"]["inconsistency_score"])),
                        ("uncertainty", float(s["v2x"]["uncertainty"])),
                    ]
                )
            risk = max(v for _, v in components)
            reasons = []
            identity = s["identity"]
            for failed, name in [
                (identity["subject_id"] != request.subject_id, "subject_mismatch"),
                (not identity["credential_valid"], "credential_invalid"),
                (identity["revoked"], "credential_revoked"),
                (not identity["attestation_ok"], "attestation_failed"),
                (
                    (request.subject_id, request.vehicle_id, request.resource, request.operation)
                    not in self._permissions,
                    "scope_not_authorized",
                ),
                (request.resource == "vehicle.actuator", "actuator_control_out_of_scope"),
                (
                    missing
                    and (request.resource != "vehicle.telemetry" or request.operation != "read"),
                    "degraded_scope_not_authorized",
                ),
            ]:
                if failed:
                    reasons.append(name)
            motion = {}
            cold = False
            if not reasons:
                temporal, cold, motion = self._temporal_risk(request, bodies, risk, now_ms)
                components.append(("temporal_speed_inconsistency", temporal))
                risk = max(risk, temporal)
            mode = "none"
            if reasons:
                outcome = "DENY"
            elif risk >= cfg.deny_threshold:
                outcome = "DENY"
                reasons.append("high_context_risk")
            elif cold:
                outcome = "CHALLENGE"
                reasons.append("temporal_enrollment_requires_next_context")
            elif missing and risk < cfg.allow_threshold:
                outcome, mode = "RESTRICT", "snapshot"
                reasons.append("v2x_unavailable_snapshot_only")
            elif missing:
                outcome = "CHALLENGE"
                reasons.append("remaining_risk_blocks_degraded_access")
            elif risk < cfg.allow_threshold:
                outcome, mode = "ALLOW", "stream" if request.operation == "read" else "single"
                reasons.append("context_contract_satisfied")
            elif request.operation == "read" and risk < cfg.restrict_threshold:
                outcome, mode = "RESTRICT", "snapshot"
                reasons.append("diagnostic_snapshot_only")
            else:
                outcome = "CHALLENGE"
                reasons.append("fresh_corroborating_evidence_required")
            if missing and "v2x_unavailable_snapshot_only" not in reasons:
                reasons.append("v2x_unavailable")
            if motion.get("state") == "imu_coverage_lost":
                reasons.append("temporal_imu_coverage_lost")
            if motion.get("state") == "reference_expired":
                reasons.append("temporal_reference_expired")
            reasons.extend(
                "elevated_" + name for name, value in components if value >= cfg.allow_threshold
            )
            context = digest(
                dict(authenticated_sources=bodies, motion_reference=motion, v2x_missing=missing)
            )
            return decision(outcome, mode, risk, reasons, components, context)
