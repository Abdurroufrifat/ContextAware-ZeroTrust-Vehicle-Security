"""Authenticated context and deterministic, per-request vehicle access policy.

All thresholds are illustrative research defaults. A signed collector record is
not proof that a physical sensor is truthful. State is process-local and must be
persisted before deployment across restarts or multiple enforcement instances.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

SOURCES = ("identity", "gnss", "imu", "wheel", "can", "v2x")
SOURCE_DOMAIN = b"COACT-VEHICLE-SOURCE-V1\0"
CAPABILITY_DOMAIN = b"COACT-VEHICLE-CAPABILITY-V1\0"
MAX_WIRE_BYTES = 16384


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite_number")


def strict_loads(wire: str) -> dict:
    if type(wire) is not str or len(wire.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("invalid_wire_size_or_type")
    try:
        value = json.loads(wire, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid_json") from exc
    if type(value) is not dict:
        raise ValueError("object_required")
    # Reject overflow such as 1e999 as well as explicit NaN/Infinity.
    canonical_json(value)
    return value


def _text(value: Any) -> bool:
    return type(value) is str and 0 < len(value) <= 256 and all(32 <= ord(c) <= 126 for c in value)


def _integer(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 2**53 - 1


def _number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _mac(body: dict, key: bytes, domain: bytes) -> str:
    return hmac.new(key, domain + canonical_json(body).encode("ascii"), hashlib.sha256).hexdigest()


def seal_source(
    source: str,
    vehicle_id: str,
    frame_id: str,
    timestamp_ms: int,
    sequence: int,
    signals: dict,
    key: bytes,
) -> str:
    """Called by a trusted post-ingestion collector, never by untrusted telemetry.

    The policy independently validates all fields after signature verification.
    This function is also useful for controlled signed-source fault experiments.
    """
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("collector_key_requires_32_bytes")
    body = dict(
        source=source,
        vehicle_id=vehicle_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        sequence=sequence,
        signals=signals,
    )
    return canonical_json({"body": body, "mac": _mac(body, key, SOURCE_DOMAIN)})


@dataclass(frozen=True)
class VehicleRequest:
    request_id: str
    nonce: str
    subject_id: str
    vehicle_id: str
    resource: str
    operation: str

    def __post_init__(self):
        if not all(_text(v) for v in asdict(self).values()):
            raise ValueError("request_fields_require_nonempty_printable_strings")


@dataclass(frozen=True)
class PolicyConfig:
    source_ttl_ms: int = 500
    maximum_skew_ms: int = 50
    capability_ttl_ms: int = 250
    allow_threshold: float = 0.25
    restrict_threshold: float = 0.50
    deny_threshold: float = 0.75
    speed_scale_mps: float = 5.0
    acceleration_scale_mps2: float = 3.0
    yaw_scale_rps: float = 0.5

    def __post_init__(self):
        for value in (self.source_ttl_ms, self.maximum_skew_ms, self.capability_ttl_ms):
            if not _integer(value) or value == 0:
                raise ValueError("positive_integer_time_limits_required")
        limits = (self.allow_threshold, self.restrict_threshold, self.deny_threshold)
        if (
            not all(_number(v, 0, 1) for v in limits)
            or not 0 < limits[0] < limits[1] < limits[2] <= 1
        ):
            raise ValueError("ordered_risk_thresholds_required")
        for value in (self.speed_scale_mps, self.acceleration_scale_mps2, self.yaw_scale_rps):
            if not _number(value, 1e-9, 1e6):
                raise ValueError("positive_finite_scales_required")


@dataclass(frozen=True)
class PolicyDecision:
    request: VehicleRequest
    outcome: str
    mode: str
    risk: float
    reasons: tuple[str, ...]
    components: tuple[tuple[str, float], ...]
    context_digest: str
    policy_digest: str
    evaluated_ms: int
    expires_ms: int


@dataclass(frozen=True)
class AuthorizationResult:
    authorized: bool
    mode: str
    reasons: tuple[str, ...]
    current_decision: PolicyDecision | None = None
    expires_ms: int = 0


class VehiclePolicyEngine:
    """Policy engine plus an in-process reference enforcement point.

    Permissions are explicit (subject, vehicle, resource, operation) tuples.
    The trusted collector keys authenticate vehicle identity on every record.
    """

    def __init__(
        self,
        source_keys: dict[str, bytes],
        capability_key: bytes,
        permissions: set[tuple[str, str, str, str]],
        config: PolicyConfig | None = None,
    ):
        keys = dict(source_keys)
        if (
            set(keys) != set(SOURCES)
            or any(type(k) is not bytes or len(k) < 32 for k in keys.values())
            or len(set(keys.values())) != len(SOURCES)
        ):
            raise ValueError("six_distinct_collector_keys_required")
        if (
            type(capability_key) is not bytes
            or len(capability_key) < 32
            or capability_key in keys.values()
        ):
            raise ValueError("distinct_capability_key_required")
        if any(
            type(p) is not tuple or len(p) != 4 or not all(_text(v) for v in p) for p in permissions
        ):
            raise ValueError("explicit_per_vehicle_permission_tuples_required")
        self._keys = keys
        self._capability_key = capability_key
        self._permissions = frozenset(permissions)
        self._config = config or PolicyConfig()
        self._policy_digest = digest(
            {
                "version": "coact-vehicle-stage1-v1",
                "config": asdict(self.config),
                "permissions": sorted(self._permissions),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        self._watermarks: dict[tuple[str, str], int] = {}
        self._decisions: OrderedDict[str, PolicyDecision] = OrderedDict()
        self._used_nonces: dict[tuple[str, str], int] = {}
        self._clock_watermark = -1
        self._lock = RLock()

    @property
    def config(self) -> PolicyConfig:
        return self._config

    @property
    def policy_digest(self) -> str:
        return self._policy_digest

    @staticmethod
    def _validate_signals(source: str, signals: dict):
        schemas = {
            "gnss": {"speed_mps": (0, 100)},
            "imu": {"acceleration_mps2": (-30, 30), "yaw_rate_rps": (-10, 10)},
            "wheel": {
                "speed_mps": (0, 100),
                "acceleration_mps2": (-30, 30),
                "yaw_rate_rps": (-10, 10),
            },
            "can": {"anomaly_score": (0, 1)},
            "v2x": {"inconsistency_score": (0, 1), "uncertainty": (0, 1)},
        }
        if type(signals) is not dict:
            raise ValueError("invalid_signals")
        if source == "identity":
            fields = {"subject_id", "credential_valid", "revoked", "attestation_ok"}
            if (
                set(signals) != fields
                or not _text(signals["subject_id"])
                or any(type(signals[f]) is not bool for f in fields - {"subject_id"})
            ):
                raise ValueError("invalid_identity_schema")
        elif set(signals) != set(schemas[source]) or any(
            not _number(signals[k], *bounds) for k, bounds in schemas[source].items()
        ):
            raise ValueError("invalid_numeric_schema")

    def _context(self, request: VehicleRequest, frames: dict[str, str], now_ms: int):
        if type(frames) is not dict or set(frames) != set(SOURCES):
            raise ValueError("required_source_set_missing_or_extra")
        bodies = {}
        for source in SOURCES:
            packet = strict_loads(frames[source])
            if set(packet) != {"body", "mac"} or type(packet["body"]) is not dict:
                raise ValueError(f"{source}:invalid_envelope")
            body, mac = packet["body"], packet["mac"]
            if (
                type(mac) is not str
                or len(mac) != 64
                or not hmac.compare_digest(mac, _mac(body, self._keys[source], SOURCE_DOMAIN))
            ):
                raise ValueError(f"{source}:authentication_failed")
            if set(body) != {
                "source",
                "vehicle_id",
                "frame_id",
                "timestamp_ms",
                "sequence",
                "signals",
            }:
                raise ValueError(f"{source}:invalid_source_schema")
            if (
                body["source"] != source
                or body["vehicle_id"] != request.vehicle_id
                or not _text(body["frame_id"])
                or not _integer(body["sequence"])
                or not _integer(body["timestamp_ms"])
            ):
                raise ValueError(f"{source}:identity_or_metadata_mismatch")
            age = now_ms - body["timestamp_ms"]
            if not 0 <= age <= self.config.source_ttl_ms:
                raise ValueError(f"{source}:stale_or_future")
            self._validate_signals(source, body["signals"])
            if body["sequence"] <= self._watermarks.get((request.vehicle_id, source), -1):
                raise ValueError(f"{source}:sequence_replay")
            bodies[source] = body
        times = [b["timestamp_ms"] for b in bodies.values()]
        if len({b["frame_id"] for b in bodies.values()}) != 1:
            raise ValueError("frame_mismatch")
        if max(times) - min(times) > self.config.maximum_skew_ms:
            raise ValueError("source_time_skew")
        # Advance only after the complete authenticated context passes validation.
        for source, body in bodies.items():
            self._watermarks[request.vehicle_id, source] = body["sequence"]
        return bodies

    def _remember(self, decision: PolicyDecision) -> PolicyDecision:
        self._decisions[digest(asdict(decision))] = decision
        while len(self._decisions) > 4096:
            self._decisions.popitem(last=False)
        return decision

    def evaluate(
        self, request: VehicleRequest, frames: dict[str, str], now_ms: int
    ) -> PolicyDecision:
        if not isinstance(request, VehicleRequest) or not _integer(now_ms):
            raise ValueError("typed_request_and_integer_clock_required")
        with self._lock:
            if now_ms < self._clock_watermark:
                return self._remember(
                    PolicyDecision(
                        request,
                        "DENY",
                        "none",
                        1.0,
                        ("clock_rollback",),
                        (),
                        "",
                        self.policy_digest,
                        now_ms,
                        now_ms,
                    )
                )
            self._clock_watermark = now_ms
            try:
                bodies = self._context(request, frames, now_ms)
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                return self._remember(
                    PolicyDecision(
                        request,
                        "DENY",
                        "none",
                        1.0,
                        (str(exc),),
                        (),
                        "",
                        self.policy_digest,
                        now_ms,
                        now_ms,
                    )
                )
            s = {key: body["signals"] for key, body in bodies.items()}
            cfg = self.config
            components = (
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
                ("v2x_inconsistency", float(s["v2x"]["inconsistency_score"])),
                ("uncertainty", float(s["v2x"]["uncertainty"])),
            )
            risk = max(v for _, v in components)
            reasons = []
            identity = s["identity"]
            if identity["subject_id"] != request.subject_id:
                reasons.append("subject_mismatch")
            if not identity["credential_valid"]:
                reasons.append("credential_invalid")
            if identity["revoked"]:
                reasons.append("credential_revoked")
            if not identity["attestation_ok"]:
                reasons.append("attestation_failed")
            if (
                request.subject_id,
                request.vehicle_id,
                request.resource,
                request.operation,
            ) not in self._permissions:
                reasons.append("scope_not_authorized")
            if request.resource == "vehicle.actuator":
                reasons.append("actuator_control_out_of_scope")
            mode = "none"
            if reasons:
                outcome = "DENY"
            elif risk >= cfg.deny_threshold:
                outcome = "DENY"
                reasons.append("high_context_risk")
            elif risk < cfg.allow_threshold:
                outcome, mode = "ALLOW", "stream" if request.operation == "read" else "single"
                reasons.append("context_contract_satisfied")
            elif request.operation == "read" and risk < cfg.restrict_threshold:
                outcome, mode = "RESTRICT", "snapshot"
                reasons.append("diagnostic_snapshot_only")
            else:
                outcome = "CHALLENGE"
                reasons.append("fresh_corroborating_evidence_required")
            reasons.extend(
                "elevated_" + name for name, value in components if value >= cfg.allow_threshold
            )
            return self._remember(
                PolicyDecision(
                    request,
                    outcome,
                    mode,
                    risk,
                    tuple(reasons),
                    components,
                    digest(bodies),
                    self.policy_digest,
                    now_ms,
                    now_ms + cfg.capability_ttl_ms,
                )
            )

    def issue_capability(self, decision: PolicyDecision) -> str:
        """Only a decision issued by this engine can receive a signed grant."""
        with self._lock:
            if self._decisions.get(
                digest(asdict(decision))
            ) != decision or decision.outcome not in {"ALLOW", "RESTRICT"}:
                raise ValueError("eligible_engine_decision_required")
            body = {
                "request": asdict(decision.request),
                "mode": decision.mode,
                "context_digest": decision.context_digest,
                "policy_digest": decision.policy_digest,
                "issued_ms": decision.evaluated_ms,
                "expires_ms": decision.expires_ms,
            }
            return canonical_json(
                {"body": body, "mac": _mac(body, self._capability_key, CAPABILITY_DOMAIN)}
            )

    def consume_capability(
        self, capability: str, request: VehicleRequest, frames: dict[str, str], now_ms: int
    ) -> AuthorizationResult:
        """One-use reference PEP; caller must honor returned mode on its service.

        Re-evaluate current context before authorizing a data operation. This
        method returns a verdict; it does not send a network or actuator command.
        The data service must enforce expires_ms for any bounded stream lease.
        """
        if not isinstance(request, VehicleRequest) or not _integer(now_ms):
            raise ValueError("typed_request_and_integer_clock_required")
        with self._lock:
            if now_ms < self._clock_watermark:
                return AuthorizationResult(False, "none", ("clock_rollback",))
            self._clock_watermark = now_ms
            try:
                packet = strict_loads(capability)
                if set(packet) != {"body", "mac"} or type(packet["body"]) is not dict:
                    raise ValueError("invalid_capability_envelope")
                body, mac = packet["body"], packet["mac"]
                if (
                    type(mac) is not str
                    or len(mac) != 64
                    or not hmac.compare_digest(
                        mac, _mac(body, self._capability_key, CAPABILITY_DOMAIN)
                    )
                ):
                    raise ValueError("capability_authentication_failed")
                if set(body) != {
                    "request",
                    "mode",
                    "context_digest",
                    "policy_digest",
                    "issued_ms",
                    "expires_ms",
                }:
                    raise ValueError("invalid_capability_schema")
                if canonical_json(body["request"]) != canonical_json(asdict(request)):
                    raise ValueError("capability_request_mismatch")
                if body["policy_digest"] != self.policy_digest:
                    raise ValueError("capability_policy_mismatch")
                if (
                    not _integer(body["issued_ms"])
                    or not _integer(body["expires_ms"])
                    or not body["issued_ms"] <= now_ms < body["expires_ms"]
                    or body["expires_ms"] - body["issued_ms"] != self.config.capability_ttl_ms
                ):
                    raise ValueError("capability_expired_or_future")
                if body["mode"] not in {"stream", "single", "snapshot"}:
                    raise ValueError("invalid_capability_mode")
                # Retain every consumed nonce until its signed capability expires.
                self._used_nonces = {
                    key: expiry for key, expiry in self._used_nonces.items() if expiry > now_ms
                }
                nonce_key = (request.vehicle_id, request.nonce)
                if nonce_key in self._used_nonces:
                    raise ValueError("capability_replay")
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                return AuthorizationResult(False, "none", (str(exc),))
            current = self.evaluate(request, frames, now_ms)
            if current.outcome not in {"ALLOW", "RESTRICT"}:
                return AuthorizationResult(
                    False, "none", ("current_context_denied",) + current.reasons, current
                )
            self._used_nonces[nonce_key] = body["expires_ms"]
            mode = "snapshot" if "snapshot" in (body["mode"], current.mode) else body["mode"]
            return AuthorizationResult(
                True, mode, ("current_context_contract_satisfied",), current, body["expires_ms"]
            )
