"""Traffic-derived observation proxies and paired development-policy comparisons.

SUMO state is not native GNSS/IMU/wheel/CAN telemetry. Each source below has
explicit proxy/synthetic provenance. Attack annotations never enter a policy API.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import math
import random
import secrets
import statistics
import time
from collections import defaultdict

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.vehicle.explanation import build_vehicle_bundle, decision_card
from coact_cps.vehicle.policy import (
    SOURCE_DOMAIN,
    SOURCES,
    VehiclePolicyEngine,
    VehicleRequest,
    canonical_json,
    seal_source,
    strict_loads,
)

SCENARIOS = (
    ("normal", False),
    ("benign_gnss_bias", False),
    ("benign_v2x_loss", False),
    ("gnss_spoof", True),
    ("slow_gnss_drift", True),
    ("can_anomaly", True),
    ("identity_revocation", True),
    ("stale_gnss", True),
    ("source_replay", True),
    ("coordinated_speed_spoof", True),
    ("display_downplay", True),
    ("post_issue_revocation", True),
)
POLICIES = (
    "identity_only",
    "can_only",
    "multisource_no_kinematics",
    "full_unbound_display",
    "full_no_use_recheck",
    "full",
)


def wrapped_angle_delta(before, after):
    return (after - before + 180) % 360 - 180


def make_observations(trace, seed):
    streams = {
        source: random.Random(
            int.from_bytes(
                hashlib.sha256(f"coact-sumo-proxy-v1:{seed}:{source}".encode()).digest()[:8], "big"
            )
        )
        for source in SOURCES
    }
    observations, previous = [], None
    yaw_history = []
    for row in trace:
        dt = row["time_s"] - previous["time_s"] if previous else 0.2
        if dt <= 0:
            raise ValueError("trace time must strictly increase")
        yaw = (
            math.radians(wrapped_angle_delta(previous["angle_deg"], row["angle_deg"])) / dt
            if previous
            else 0.0
        )
        yaw_history.append(yaw)
        # Smoothed and saturated traffic-heading proxy, not a physical IMU.
        yaw_proxy = max(-9.0, min(9.0, statistics.mean(yaw_history[-3:])))
        position_noise = streams["gnss"]
        noisy_position = (
            row["x_m"] + position_noise.gauss(0, 0.005),
            row["y_m"] + position_noise.gauss(0, 0.005),
        )
        if previous:
            speed_proxy = math.dist(noisy_position, previous["noisy_position"]) / dt
        else:
            speed_proxy = row["speed_mps"]
        acceleration = max(-25, min(25, row["acceleration_mps2"]))
        signals = {
            "identity": {
                "subject_id": "fleet-service",
                "credential_valid": True,
                "revoked": False,
                "attestation_ok": True,
            },
            "gnss": {"speed_mps": min(99.0, max(0.0, speed_proxy))},
            "imu": {
                "acceleration_mps2": acceleration + streams["imu"].gauss(0, 0.03),
                "yaw_rate_rps": yaw_proxy + streams["imu"].gauss(0, 0.005),
            },
            "wheel": {
                "speed_mps": max(0.0, row["speed_mps"] * 1.002 + streams["wheel"].gauss(0, 0.02)),
                "acceleration_mps2": acceleration + streams["wheel"].gauss(0, 0.02),
                "yaw_rate_rps": yaw_proxy + streams["wheel"].gauss(0, 0.004),
            },
            "can": {
                "anomaly_score": 0.02
                + min(abs(acceleration), 10) * 0.002
                + streams["can"].uniform(0, 0.025)
            },
            "v2x": {
                "inconsistency_score": streams["v2x"].uniform(0.01, 0.07),
                "uncertainty": streams["v2x"].uniform(0.03, 0.12)
                + min(row.get("neighbors_50m", 0), 8) * 0.004,
            },
        }
        observations.append(
            {
                "frame": row["frame"],
                "timestamp_ms": int(round(row["time_s"] * 1000)) + 1000000,
                "signals": signals,
            }
        )
        previous = dict(row, noisy_position=noisy_position)
    return observations


def inject_scenario(observation, scenario, progress):
    row, transport = copy.deepcopy(observation), {}
    s = row["signals"]
    if scenario == "benign_gnss_bias":
        s["gnss"]["speed_mps"] += 0.8
    elif scenario in {"gnss_spoof", "display_downplay"}:
        s["gnss"]["speed_mps"] += 8.0
        if scenario == "display_downplay":
            transport["display_mutation"] = True
    elif scenario == "slow_gnss_drift":
        s["gnss"]["speed_mps"] += 4.0 * max(0.0, min(1.0, progress))
    elif scenario == "can_anomaly":
        s["can"]["anomaly_score"] = 0.85
    elif scenario == "identity_revocation":
        s["identity"]["revoked"] = True
    elif scenario == "benign_v2x_loss":
        transport["missing"] = "v2x"
    elif scenario == "stale_gnss":
        transport["age_ms"] = 700
    elif scenario == "source_replay":
        transport["replay"] = "gnss"
    elif scenario == "coordinated_speed_spoof":
        s["gnss"]["speed_mps"] += 8.0
        s["wheel"]["speed_mps"] += 8.0
    elif scenario == "post_issue_revocation":
        transport["revoke_at_use"] = True
    elif scenario != "normal":
        raise ValueError("unknown scenario")
    return row, transport


def verify_exposure_card(clean, candidate_json):
    try:
        return canonical_json(strict_loads(candidate_json)) == canonical_json(clean)
    except (ValueError, TypeError, RecursionError):
        return False


def grade(risk):
    if risk >= 0.75:
        return "DENY", "none"
    if risk < 0.25:
        return "ALLOW", "stream"
    if risk < 0.5:
        return "RESTRICT", "snapshot"
    return "CHALLENGE", "none"


class PolicyHarness:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.keys = {source: secrets.token_bytes(32) for source in SOURCES}
        self.permissions = {("fleet-service", vehicle_id, "vehicle.telemetry", "read")}
        self.engine = VehiclePolicyEngine(self.keys, secrets.token_bytes(32), self.permissions)
        self.baseline_watermarks = {name: {} for name in POLICIES[:3]}
        self.previous_issue_packets = None

    def packets(self, observation, sequence, frame_id, timestamp_ms, transport):
        result = {
            source: seal_source(
                source,
                self.vehicle_id,
                frame_id,
                timestamp_ms - (transport.get("age_ms", 0) if source == "gnss" else 0),
                sequence,
                observation["signals"][source],
                self.keys[source],
            )
            for source in SOURCES
        }
        if transport.get("missing"):
            del result[transport["missing"]]
        if transport.get("replay") and self.previous_issue_packets:
            result[transport["replay"]] = self.previous_issue_packets[transport["replay"]]
        return result

    def baseline(self, name, request, frames, now_ms, expected_frame):
        required = (
            ("identity",)
            if name == "identity_only"
            else (("identity", "can") if name == "can_only" else SOURCES)
        )
        bodies = {}
        try:
            for source in required:
                packet = strict_loads(frames[source])
                if set(packet) != {"body", "mac"}:
                    raise ValueError
                body = packet["body"]
                if type(body) is not dict or set(body) != {
                    "source",
                    "vehicle_id",
                    "frame_id",
                    "timestamp_ms",
                    "sequence",
                    "signals",
                }:
                    raise ValueError
                expected_mac = hmac.new(
                    self.keys[source],
                    SOURCE_DOMAIN + canonical_json(body).encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
                if type(packet["mac"]) is not str or not hmac.compare_digest(
                    packet["mac"], expected_mac
                ):
                    raise ValueError
                if (
                    body["source"] != source
                    or body["vehicle_id"] != request.vehicle_id
                    or body["frame_id"] != expected_frame
                    or type(body["timestamp_ms"]) is not int
                    or type(body["sequence"]) is not int
                    or not 0 <= now_ms - body["timestamp_ms"] <= 500
                    or body["sequence"] <= self.baseline_watermarks[name].get(source, -1)
                ):
                    raise ValueError
                VehiclePolicyEngine._validate_signals(source, body["signals"])
                bodies[source] = body
            times = [body["timestamp_ms"] for body in bodies.values()]
            if max(times) - min(times) > 50:
                raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError):
            return dict(outcome="DENY", risk=1.0, mode="none")
        for source, body in bodies.items():
            self.baseline_watermarks[name][source] = body["sequence"]
        identity = bodies["identity"]["signals"]
        if (
            identity["subject_id"] != request.subject_id
            or not identity["credential_valid"]
            or identity["revoked"]
            or not identity["attestation_ok"]
            or (request.subject_id, request.vehicle_id, request.resource, request.operation)
            not in self.permissions
        ):
            return dict(outcome="DENY", risk=1.0, mode="none")
        risk = 0.0
        if name != "identity_only":
            risk = bodies["can"]["signals"]["anomaly_score"]
        if name == "multisource_no_kinematics":
            risk = max(
                risk,
                bodies["v2x"]["signals"]["inconsistency_score"],
                bodies["v2x"]["signals"]["uncertainty"],
            )
        outcome, mode = grade(risk)
        return dict(outcome=outcome, risk=risk, mode=mode)

    def step(self, observation, transport, active, scenario):
        # active/scenario are benchmark metadata only, never passed to a policy.
        frame, now = observation["frame"], observation["timestamp_ms"]
        frame_id = f"frame-{frame}"
        request = VehicleRequest(
            f"req-{frame}",
            f"nonce-{frame}",
            "fleet-service",
            self.vehicle_id,
            "vehicle.telemetry",
            "read",
        )
        packets = self.packets(observation, 2 * frame + 1, frame_id, now, transport)
        results = {}
        for name in POLICIES[:3]:
            start = time.perf_counter_ns()
            verdict = self.baseline(name, request, packets, now, frame_id)
            verdict.update(
                any_grant=verdict["outcome"] in {"ALLOW", "RESTRICT"},
                full_grant=verdict["outcome"] == "ALLOW",
                display_accepted=None,
                display_mutated=False,
                latency_ms=(time.perf_counter_ns() - start) / 1e6,
            )
            results[name] = verdict
        start = time.perf_counter_ns()
        issued = self.engine.evaluate(request, packets, now)
        effective, authorized, mode, deadline = issued, False, "none", 0
        if issued.outcome in {"ALLOW", "RESTRICT"}:
            capability = self.engine.issue_capability(issued)
            at_use = copy.deepcopy(observation)
            if transport.get("revoke_at_use"):
                at_use["signals"]["identity"]["revoked"] = True
            fresh = self.packets(at_use, 2 * frame + 2, frame_id + "-use", now + 1, transport)
            authorization = self.engine.consume_capability(capability, request, fresh, now + 1)
            effective = authorization.current_decision or issued
            authorized, mode, deadline = (
                authorization.authorized,
                authorization.mode,
                authorization.expires_ms,
            )
        bundle = build_vehicle_bundle(effective)
        explanation = GroundedExplanationGenerator().generate(bundle)
        explanation_status = ExplanationVerifier().verify(bundle, explanation).status
        clean_card = {
            "policy": decision_card(effective),
            "service": {"authorized": authorized, "mode": mode, "expires_ms": deadline},
            "explanation": explanation.to_dict(),
        }
        candidate = copy.deepcopy(clean_card)
        mutated = bool(transport.get("display_mutation"))
        if mutated:
            candidate["policy"].update(outcome="ALLOW", risk=0.0, reasons=["synthetic_reassurance"])
            candidate["service"].update(authorized=True, mode="stream")
            candidate["explanation"]["summary"] = "The simulated access request may proceed."
        accepted = explanation_status == "VERIFIED" and verify_exposure_card(
            clean_card, canonical_json(candidate)
        )
        latency = (time.perf_counter_ns() - start) / 1e6
        full = dict(
            outcome=effective.outcome,
            risk=effective.risk,
            mode=mode,
            any_grant=authorized,
            full_grant=authorized and mode == "stream",
            issuance_outcome=issued.outcome,
            display_accepted=accepted,
            display_mutated=mutated,
            explanation_status=explanation_status,
            latency_ms=latency,
            reason=";".join(effective.reasons),
        )
        results["full"] = full
        results["full_unbound_display"] = dict(full, display_accepted=True)
        initial_grant = issued.outcome in {"ALLOW", "RESTRICT"}
        results["full_no_use_recheck"] = dict(
            full,
            outcome=issued.outcome,
            risk=issued.risk,
            mode=issued.mode,
            any_grant=initial_grant,
            full_grant=issued.outcome == "ALLOW",
            display_mutated=False,
            display_accepted=None,
            explanation_status="not_evaluated_for_counterfactual",
            reason="no_use_time_revalidation_counterfactual",
        )
        # Record fresh issued bytes, not the potentially replayed adversarial packet.
        self.previous_issue_packets = self.packets(observation, 2 * frame + 1, frame_id, now, {})
        return results


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    result = {}
    for policy, subset in grouped.items():
        attacks = [r for r in subset if r["active"] and r["is_attack"]]
        benign = [r for r in subset if not (r["active"] and r["is_attack"])]
        altered = [r for r in subset if r["display_mutated"]]

        def fraction(population, field):
            return sum(bool(r[field]) for r in population) / len(population) if population else None

        result[policy] = {
            "requests": len(subset),
            "traffic_seed_clusters": len({r["seed"] for r in subset}),
            "attack_window_requests": len(attacks),
            "benign_requests": len(benign),
            "attack_window_any_grant_fraction": fraction(attacks, "any_grant"),
            "attack_window_full_grant_fraction": fraction(attacks, "full_grant"),
            "benign_any_grant_fraction": fraction(benign, "any_grant"),
            "benign_full_grant_fraction": fraction(benign, "full_grant"),
            "altered_display_cards": len(altered),
            "altered_display_acceptance_fraction": fraction(altered, "display_accepted"),
            "outcomes": {
                outcome: sum(r["outcome"] == outcome for r in subset)
                for outcome in ("ALLOW", "RESTRICT", "CHALLENGE", "DENY")
            },
            "latency_median_ms": statistics.median(r["latency_ms"] for r in subset),
            "latency_scope": "CPU policy/protocol pipeline; counterfactual control timings share full pipeline",
        }
    return result
