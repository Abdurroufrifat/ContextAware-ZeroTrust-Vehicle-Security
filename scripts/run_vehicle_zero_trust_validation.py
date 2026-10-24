"""Reproducible synthetic contract cases, not an AV performance benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import secrets
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.types import GeneratedExplanation
from coact_cps.vehicle.explanation import (
    analyze_vehicle_request,
    verify_decision_card,
)
from coact_cps.vehicle.policy import (
    SOURCES,
    PolicyConfig,
    PolicyDecision,
    VehiclePolicyEngine,
    VehicleRequest,
    canonical_json,
    seal_source,
)

BASE_SIGNALS = {
    "identity": {
        "subject_id": "fleet-service",
        "credential_valid": True,
        "revoked": False,
        "attestation_ok": True,
    },
    "gnss": {"speed_mps": 12.0},
    "imu": {"acceleration_mps2": 0.1, "yaw_rate_rps": 0.01},
    "wheel": {"speed_mps": 12.1, "acceleration_mps2": 0.1, "yaw_rate_rps": 0.01},
    "can": {"anomaly_score": 0.05},
    "v2x": {"inconsistency_score": 0.05, "uncertainty": 0.05},
}
# Expectations are fixture oracle labels. They never enter the policy input.
CASES = [
    ("benign_consistent", {}, "ALLOW"),
    (
        "benign_stationary",
        {"signals": {"gnss": {"speed_mps": 0}, "wheel": {"speed_mps": 0}}},
        "ALLOW",
    ),
    (
        "benign_deceleration",
        {"signals": {"imu": {"acceleration_mps2": -2}, "wheel": {"acceleration_mps2": -2}}},
        "ALLOW",
    ),
    ("revoked_credential", {"signals": {"identity": {"revoked": True}}}, "DENY"),
    ("invalid_credential", {"signals": {"identity": {"credential_valid": False}}}, "DENY"),
    ("failed_attestation", {"signals": {"identity": {"attestation_ok": False}}}, "DENY"),
    ("identity_mismatch", {"signals": {"identity": {"subject_id": "other"}}}, "DENY"),
    ("unlisted_resource", {"resource": "vehicle.private"}, "DENY"),
    ("actuator_write", {"resource": "vehicle.actuator", "operation": "write"}, "DENY"),
    ("signed_speed_conflict", {"signals": {"gnss": {"speed_mps": 25}}}, "DENY"),
    ("signed_acceleration_conflict", {"signals": {"imu": {"acceleration_mps2": 5}}}, "DENY"),
    ("signed_yaw_conflict", {"signals": {"imu": {"yaw_rate_rps": 1}}}, "DENY"),
    ("high_can_anomaly", {"signals": {"can": {"anomaly_score": 0.9}}}, "DENY"),
    ("v2x_conflict", {"signals": {"v2x": {"inconsistency_score": 0.9}}}, "DENY"),
    ("high_uncertainty", {"signals": {"v2x": {"uncertainty": 0.9}}}, "DENY"),
    ("moderate_read_risk", {"signals": {"can": {"anomaly_score": 0.3}}}, "RESTRICT"),
    (
        "moderate_route_update_risk",
        {
            "resource": "vehicle.planner",
            "operation": "update_route",
            "signals": {"can": {"anomaly_score": 0.3}},
        },
        "CHALLENGE",
    ),
    ("challenge_risk", {"signals": {"can": {"anomaly_score": 0.6}}}, "CHALLENGE"),
    ("missing_sensor", {"missing": "gnss"}, "DENY"),
    ("expired_context", {"age_ms": 501}, "DENY"),
    ("future_context", {"age_ms": -1}, "DENY"),
    ("unsynchronized_context", {"skew_ms": 100}, "DENY"),
    ("source_forgery", {"tamper": True}, "DENY"),
    ("source_replay", {"replay": True}, "DENY"),
]


def make_context(keys, modifiers, seq=1, now=100000):
    signals = copy.deepcopy(BASE_SIGNALS)
    for source, values in modifiers.get("signals", {}).items():
        signals[source].update(values)
    frames = {
        source: seal_source(
            source,
            "av-1",
            "frame-1",
            now
            - modifiers.get("age_ms", 0)
            - (modifiers.get("skew_ms", 0) if source == "gnss" else 0),
            seq,
            signals[source],
            keys[source],
        )
        for source in SOURCES
    }
    if "missing" in modifiers:
        del frames[modifiers["missing"]]
    if modifiers.get("tamper"):
        packet = json.loads(frames["gnss"])
        packet["body"]["signals"]["speed_mps"] = 0
        frames["gnss"] = canonical_json(packet)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/vehicle_zero_trust_stage1")
    parser.add_argument("--config", default="configs/vehicle_zero_trust_stage1.json")
    args = parser.parse_args()
    cfg_path, output = Path(args.config), Path(args.output)
    config = PolicyConfig(**json.loads(cfg_path.read_text())["policy"])
    output.mkdir(parents=True, exist_ok=True)
    keys = {source: secrets.token_bytes(32) for source in SOURCES}
    capability_key = secrets.token_bytes(32)
    permissions = {
        ("fleet-service", "av-1", "vehicle.telemetry", "read"),
        ("fleet-service", "av-1", "vehicle.planner", "update_route"),
    }
    records, cards, mutation_rows = [], [], []
    explanation_mutations = []
    for case_id, modifiers, expected in CASES:
        engine = VehiclePolicyEngine(keys, capability_key, permissions, config)
        request = VehicleRequest(
            case_id,
            case_id + "-nonce",
            "fleet-service",
            "av-1",
            modifiers.get("resource", "vehicle.telemetry"),
            modifiers.get("operation", "read"),
        )
        if modifiers.get("replay"):
            engine.evaluate(request, make_context(keys, {}), 100000)
        start = time.perf_counter_ns()
        result = analyze_vehicle_request(engine, request, make_context(keys, modifiers), 100000)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        card = result["policy_card"]
        # Independently parse the result for the case-level oracle comparison.
        records.append(
            {
                "case_id": case_id,
                "expected": expected,
                "actual": card["outcome"],
                "contract_passed": card["outcome"] == expected,
                "risk_index": card["risk"],
                "mode": card["mode"],
                "explanation_status": result["verification"]["status"],
                "elapsed_ms": elapsed_ms,
            }
        )
        cards.append({"case_id": case_id, **result, "capability": None})
        # Obtain the trusted decision by its existing registered digest. The
        # engine does not receive any fixture labels, and is not retuned here.
        values = dict(card)
        values["request"] = VehicleRequest(**card["request"])
        values["reasons"] = tuple(card["reasons"])
        values["components"] = tuple(card["components"].items())
        decision = PolicyDecision(**values)
        for field in (
            "outcome",
            "risk",
            "reasons",
            "mode",
            "context_digest",
            "policy_digest",
            "expires_ms",
            "request",
        ):
            altered = copy.deepcopy(card)
            if field == "outcome":
                altered[field] = "DENY" if card[field] != "DENY" else "ALLOW"
            elif field == "risk":
                altered[field] = card[field] + 0.123
            elif field == "reasons":
                altered[field] = []
            elif field == "expires_ms":
                altered[field] += 1
            elif field == "request":
                altered[field]["resource"] = (
                    "vehicle.diagnostics"
                    if card[field]["resource"] == "vehicle.actuator"
                    else "vehicle.actuator"
                )
            else:
                altered[field] = "tampered"
            accepted = verify_decision_card(decision, canonical_json(altered))
            mutation_rows.append({"case_id": case_id, "field": field, "accepted": accepted})
        from coact_cps.vehicle.explanation import build_vehicle_bundle

        bundle = build_vehicle_bundle(decision)
        for field in ("summary", "recommendation", "probable_cause", "evidence_ids"):
            altered = GeneratedExplanation(**copy.deepcopy(result["explanation"]))
            setattr(altered, field, [] if field == "evidence_ids" else "unsupported request")
            status = ExplanationVerifier().verify(bundle, altered).status
            explanation_mutations.append(
                {"case_id": case_id, "field": field, "accepted": status == "VERIFIED"}
            )
    for filename, rows in [
        ("contract_cases.csv", records),
        ("policy_card_mutations.csv", mutation_rows),
        ("explanation_mutations.csv", explanation_mutations),
    ]:
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "decision_examples.json").write_text(json.dumps(cards, indent=2))
    source_files = [
        Path(__file__),
        cfg_path,
        Path("src/coact_cps/vehicle/policy.py"),
        Path("src/coact_cps/vehicle/explanation.py"),
        Path("src/coact_cps/security/operator_surface.py"),
        Path("src/coact_cps/security/verifier.py"),
        Path("src/coact_cps/genai/grounded_generator.py"),
        Path("src/coact_cps/types.py"),
    ]
    metrics = {
        "evidence_scope": "synthetic software contract validation; no AV generalization claim",
        "cases": len(records),
        "case_contracts_passed": sum(r["contract_passed"] for r in records),
        "authorized_cards_verified": sum(r["explanation_status"] == "VERIFIED" for r in records),
        "policy_card_mutations": len(mutation_rows),
        "policy_card_mutations_accepted": sum(r["accepted"] for r in mutation_rows),
        "explanation_mutations": len(explanation_mutations),
        "explanation_mutations_accepted": sum(r["accepted"] for r in explanation_mutations),
        "policy": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "latency_note": (
            "24 heterogeneous software cases; descriptive timing only, not a real-time bound"
        ),
        "latency_median_ms": statistics.median(r["elapsed_ms"] for r in records),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "keys": "ephemeral process-local random test keys; never saved",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"source_sha256", "policy", "platform"}
            },
            indent=2,
        )
    )
    if (
        metrics["case_contracts_passed"] != len(records)
        or metrics["authorized_cards_verified"] != len(records)
        or any(r["accepted"] for r in mutation_rows + explanation_mutations)
    ):
        raise SystemExit("Contract validation failed; inspect per-case outputs")


if __name__ == "__main__":
    main()
