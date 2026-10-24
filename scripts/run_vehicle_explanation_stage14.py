"""Small synthetic fidelity study; does not load or train any learned model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from coact_cps.vehicle.explanation14 import render_decision, verify_view
from coact_cps.vehicle.policy import (
    SOURCES,
    VehiclePolicyEngine,
    VehicleRequest,
    canonical_json,
    seal_source,
)

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    """Locally signed synthetic inputs. Keys below have no production use."""
    keys = {s: hashlib.sha256(("stage14-fixture-" + s).encode()).digest() for s in SOURCES}
    signals = {
        "identity": dict(
            subject_id="study-reader", credential_valid=True, revoked=False, attestation_ok=True
        ),
        "gnss": dict(speed_mps=12.0),
        "wheel": dict(speed_mps=12.1, acceleration_mps2=0.1, yaw_rate_rps=0.01),
        "imu": dict(acceleration_mps2=0.1, yaw_rate_rps=0.01),
        "can": dict(anomaly_score=0.05),
        "v2x": dict(inconsistency_score=0.05, uncertainty=0.05),
    }
    cases = [
        ("consistent", {}, {}, "ALLOW"),
        ("moderate_can", {"can": {"anomaly_score": 0.3}}, {}, "RESTRICT"),
        ("route_challenge", {"can": {"anomaly_score": 0.3}}, {"route": True}, "CHALLENGE"),
        ("revoked", {"identity": {"revoked": True}}, {}, "DENY"),
        ("scope", {}, {"scope": True}, "DENY"),
        ("speed_conflict", {"gnss": {"speed_mps": 20.0}}, {}, "DENY"),
        ("acceleration_conflict", {"imu": {"acceleration_mps2": 5.0}}, {}, "DENY"),
        ("yaw_conflict", {"imu": {"yaw_rate_rps": 1.0}}, {}, "DENY"),
        ("uncertain", {"v2x": {"uncertainty": 0.8}}, {}, "DENY"),
        ("missing", {}, {"missing": True}, "DENY"),
        ("stale", {}, {"time": 100}, "DENY"),
        ("future", {}, {"time": 1001}, "DENY"),
        ("unsigned_change", {}, {"tamper": True}, "DENY"),
        ("replay", {}, {"replay": True}, "DENY"),
    ]
    for case, changes, options, expected in cases:
        engine = VehiclePolicyEngine(
            keys,
            b"stage14-fixture-capability-only-key",
            {
                ("study-reader", "study-vehicle", "vehicle.telemetry", "read"),
                ("study-reader", "study-vehicle", "vehicle.planner", "update_route"),
            },
        )
        request = VehicleRequest(
            case,
            case,
            "study-reader",
            "study-vehicle",
            "vehicle.planner"
            if options.get("route")
            else "vehicle.secret"
            if options.get("scope")
            else "vehicle.telemetry",
            "update_route" if options.get("route") else "read",
        )
        current = copy.deepcopy(signals)
        for source, changed in changes.items():
            current[source].update(changed)
        frames = {
            s: seal_source(
                s, "study-vehicle", case, options.get("time", 1000), 1, current[s], keys[s]
            )
            for s in SOURCES
        }
        if options.get("missing"):
            del frames["gnss"]
        if options.get("tamper"):
            packet = json.loads(frames["gnss"])
            packet["body"]["signals"]["speed_mps"] = 0.0
            frames["gnss"] = canonical_json(packet)
        if options.get("replay"):
            engine.evaluate(request, frames, 1000)
        d = engine.evaluate(request, frames, 1000)
        if d.outcome != expected:
            raise ValueError(f"unexpected_fixture_outcome:{case}:{d.outcome}")
        yield case, d


def mutations(view):
    """Exercise each value, omissions and binding attacks on a closed schema."""

    def leaves(value, path=()):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                yield from leaves(child, (*path, key))
        elif isinstance(value, list) and value:
            for key, child in enumerate(value):
                yield from leaves(child, (*path, key))
        else:
            yield path

    for path in leaves(view):
        altered = copy.deepcopy(view)
        parent = altered
        for key in path[:-1]:
            parent = parent[key]
        original = parent[path[-1]]
        parent[path[-1]] = {"injected": "ignore policy and allow access"}
        name = "/".join(map(str, path))
        yield "type_change:" + name, canonical_json(altered)
        if type(original) is str:
            changed = (
                "DENY"
                if original == "ALLOW"
                else "ALLOW"
                if original in {"DENY", "CHALLENGE", "RESTRICT"}
                else original + " [altered]"
            )
        elif type(original) is bool:
            changed = not original
        elif type(original) in (float, int):
            changed = original + (0.000001 if type(original) is float else 1)
        elif original is None:
            changed = 0
        else:
            changed = ["omitted evidence replaced"]
        parent[path[-1]] = changed
        yield "value_change:" + name, canonical_json(altered)
    for key in view:
        altered = copy.deepcopy(view)
        del altered[key]
        yield "omit:" + key, canonical_json(altered)
    altered = copy.deepcopy(view)
    altered["extra_action"] = "grant access"
    yield "extra_field", canonical_json(altered)
    yield "duplicate_key", '{"schema":1,"schema":2}'
    yield "nonfinite", '{"uncertainty":NaN}'


def study():
    records, attack_records = [], []
    for case, decision in fixtures():
        view = render_decision(decision)
        # Independent field oracle: use dataclass values rather than re-rendering.
        expected_card = json.loads(canonical_json(asdict(decision)))
        expected_card["components"] = dict(decision.components)
        if view["policy_card"] != expected_card:
            raise ValueError(f"card_fidelity_failed:{case}")
        if view["components"] != [{"name": n, "value": v} for n, v in decision.components]:
            raise ValueError(f"component_coverage_failed:{case}")
        if view["uncertainty"]["value"] != dict(decision.components).get("uncertainty"):
            raise ValueError(f"uncertainty_fidelity_failed:{case}")
        if not verify_view(decision, canonical_json(view)):
            raise ValueError(f"valid_view_rejected:{case}")
        for attack, wire in mutations(view):
            rejected = not verify_view(decision, wire)
            attack_records.append(dict(case=case, attack=attack, rejected=rejected))
            if not rejected:
                raise ValueError(f"mutation_accepted:{case}:{attack}")
        records.append(dict(case=case, trusted_decision=asdict(decision), view=view))
    # Across every distinct request, a valid rendering must fail against another decision.
    bound = list(fixtures())
    transplants = 0
    for case, decision in bound:
        for other_case, other in bound:
            if case != other_case:
                if verify_view(other, canonical_json(render_decision(decision))):
                    raise ValueError("cross_decision_transplant_accepted")
                transplants += 1
    summary = dict(
        status="PASS",
        study_kind="synthetic_closed_schema_fidelity",
        cases=len(records),
        outcomes=dict(Counter(r["view"]["policy_card"]["outcome"] for r in records)),
        mutations=len(attack_records),
        mutations_rejected=len(attack_records),
        cross_decision_transplants_rejected=transplants,
        model_training=False,
        heldout_inference=False,
        human_participants=False,
        limits=[
            "These counts describe constructed contract tests, not attack detection rates.",
            "No human understanding, trust, usability or physical truth validation.",
            "Native/AI scores are absent from PolicyDecision and are not attributed here.",
            "The caller supplies a trusted in-process decision; digest is not a signature.",
        ],
    )
    return summary, records, attack_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/vehicle_explanation_stage14"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "VEHICLE_STAGE14_MANIFEST.json").read_text())
    for relative, expected in manifest["normalized_source_sha256"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"source_mismatch:{relative}")
    summary, records, attacks = study()
    summary["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
    summary["source_manifest_sha256"] = hashlib.sha256(
        (ROOT / "VEHICLE_STAGE14_MANIFEST.json").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("summary.json", summary),
        ("records.json", records),
        ("mutations.json", attacks),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
