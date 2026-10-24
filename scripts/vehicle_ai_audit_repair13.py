"""Explicit precision-only audit addendum. Original frozen R1 bytes stay intact."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_vehicle_ai_stage13 import (
    OUT,
    check_frozen,
    evaluate_phase,
    fit,
    freeze,
    sha,
    source_digest,
    verify_fit,
    write_json,
)

REPAIR = "audit_precision_addendum.json"
EXTRA = [
    "scripts/vehicle_ai_audit_repair13.py",
    "scripts/audit_vehicle_ai_stage13_float64.py",
    "scripts/resume_vehicle_ai_validation13.py",
]


def register(root):
    root = Path(root).resolve()
    check_frozen(root)
    p = root / OUT / REPAIR
    if p.exists():
        return check_repair(root)
    lock = root / OUT / "fit/model_lock.json"
    partial = root / OUT / "validation"
    if partial.exists() and (partial / "summary.json").exists():
        raise ValueError("completed_validation_must_not_be_replaced")
    prior = (
        {
            v.relative_to(partial).as_posix(): sha(v.read_bytes())
            for v in sorted(partial.rglob("*"))
            if v.is_file()
        }
        if partial.exists()
        else {}
    )
    value = dict(
        schema="vehicle-ai-stage13r1-audit-precision-addendum-v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        original_model_lock_sha256=sha(lock.read_bytes()) if lock.exists() else None,
        source_sha256={n: source_digest(root / n) for n in EXTRA},
        interrupted_validation_sha256=prior,
        reason=(
            "fit calibration casts raw LOF scores to float64; auditor must use same precision. "
            "No model/scaler/threshold/feature/scoring change. "
            "Incomplete validation retained; only missing evidence completed."
        ),
        status="EXPERIMENTAL_NOT_PROMOTED",
    )
    write_json(p, value)
    return value


def check_repair(root):
    root = Path(root)
    check_frozen(root)
    p = root / OUT / REPAIR
    v = json.loads(p.read_bytes())
    if v["original_protocol_sha256"] != sha((root / OUT / "protocol.json").read_bytes()):
        raise ValueError("audit_addendum_protocol_changed")
    if set(v["source_sha256"]) != set(EXTRA):
        raise ValueError("audit_addendum_source_inventory_changed")
    for n, h in v["source_sha256"].items():
        if source_digest(root / n) != h:
            raise ValueError("audit_addendum_source_changed:" + n)
    if v["original_model_lock_sha256"] is not None and v["original_model_lock_sha256"] != sha(
        (root / OUT / "fit/model_lock.json").read_bytes()
    ):
        raise ValueError("audit_repair_must_not_change_fit")
    if v["interrupted_validation_sha256"]:
        old = root / OUT / "validation_interrupted"
        if not old.exists():
            old = root / OUT / "validation"
        actual = {
            p.relative_to(old).as_posix(): sha(p.read_bytes())
            for p in old.rglob("*")
            if p.is_file()
        }
        if actual != v["interrupted_validation_sha256"]:
            raise ValueError("interrupted_validation_evidence_changed")
    return v


def assert_fit_audited(root):
    root = Path(root)
    f, lock = verify_fit(root)
    folder = root / OUT / "fit"
    a = json.loads((folder / "independent_audit.json").read_bytes())
    if a.get("status") != "PASS" or a.get("phase") != "fit":
        raise ValueError("completed_fit_audit_required")
    checks = {
        "protocol_sha256": root / OUT / "protocol.json",
        "model_lock_sha256": folder / "model_lock.json",
        "phase_hashes_sha256": folder / "file_hashes.json",
        "audit_repair_sha256": root / OUT / REPAIR,
    }
    if any(a.get(k) != sha(p.read_bytes()) for k, p in checks.items()):
        raise ValueError("fit_audit_binding_changed")
    return f, lock


def verify_recovered_copies(folder, summary, addendum, addendum_hash):
    """Original NPZ bytes and declared completion counts must agree independently."""
    folder = Path(folder)
    prior = addendum["interrupted_validation_sha256"]
    contexts = {n: h for n, h in prior.items() if n.startswith("contexts/") and n.endswith(".npz")}
    blocks = {n: h for n, h in prior.items() if n.startswith("blocks/") and n.endswith(".npz")}
    rows = summary["rows"]
    expected = {r[k] for r in rows for k in ["context", "block"]}
    if not (set(contexts) | set(blocks)).issubset(expected):
        raise ValueError("retained_file_missing_from_completed_grid")
    for n, h in {**contexts, **blocks}.items():
        if sha((folder / n).read_bytes()) != h:
            raise ValueError("retained_output_copy_changed:" + n)
    recovery = summary.get("recovery", {})
    expected_counts = dict(
        reused_contexts=len(contexts),
        reused_score_blocks=len(blocks),
        new_score_blocks=len(rows) - len(blocks),
    )
    if any(recovery.get(k) != v for k, v in expected_counts.items()):
        raise ValueError("recovery_counts_do_not_match_registered_evidence")
    if (
        recovery.get("addendum_sha256") != addendum_hash
        or recovery.get("retained_folder") != "validation_interrupted"
        or recovery.get("no_model_refit") is not True
        or recovery.get("no_threshold_change") is not True
        or recovery.get("invalid_partial_files") != []
    ):
        raise ValueError("recovery_addendum_claim_mismatch")
    return expected_counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--register", action="store_true")
    p.add_argument("--phase", choices=["freeze", "fit", "validation", "test"])
    a = p.parse_args()
    root = a.root.resolve()
    if a.register:
        register(root)
    elif a.phase == "freeze":
        freeze(root)
        register(root)
    else:
        check_repair(root)
        if a.phase == "fit":
            fit(root)
        elif a.phase == "validation":
            assert_fit_audited(root)
            evaluate_phase(root, "validation")
        elif a.phase == "test":
            assert_fit_audited(root)
            evaluate_phase(root, "test")
        else:
            p.error("--register or --phase required")
