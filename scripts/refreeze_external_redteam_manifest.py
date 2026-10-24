from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

from coact_cps.security.verifier import ExplanationVerifier


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a development-hardened verifier against unchanged external red-team "
            "task files before any held-out responses are generated"
        )
    )
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--development-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-heldout-uninspected", action="store_true")
    args = parser.parse_args()

    if not args.confirm_heldout_uninspected:
        raise RuntimeError(
            "Refusing to refreeze without --confirm-heldout-uninspected. This step is valid "
            "only before generating or inspecting held-out responses."
        )

    tasks_dir = Path(args.tasks_dir)
    parent_path = tasks_dir / "manifest.json"
    development_path = tasks_dir / "development_tasks.jsonl"
    heldout_path = tasks_dir / "heldout_tasks.jsonl"
    metrics_path = Path(args.development_metrics)
    for path in (parent_path, development_path, heldout_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if file_sha256(development_path) != parent.get("development_tasks_sha256"):
        raise RuntimeError("Development task file differs from the parent manifest")
    if file_sha256(heldout_path) != parent.get("heldout_tasks_sha256"):
        raise RuntimeError("Held-out task file differs from the parent manifest")

    verifier_path = Path(inspect.getsourcefile(ExplanationVerifier) or "")
    refrozen = dict(parent)
    refrozen.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_stage": "post-development_pre-heldout",
            "verifier_sha256": file_sha256(verifier_path),
            "parent_manifest_sha256": file_sha256(parent_path),
            "parent_verifier_sha256": parent.get("verifier_sha256"),
            "development_metrics_sha256": file_sha256(metrics_path),
            "heldout_responses_generated_before_refreeze": False,
            "hardening_note": (
                "The integrity-critical operator surface is now a canonical rendering of "
                "the closed evidence bundle; development attacks informed this design."
            ),
            "freeze_rule": (
                "Do not change the verifier, task prompts, model, or decoding settings after "
                "this refreeze and before reporting held-out outcomes."
            ),
        }
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(refrozen, indent=2), encoding="utf-8")
    print(json.dumps(refrozen, indent=2))
    print(f"Saved development-hardened manifest to {destination.resolve()}")


if __name__ == "__main__":
    main()
