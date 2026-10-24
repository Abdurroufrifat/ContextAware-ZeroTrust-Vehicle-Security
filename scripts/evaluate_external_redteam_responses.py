from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from coact_cps.security.external_redteam import evaluate_external_responses
from coact_cps.security.verifier import ExplanationVerifier


def read_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate external LLM red-team responses against frozen verifiers"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    verifier_path = Path(inspect.getsourcefile(ExplanationVerifier) or "")
    current_hash = file_sha256(verifier_path)
    if current_hash != manifest.get("verifier_sha256"):
        raise RuntimeError(
            "Verifier drift detected. Restore the frozen verifier before evaluation; "
            "held-out evaluation cannot continue with modified defense code."
        )

    tasks = read_jsonl(args.tasks)
    splits = {str(task.get("split", "")) for task in tasks}
    if len(splits) != 1:
        raise RuntimeError("Evaluation task file must contain exactly one frozen split")
    split = next(iter(splits))
    task_hash_key = (
        "development_tasks_sha256" if split == "development" else "heldout_tasks_sha256"
    )
    if file_sha256(args.tasks) != manifest.get(task_hash_key):
        raise RuntimeError(
            "Task-file drift detected. Restore the frozen task JSONL before evaluation."
        )
    responses = read_jsonl(args.responses)
    result = evaluate_external_responses(tasks, responses)
    result["frozen_verifier_sha256"] = current_hash
    result["response_models"] = sorted(
        {str(response.get("model_id", "unreported")) for response in responses}
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved external red-team metrics to {destination.resolve()}")


if __name__ == "__main__":
    main()
