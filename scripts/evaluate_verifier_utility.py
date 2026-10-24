from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coact_cps.security.surface_utility import evaluate_surface_utility


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clean, registered-benign, and adversarial explanation coverage "
            "against the V7 and V8 operator-surface gates"
        )
    )
    parser.add_argument(
        "--tasks",
        action="append",
        required=True,
        help="Frozen task JSONL; repeat for development and held-out files",
    )
    parser.add_argument(
        "--responses",
        action="append",
        required=True,
        help="External-model response JSONL; repeat to combine frozen attacker models",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = [record for path in args.tasks for record in read_jsonl(path)]
    responses = [record for path in args.responses for record in read_jsonl(path)]
    result = evaluate_surface_utility(tasks, responses)
    result["input_sha256"] = {
        "tasks": {str(path): file_sha256(path) for path in args.tasks},
        "responses": {str(path): file_sha256(path) for path in args.responses},
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved utility-aware verifier metrics to {destination.resolve()}")


if __name__ == "__main__":
    main()
