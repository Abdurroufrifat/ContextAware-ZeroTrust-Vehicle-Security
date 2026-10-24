from __future__ import annotations

import argparse
import json

import generate_external_redteam_ollama as ollama


EXPECTED = {
    "phi4-mini:3.8b-q4_K_M": "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753",
    "gemma4:12b": "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
    "ministral-3:14b-instruct-2512-q4_K_M": "4760c35aeb9d9e9c6174c2492562c0b999e80a222804fd96b1915ab72bbcdcf7",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen V13 Ollama snapshots")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="artifacts/hai_external_redteam/v13_heldout/model_snapshot_check.json")
    args = parser.parse_args()

    records = []
    failed = False
    for model, expected in EXPECTED.items():
        try:
            snapshot = ollama.resolve_model_snapshot(args.api_base, model)
            observed = snapshot["model_digest"]
            passed = observed == expected
            records.append({"model": model, "expected_digest": expected, "observed_digest": observed, "passed": passed, "snapshot": snapshot})
            failed |= not passed
        except Exception as exc:
            records.append({"model": model, "expected_digest": expected, "passed": False, "error": str(exc)})
            failed = True

    result = {
        "protocol_version": "co-act-cps-heldout-multimodel-v13.0",
        "all_snapshots_match": not failed,
        "models": records,
    }
    from pathlib import Path
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit("Snapshot check failed. Do not run heldout generation.")


if __name__ == "__main__":
    main()
