from __future__ import annotations

import argparse
import re

if __package__:
    from scripts.generate_external_redteam_ollama import resolve_model_snapshot
else:  # Direct execution: python scripts/<name>.py
    from generate_external_redteam_ollama import resolve_model_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed on Ollama model drift")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434/api")
    args = parser.parse_args()
    expected = args.expected_digest.casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("--expected-digest must be a full SHA-256 digest")
    snapshot = resolve_model_snapshot(args.api_base, args.model_id)
    observed = str(snapshot["model_digest"]).casefold()
    if observed != expected:
        raise SystemExit(
            f"MODEL DRIFT: {args.model_id} expected {expected}, observed {observed}. "
            "Do not generate held-out responses."
        )
    print(f"Model digest verified: {args.model_id} @ {observed}")


if __name__ == "__main__":
    main()
