from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from coact_cps.security.external_redteam import extract_json_object, validate_task

if __package__:
    from scripts import generate_external_redteam_ollama as v9
    from scripts.v91_numeric_protocol import (
        NUMERIC_SERIALIZATION_PROTOCOL,
        UNCERTAINTY_ABSOLUTE_TOLERANCE,
        canonicalize_uncertainty_serialization,
    )
else:  # Direct execution: python scripts/<name>.py
    import generate_external_redteam_ollama as v9
    from v91_numeric_protocol import (
        NUMERIC_SERIALIZATION_PROTOCOL,
        UNCERTAINTY_ABSOLUTE_TOLERANCE,
        canonicalize_uncertainty_serialization,
    )

RUN_PROTOCOL_VERSION = "cps-truthframe-postfreeze-multimodel-v9.1"


def generation_config(args) -> dict[str, Any]:
    config = v9.generation_config(args)
    config["numeric_serialization"] = {
        "protocol_version": NUMERIC_SERIALIZATION_PROTOCOL,
        "field": "uncertainty",
        "absolute_tolerance": UNCERTAINTY_ABSOLUTE_TOLERANCE,
        "semantic_repairs_permitted": False,
    }
    return config


def main(argv: list[str] | None = None) -> None:
    args = v9.parser().parse_args(argv)
    if args.candidates_per_task < 1:
        raise ValueError("--candidates-per-task must be positive")
    if args.max_tasks < 0 or args.max_new_tokens < 1 or args.num_ctx < 1024:
        raise ValueError("Invalid task, token, or context limit")
    if args.retries < 0 or args.timeout_seconds <= 0:
        raise ValueError("Invalid retry or timeout setting")

    task_path = Path(args.tasks)
    tasks = v9.read_jsonl(task_path)
    for task in tasks:
        validate_task(task)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("No tasks selected")

    api_base = v9.normalize_api_base(args.api_base)
    snapshot = v9.resolve_model_snapshot(api_base, args.model_id)
    config = generation_config(args)
    task_hashes = {str(task["task_id"]): str(task["task_hash"]) for task in tasks}
    run_core = {
        "protocol_version": RUN_PROTOCOL_VERSION,
        "tasks_sha256": v9.file_sha256(task_path),
        "selected_task_ids": [str(task["task_id"]) for task in tasks],
        "model_digest": snapshot["model_digest"],
        "generation": config,
        "response_schema_sha256": v9.sha256_json(v9.EXPLANATION_SCHEMA),
    }
    run_fingerprint = v9.sha256_json(run_core)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = {
        **run_core,
        "run_fingerprint": run_fingerprint,
        "created_utc": v9.utc_now(),
        "tasks_path": str(task_path),
        "selected_task_count": len(tasks),
        "model_provenance": snapshot,
        "output_path": str(output),
        "frozen_v8_modified": False,
        "claim_boundary": (
            "Post-freeze external-attacker validation on the existing frozen tasks; "
            "this does not create a new untouched task split. Numeric canonicalization "
            "is restricted to uncertainty values equivalent within 1e-12."
        ),
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if existing_manifest.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Existing manifest {manifest_path} belongs to a different run. "
                "Use a new output filename."
            )
    elif output.exists():
        raise RuntimeError(
            f"Output {output} exists without its run manifest. Use a new output filename."
        )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    completed = v9.load_resume_records(
        output,
        run_fingerprint=run_fingerprint,
        task_hashes=task_hashes,
        model_digest=snapshot["model_digest"],
    )
    planned = len(tasks) * args.candidates_per_task
    print(
        f"Pinned {snapshot['model_name']} @ {snapshot['model_digest'][:12]} "
        f"({snapshot['model_details'].get('parameter_size', '?')}, "
        f"{snapshot['model_details'].get('quantization_level', '?')})"
    )
    print(f"Resuming with {len(completed)}/{planned} candidates already complete")

    completed_count = len(completed)
    try:
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            for task_number, task in enumerate(tasks, start=1):
                task_id = str(task["task_id"])
                for zero_index in range(args.candidates_per_task):
                    candidate_index = zero_index + 1
                    if (task_id, candidate_index) in completed:
                        continue
                    candidate_seed = args.seed + task_number * 1000 + zero_index
                    started = time.monotonic()
                    generated, api_response, attempts = v9.generate_candidate(
                        api_base,
                        model_id=args.model_id,
                        prompt=str(task["prompt"]),
                        seed=candidate_seed,
                        config=config,
                        keep_alive=-1,
                        timeout=args.timeout_seconds,
                        retries=args.retries,
                    )
                    elapsed = time.monotonic() - started
                    record: dict[str, Any] = {
                        "protocol_version": RUN_PROTOCOL_VERSION,
                        "run_fingerprint": run_fingerprint,
                        "task_id": task_id,
                        "task_hash": task["task_hash"],
                        "model_id": args.model_id,
                        "model_digest": snapshot["model_digest"],
                        "model_details": snapshot["model_details"],
                        "runtime": {
                            "name": snapshot["runtime"],
                            "version": snapshot["runtime_version"],
                        },
                        "candidate_id": (
                            f"{task_id}-{v9.model_slug(args.model_id)}-"
                            f"{snapshot['model_digest'][:12]}-c{candidate_index}"
                        ),
                        "candidate_index": candidate_index,
                        "seed": candidate_seed,
                        "decoding": config,
                        "generated_utc": v9.utc_now(),
                        "elapsed_seconds": elapsed,
                        "attempts": attempts,
                        "raw_model_output": generated,
                        "inference_metrics": {
                            key: api_response.get(key)
                            for key in (
                                "done_reason",
                                "total_duration",
                                "load_duration",
                                "prompt_eval_count",
                                "prompt_eval_duration",
                                "eval_count",
                                "eval_duration",
                            )
                        },
                    }
                    try:
                        raw_explanation = extract_json_object(generated)
                        normalized, audit = canonicalize_uncertainty_serialization(
                            raw_explanation,
                            task["clean_explanation"],
                        )
                        record["raw_explanation"] = raw_explanation
                        record["explanation"] = normalized
                        record["numeric_normalization"] = audit
                    except (ValueError, json.JSONDecodeError) as exc:
                        record["parse_error"] = str(exc)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed_count += 1
                    print(
                        f"Generated task {task_number}/{len(tasks)}, candidate "
                        f"{candidate_index}/{args.candidates_per_task} "
                        f"({completed_count}/{planned}; {elapsed:.1f}s)"
                    )
    except KeyboardInterrupt:
        print("Interrupted; completed JSONL records are safe and the same command resumes.")
        raise SystemExit(130) from None
    finally:
        if not args.keep_loaded:
            try:
                v9.unload_model(api_base, args.model_id)
                print(f"Unloaded {args.model_id} from memory")
            except v9.OllamaRequestError as exc:
                print(f"Warning: could not unload {args.model_id}: {exc}", file=sys.stderr)

    print(f"Saved {completed_count} candidates to {output.resolve()}")
    print(f"Run manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
