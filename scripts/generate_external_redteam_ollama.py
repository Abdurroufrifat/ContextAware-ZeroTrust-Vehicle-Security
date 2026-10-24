from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from coact_cps.security.external_redteam import extract_json_object, validate_task

RUN_PROTOCOL_VERSION = "cps-truthframe-postfreeze-multimodel-v9.0"

EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "probable_cause": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "number"},
        "operator_question": {"type": "string"},
        "recommendation": {"type": "string"},
    },
    "required": [
        "summary",
        "probable_cause",
        "evidence_ids",
        "alternatives",
        "uncertainty",
        "operator_question",
        "recommendation",
    ],
    "additionalProperties": False,
}


class OllamaRequestError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def normalize_api_base(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise ValueError("--api-base cannot be empty")
    if not base.endswith("/api"):
        base = f"{base}/api"
    return base


def request_json(
    api_base: str,
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = f"{normalize_api_base(api_base)}/{endpoint.lstrip('/')}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local URL only
            decoded = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaRequestError(f"Ollama HTTP {exc.code} at {url}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaRequestError(
            f"Cannot reach Ollama at {url}. Start Ollama and retry: {exc}"
        ) from exc
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise OllamaRequestError(f"Ollama returned invalid JSON at {url}") from exc
    if not isinstance(parsed, dict):
        raise OllamaRequestError(f"Ollama returned a non-object response at {url}")
    if parsed.get("error"):
        raise OllamaRequestError(f"Ollama error at {url}: {parsed['error']}")
    return parsed


def resolve_model_snapshot(api_base: str, model_id: str) -> dict[str, Any]:
    version = request_json(api_base, "version")
    tags = request_json(api_base, "tags")
    models = tags.get("models", [])
    if not isinstance(models, list):
        raise OllamaRequestError("Ollama /api/tags response has no model list")
    match = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and model_id in {str(item.get("name", "")), str(item.get("model", ""))}
        ),
        None,
    )
    if match is None:
        installed = sorted(
            str(item.get("name", "")) for item in models if isinstance(item, dict)
        )
        suffix = f" Installed models: {', '.join(installed)}" if installed else ""
        raise OllamaRequestError(
            f"Model '{model_id}' is not installed. Run: ollama pull {model_id}.{suffix}"
        )
    digest = str(match.get("digest", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise OllamaRequestError(
            f"Ollama did not report a full SHA-256 digest for model '{model_id}'"
        )
    details = match.get("details") if isinstance(match.get("details"), dict) else {}
    return {
        "runtime": "ollama",
        "runtime_version": str(version.get("version", "unreported")),
        "model_requested": model_id,
        "model_name": str(match.get("name", model_id)),
        "model_digest": digest,
        "model_size_bytes": int(match.get("size", 0)),
        "model_modified_at": str(match.get("modified_at", "")),
        "model_details": details,
    }


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "candidates_per_task": args.candidates_per_task,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
        "structured_output": True,
        "thinking": False,
    }


def build_chat_payload(
    *,
    model_id: str,
    prompt: str,
    seed: int,
    config: dict[str, Any],
    keep_alive: str | int,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "format": EXPLANATION_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "seed": seed,
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "repeat_penalty": config["repeat_penalty"],
            "num_predict": config["max_new_tokens"],
            "num_ctx": config["num_ctx"],
        },
    }


def generate_candidate(
    api_base: str,
    *,
    model_id: str,
    prompt: str,
    seed: int,
    config: dict[str, Any],
    keep_alive: str | int,
    timeout: float,
    retries: int,
) -> tuple[str, dict[str, Any], int]:
    payload = build_chat_payload(
        model_id=model_id,
        prompt=prompt,
        seed=seed,
        config=config,
        keep_alive=keep_alive,
    )
    last_error: OllamaRequestError | None = None
    for attempt in range(1, retries + 2):
        try:
            response = request_json(api_base, "chat", payload=payload, timeout=timeout)
            message = response.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise OllamaRequestError("Ollama chat response has no assistant content")
            return message["content"].strip(), response, attempt
        except OllamaRequestError as exc:
            last_error = exc
            if attempt > retries:
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def unload_model(api_base: str, model_id: str) -> None:
    request_json(
        api_base,
        "generate",
        payload={"model": model_id, "keep_alive": 0},
        timeout=120.0,
    )


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.casefold()).strip("-")


def load_resume_records(
    output: Path,
    *,
    run_fingerprint: str,
    task_hashes: dict[str, str],
    model_digest: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not output.exists():
        return completed
    for record in read_jsonl(output):
        if record.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Existing output {output} was created by a different frozen run. "
                "Use a new output filename."
            )
        task_id = str(record.get("task_id", ""))
        if task_id not in task_hashes or record.get("task_hash") != task_hashes[task_id]:
            raise RuntimeError(f"Existing output contains an unknown or changed task: {task_id}")
        if record.get("model_digest") != model_digest:
            raise RuntimeError("Existing output model digest differs from the installed model")
        candidate_index = record.get("candidate_index")
        if not isinstance(candidate_index, int) or candidate_index < 1:
            raise RuntimeError("Existing output lacks a valid candidate_index")
        key = (task_id, candidate_index)
        if key in completed:
            raise RuntimeError(f"Duplicate response record in existing output: {key}")
        completed[key] = record
    return completed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Generate resumable, digest-pinned external red-team candidates through a "
            "local Ollama model without modifying the frozen V8 verifier"
        )
    )
    result.add_argument("--tasks", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--model-id", required=True)
    result.add_argument("--candidates-per-task", type=int, default=4)
    result.add_argument("--max-tasks", type=int, default=0)
    result.add_argument("--max-new-tokens", type=int, default=500)
    result.add_argument("--temperature", type=float, default=0.9)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--top-k", type=int, default=50)
    result.add_argument("--repeat-penalty", type=float, default=1.05)
    result.add_argument("--num-ctx", type=int, default=6144)
    result.add_argument("--seed", type=int, default=20260831)
    result.add_argument("--api-base", default="http://127.0.0.1:11434/api")
    result.add_argument("--timeout-seconds", type=float, default=3600.0)
    result.add_argument("--retries", type=int, default=2)
    result.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Leave the model loaded after the run; default is to unload it",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.candidates_per_task < 1:
        raise ValueError("--candidates-per-task must be positive")
    if args.max_tasks < 0 or args.max_new_tokens < 1 or args.num_ctx < 1024:
        raise ValueError("Invalid task, token, or context limit")
    if args.retries < 0 or args.timeout_seconds <= 0:
        raise ValueError("Invalid retry or timeout setting")

    task_path = Path(args.tasks)
    tasks = read_jsonl(task_path)
    for task in tasks:
        validate_task(task)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("No tasks selected")

    api_base = normalize_api_base(args.api_base)
    snapshot = resolve_model_snapshot(api_base, args.model_id)
    config = generation_config(args)
    task_hashes = {str(task["task_id"]): str(task["task_hash"]) for task in tasks}
    run_core = {
        "protocol_version": RUN_PROTOCOL_VERSION,
        "tasks_sha256": file_sha256(task_path),
        "selected_task_ids": [str(task["task_id"]) for task in tasks],
        "model_digest": snapshot["model_digest"],
        "generation": config,
        "response_schema_sha256": sha256_json(EXPLANATION_SCHEMA),
    }
    run_fingerprint = sha256_json(run_core)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = {
        **run_core,
        "run_fingerprint": run_fingerprint,
        "created_utc": utc_now(),
        "tasks_path": str(task_path),
        "selected_task_count": len(tasks),
        "model_provenance": snapshot,
        "output_path": str(output),
        "frozen_v8_modified": False,
        "claim_boundary": (
            "Post-freeze external-attacker validation on the existing frozen tasks; this "
            "does not create a new untouched task split."
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

    completed = load_resume_records(
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

    keep_alive: str | int = -1
    completed_count = len(completed)
    try:
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            for task_number, task in enumerate(tasks, start=1):
                task_id = str(task["task_id"])
                for zero_index in range(args.candidates_per_task):
                    candidate_index = zero_index + 1
                    key = (task_id, candidate_index)
                    if key in completed:
                        continue
                    candidate_seed = args.seed + task_number * 1000 + zero_index
                    started = time.monotonic()
                    generated, api_response, attempts = generate_candidate(
                        api_base,
                        model_id=args.model_id,
                        prompt=str(task["prompt"]),
                        seed=candidate_seed,
                        config=config,
                        keep_alive=keep_alive,
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
                            f"{task_id}-{model_slug(args.model_id)}-"
                            f"{snapshot['model_digest'][:12]}-c{candidate_index}"
                        ),
                        "candidate_index": candidate_index,
                        "seed": candidate_seed,
                        "decoding": config,
                        "generated_utc": utc_now(),
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
                        record["explanation"] = extract_json_object(generated)
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
                unload_model(api_base, args.model_id)
                print(f"Unloaded {args.model_id} from memory")
            except OllamaRequestError as exc:
                print(f"Warning: could not unload {args.model_id}: {exc}", file=sys.stderr)

    print(f"Saved {completed_count} candidates to {output.resolve()}")
    print(f"Run manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
