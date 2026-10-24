from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coact_cps.security.external_redteam import extract_json_object, validate_task


def read_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate external LLM red-team candidates from frozen JSONL tasks"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    if args.candidates_per_task < 1:
        raise ValueError("--candidates-per-task must be positive")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-redteam.txt before running this script") from exc

    tasks = read_jsonl(args.tasks)
    for task in tasks:
        validate_task(task)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype="auto", device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(device)
    model.eval()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with destination.open("w", encoding="utf-8") as handle:
        for task_number, task in enumerate(tasks, start=1):
            for candidate_number in range(args.candidates_per_task):
                candidate_seed = args.seed + task_number * 1000 + candidate_number
                torch.manual_seed(candidate_seed)
                messages = [{"role": "user", "content": task["prompt"]}]
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    rendered = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                inputs = tokenizer(rendered, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=50,
                        repetition_penalty=1.05,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(
                    output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                record = {
                    "task_id": task["task_id"],
                    "task_hash": task["task_hash"],
                    "model_id": args.model_id,
                    "candidate_id": f"{task['task_id']}-c{candidate_number + 1}",
                    "seed": candidate_seed,
                    "decoding": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": 50,
                    },
                    "raw_model_output": generated,
                }
                try:
                    record["explanation"] = extract_json_object(generated)
                except (ValueError, json.JSONDecodeError) as exc:
                    record["parse_error"] = str(exc)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
            print(f"Generated task {task_number}/{len(tasks)} ({completed} candidates)")
    print(f"Saved {completed} external-model candidates to {destination.resolve()}")


if __name__ == "__main__":
    main()
