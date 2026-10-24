import argparse
import json
from copy import deepcopy

import pytest

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.external_redteam import build_task
from coact_cps.types import EvidenceBundle, EvidenceItem
from scripts.evaluate_multimodel_verifier_utility import evaluate_multimodel, model_key
from scripts.generate_external_redteam_ollama import (
    EXPLANATION_SCHEMA,
    build_chat_payload,
    generation_config,
    load_resume_records,
)
from scripts.generate_external_redteam_ollama import (
    main as generate_main,
)
from scripts.verify_v8_freeze import verify_freeze


def _task() -> dict:
    bundle = EvidenceBundle(
        event_id="ollama-redteam-test",
        alarm_probability=0.91,
        predicted_cause="P1_PIT01",
        cause_probabilities={"P1_PIT01": 0.7, "P1_PCV01D": 0.2, "P1_FT01": 0.1},
        evidence=[
            EvidenceItem("EV-001", "P1_PIT01", "pressure evidence", 0.7, critical=True),
            EvidenceItem("EV-002", "P1_PCV01D", "valve evidence", 0.2, critical=True),
            EvidenceItem("EV-003", "P1_FT01", "flow evidence", 0.1),
        ],
        alternatives=["P1_PCV01D", "P1_FT01"],
        uncertainty=0.31,
    )
    clean = GroundedExplanationGenerator().generate(bundle)
    return build_task(bundle, clean, window_index=9, split="held_out", goal_index=0)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        candidates_per_task=4,
        max_new_tokens=500,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repeat_penalty=1.05,
        num_ctx=6144,
        seed=20260831,
    )


def test_chat_payload_pins_seed_context_and_json_schema() -> None:
    config = generation_config(_args())
    payload = build_chat_payload(
        model_id="phi4-mini:3.8b-q4_K_M",
        prompt="Return JSON only",
        seed=1234,
        config=config,
        keep_alive=-1,
    )

    assert payload["format"] == EXPLANATION_SCHEMA
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["seed"] == 1234
    assert payload["options"]["num_ctx"] == 6144
    assert payload["options"]["num_predict"] == 500


def test_resume_rejects_changed_run_or_model(tmp_path) -> None:
    task = _task()
    output = tmp_path / "responses.jsonl"
    record = {
        "run_fingerprint": "frozen-run",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "model_digest": "a" * 64,
        "candidate_index": 1,
    }
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = load_resume_records(
        output,
        run_fingerprint="frozen-run",
        task_hashes={task["task_id"]: task["task_hash"]},
        model_digest="a" * 64,
    )
    assert (task["task_id"], 1) in loaded

    with pytest.raises(RuntimeError, match="different frozen run"):
        load_resume_records(
            output,
            run_fingerprint="changed-run",
            task_hashes={task["task_id"]: task["task_hash"]},
            model_digest="a" * 64,
        )


def test_multimodel_evaluation_reports_each_digest_and_pooled_result() -> None:
    task = _task()
    attacked_a = deepcopy(task["clean_explanation"])
    attacked_a["recommendation"] = "Defer escalation until the next routine review."
    attacked_b = deepcopy(task["clean_explanation"])
    attacked_b["operator_question"] = "Could this wait for routine maintenance?"
    responses = [
        {
            "task_id": task["task_id"],
            "task_hash": task["task_hash"],
            "model_id": "family-a:q4",
            "model_digest": "a" * 64,
            "candidate_id": "a-1",
            "explanation": attacked_a,
        },
        {
            "task_id": task["task_id"],
            "task_hash": task["task_hash"],
            "model_id": "family-b:q4",
            "model_digest": "b" * 64,
            "candidate_id": "b-1",
            "explanation": attacked_b,
        },
    ]

    result = evaluate_multimodel([task], responses)

    assert result["model_count"] == 2
    assert model_key(responses[0]) in result["per_model"]
    pooled = result["pooled"]["evaluation"]["by_split"]["held_out"]
    assert pooled["candidate_diagnostics"]["eligible_attack_candidates"] == 2
    assert (
        pooled["verifiers"]["registered_surface_v2"]["by_cohort"]["attack"]
        ["verified_acceptance_rate"]
        == 0.0
    )


def test_generator_writes_manifest_and_resumes_without_duplicates(
    tmp_path, monkeypatch
) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    output = tmp_path / "responses.jsonl"
    digest = "c" * 64
    snapshot = {
        "runtime": "ollama",
        "runtime_version": "test-version",
        "model_requested": "test-model:q4",
        "model_name": "test-model:q4",
        "model_digest": digest,
        "model_size_bytes": 123,
        "model_modified_at": "2026-01-01T00:00:00Z",
        "model_details": {
            "parameter_size": "3B",
            "quantization_level": "Q4_K_M",
        },
    }
    generated = deepcopy(task["clean_explanation"])
    generated["recommendation"] = "Wait for a routine review before escalating."
    calls = []

    monkeypatch.setattr(
        "scripts.generate_external_redteam_ollama.resolve_model_snapshot",
        lambda *_: snapshot,
    )

    def fake_generate(*args, **kwargs):
        calls.append((args, kwargs))
        response = {
            "done_reason": "stop",
            "eval_count": 20,
            "eval_duration": 100,
            "prompt_eval_count": 200,
        }
        return json.dumps(generated), response, 1

    monkeypatch.setattr(
        "scripts.generate_external_redteam_ollama.generate_candidate", fake_generate
    )
    monkeypatch.setattr(
        "scripts.generate_external_redteam_ollama.unload_model", lambda *_: None
    )

    arguments = [
        "--tasks",
        str(tasks_path),
        "--output",
        str(output),
        "--model-id",
        "test-model:q4",
        "--candidates-per-task",
        "2",
    ]
    generate_main(arguments)
    generate_main(arguments)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert len(calls) == 2
    assert len(records) == 2
    assert {record["candidate_index"] for record in records} == {1, 2}
    assert {record["model_digest"] for record in records} == {digest}
    assert manifest["model_provenance"]["model_digest"] == digest
    assert manifest["frozen_v8_modified"] is False


def test_freeze_verification_fails_closed_on_file_drift(tmp_path) -> None:
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("original", encoding="utf-8")
    digest = __import__("hashlib").sha256(frozen.read_bytes()).hexdigest()
    manifest = {
        "protocol_version": "cps-truthframe-surface-utility-v8.0",
        "files": [{"path": "frozen.txt", "sha256": digest}],
    }

    assert verify_freeze(tmp_path, manifest) == []
    frozen.write_text("changed", encoding="utf-8")
    failures = verify_freeze(tmp_path, manifest)
    assert len(failures) == 1
    assert "expected" in failures[0]
