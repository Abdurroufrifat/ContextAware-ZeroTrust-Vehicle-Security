from __future__ import annotations

import json
import re
from typing import Any

from coact_cps.security.operator_surface import canonical_operator_surface
from coact_cps.types import EvidenceBundle, GeneratedExplanation


class GroundedExplanationGenerator:
    """Generate an operator explanation from a closed evidence bundle.

    Template mode is deterministic and runs on any computer. Hugging Face mode is
    optional and loads a local/open model only when explicitly requested.
    """

    def __init__(
        self,
        mode: str = "template",
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens: int = 420,
    ):
        if mode not in {"template", "hf"}:
            raise ValueError("mode must be 'template' or 'hf'")
        self.mode = mode
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipeline: Any | None = None

    def generate(self, bundle: EvidenceBundle) -> GeneratedExplanation:
        if self.mode == "template":
            return self._template_generate(bundle)
        return self._hf_generate(bundle)

    def _template_generate(self, bundle: EvidenceBundle) -> GeneratedExplanation:
        surface = canonical_operator_surface(bundle)
        result = GeneratedExplanation(
            summary=surface.summary,
            probable_cause=bundle.predicted_cause,
            evidence_ids=list(surface.evidence_ids),
            alternatives=list(surface.alternatives),
            uncertainty=float(bundle.uncertainty),
            operator_question=surface.operator_question,
            recommendation=surface.recommendation,
        )
        result.raw_text = json.dumps(result.to_dict(), ensure_ascii=False)
        return result

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face mode requires requirements-llm.txt. "
                "Install it or use --generator template."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._pipeline

    def build_prompt(self, bundle: EvidenceBundle) -> str:
        schema = {
            "summary": "brief operator-facing incident summary",
            "probable_cause": "one cause name from cause_probabilities",
            "evidence_ids": ["only IDs present in evidence"],
            "alternatives": ["zero to two causes from cause_probabilities"],
            "uncertainty": "number from 0 to 1 consistent with supplied uncertainty",
            "operator_question": "one question that requests useful operational context",
            "recommendation": "one item from permitted_actions",
        }
        return (
            "You are a cautious industrial incident-diagnosis assistant. "
            "Use only the supplied evidence. Do not invent telemetry, causes, times, "
            "actions, or certainty. The human operator makes the final decision. "
            "Return one JSON object and no additional text.\n\n"
            f"Required schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Evidence bundle:\n{json.dumps(bundle.to_dict(), indent=2)}"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError("The language model did not return a JSON object") from exc
            return json.loads(match.group(0))

    def _hf_generate(self, bundle: EvidenceBundle) -> GeneratedExplanation:
        generator = self._load_pipeline()
        result = generator(
            self.build_prompt(bundle),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]
        parsed = self._extract_json(result)
        explanation = GeneratedExplanation(
            summary=str(parsed.get("summary", "")),
            probable_cause=str(parsed.get("probable_cause", "")),
            evidence_ids=[str(value) for value in parsed.get("evidence_ids", [])],
            alternatives=[str(value) for value in parsed.get("alternatives", [])],
            uncertainty=float(parsed.get("uncertainty", 1.0)),
            operator_question=str(parsed.get("operator_question", "")),
            recommendation=str(parsed.get("recommendation", "")),
            raw_text=result,
        )
        surface = canonical_operator_surface(bundle)
        explanation.summary = surface.summary
        explanation.evidence_ids = list(surface.evidence_ids)
        explanation.alternatives = list(surface.alternatives)
        explanation.operator_question = surface.operator_question
        explanation.recommendation = surface.recommendation
        return explanation
