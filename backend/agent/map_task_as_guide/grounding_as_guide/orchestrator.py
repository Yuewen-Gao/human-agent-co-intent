"""Parallel H/W diagnosis followed by data-informed R selection."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .codebook import definitions_for
from .contract import parse_h, parse_repair, parse_repair_plan, parse_w
from .fewshot_repository import (
    format_h_fewshots,
    format_repair_examples,
    retrieve_repair_examples,
)
from .grounding_rules import (
    associated_repair_codes,
    update_mode,
)


_PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts"


def _call_json(llm_client: Any, prompt: str, max_tokens: int) -> str:
    return llm_client.chat_completions_create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


def safe_decide(orchestrator: Any, evidence: str) -> dict[str, Any] | None:
    """Return no grounding decision when diagnosis cannot complete.

    The caller can then use the normal Guide reply path instead of dropping a
    participant-visible turn because a treatment-only LLM call failed.
    """
    try:
        return orchestrator.decide(evidence)
    except Exception:
        return None


class GroundingOrchestrator:
    def __init__(self, llm_client: Any):
        self._llm = llm_client

    def decide(self, evidence: str) -> dict[str, Any]:
        """Diagnose H/W, select an auditable plan, then draft from current facts."""
        h_prompt = (
            (_PROMPTS / "grounding_detection_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<H CODEBOOK>\n"
            + definitions_for("H")
            + "\n</H CODEBOOK>\n"
            + format_h_fewshots()
            + "\n<CURRENT OBSERVABLE EVIDENCE>\n"
            + evidence
            + "\n</CURRENT OBSERVABLE EVIDENCE>"
        )
        w_prompt = (
            (_PROMPTS / "grounding_object_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<W CODEBOOK>\n"
            + definitions_for("W")
            + "\n</W CODEBOOK>\nEvidence:\n"
            + evidence
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            h_future = executor.submit(_call_json, self._llm, h_prompt, 400)
            w_future = executor.submit(_call_json, self._llm, w_prompt, 700)
            h = parse_h(h_future.result())
            w = parse_w(w_future.result())

        w_codes = w.get("w_codes", ())
        if not h or h.get("h_code") == "H7" or not w_codes:
            return self._no_repair(h, w, w_codes)

        retrieved = retrieve_repair_examples(
            evidence, associated_repair_codes(h["h_code"], w_codes)
        )
        plan_prompt = (
            (_PROMPTS / "grounding_repair_plan_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<H CODEBOOK>\n"
            + definitions_for("H")
            + "\n</H CODEBOOK>\n<W CODEBOOK>\n"
            + definitions_for("W")
            + "\n</W CODEBOOK>\n<R CODEBOOK>\n"
            + definitions_for("R")
            + "\n</R CODEBOOK>\n<DETECTED H>\n"
            + json.dumps(h, ensure_ascii=False)
            + "\n</DETECTED H>\n<DIAGNOSED W>\n"
            + json.dumps(w, ensure_ascii=False)
            + "\n</DIAGNOSED W>\n"
            + format_repair_examples(retrieved)
            + "\n<BASE GUIDE PROMPT>\n"
            + evidence
            + "\n</BASE GUIDE PROMPT>"
        )
        plan = parse_repair_plan(_call_json(self._llm, plan_prompt, 1000))
        if not plan:
            return self._no_repair(h, w, w_codes)

        retrieved_by_id = {example["example_id"]: example for example in retrieved}
        selected_example_ids = {
            example_id
            for step in plan["immediate_steps"]
            for example_id in step["semantic_example_ids"]
            if example_id in retrieved_by_id
        }
        selected_examples = tuple(
            example for example in retrieved if example["example_id"] in selected_example_ids
        )
        draft_prompt = (
            (_PROMPTS / "grounding_repair_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<R CODEBOOK>\n"
            + definitions_for("R")
            + "\n</R CODEBOOK>\n<SELECTED REPAIR PLAN>\n"
            + json.dumps(plan, ensure_ascii=False)
            + "\n</SELECTED REPAIR PLAN>\n"
            + format_repair_examples(selected_examples)
            + "\n<BASE GUIDE PROMPT>\n"
            + evidence
            + "\n</BASE GUIDE PROMPT>"
        )
        repair = parse_repair(_call_json(self._llm, draft_prompt, 700))
        if not repair:
            return self._no_repair(h, w, w_codes, plan)
        r_codes = tuple(step["r_code"] for step in plan["immediate_steps"])
        deferred_r_codes = tuple(step["r_code"] for step in plan["deferred_steps"])
        return {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "repair_plan": plan,
            "r": r_codes[0],
            "r_codes": r_codes,
            "deferred_r_codes": deferred_r_codes,
            "m": update_mode(r_codes[0], adds_missing_detail=repair["adds_missing_detail"]),
            "m_codes": tuple(
                update_mode(r_code, adds_missing_detail=repair["adds_missing_detail"])
                for r_code in r_codes
            ),
            "reply": repair["reply"],
            "selection_evidence": repair["selection_evidence"],
        }

    @staticmethod
    def _no_repair(
        h: dict[str, Any],
        w: dict[str, Any],
        w_codes: tuple[str, ...],
        repair_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "repair_plan": repair_plan or {},
            "r": None,
            "r_codes": (),
            "deferred_r_codes": (),
            "m": "M5",
            "m_codes": ("M5",),
            "reply": "",
            "needs_base_reply": True,
        }
