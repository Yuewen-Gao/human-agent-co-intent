"""Parallel H/W diagnosis followed by data-informed R selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .codebook import definitions_for, definitions_for_codes
from .contract import parse_joint_diagnosis, parse_repair
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


def _format_w_repair_context(w: dict[str, Any]) -> str:
    """Give R only the selected W definitions and their current propositions."""
    codes = tuple(w.get("w_codes") or ())
    definitions = definitions_for_codes("W", codes)
    assessments = w.get("assessments") or ()
    return (
        "<W REPAIR CONTEXT>\n"
        + (definitions or "- no content-level W diagnosis\n")
        + "\n<CURRENT W PROPOSITIONS>\n"
        + json.dumps(assessments, ensure_ascii=False)
        + "\n</CURRENT W PROPOSITIONS>\n"
        + "</W REPAIR CONTEXT>"
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
        """Jointly diagnose H/W, then select and realize repairs in one LLM call."""
        diagnosis_prompt = (
            (_PROMPTS / "grounding_joint_diagnosis_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<H CODEBOOK>\n"
            + definitions_for("H")
            + "\n</H CODEBOOK>\n"
            + format_h_fewshots()
            + "\n\n<W CODEBOOK>\n"
            + definitions_for("W")
            + "\n</W CODEBOOK>\n<CURRENT OBSERVABLE EVIDENCE>\n"
            + evidence
            + "\n</CURRENT OBSERVABLE EVIDENCE>"
        )
        diagnosis = parse_joint_diagnosis(
            _call_json(self._llm, diagnosis_prompt, 900)
        )
        h = diagnosis["h"]
        w = diagnosis["w"]

        w_codes = w.get("w_codes", ())
        # H1-H6 are already operational repair triggers. W enriches repair
        # selection/retrieval, but an empty W diagnosis must not suppress a
        # concrete, observable H diagnosis.
        if not h or h.get("h_code") == "H7":
            decision = self._no_repair(h, w, w_codes)
            if (
                h
                and "<AGENT TRIGGER>\nfollower_trajectory_changed\n</AGENT TRIGGER>" in evidence
                and h.get("next_guide_action") != "give_next_segment"
            ):
                # A correct-but-incomplete or unassessable drawing leaves the
                # existing instruction in force; do not ask the base Guide to
                # generate a new, potentially repeated instruction.
                decision["suppress_base_reply"] = True
            return decision

        retrieved = retrieve_repair_examples(
            evidence, associated_repair_codes(h["h_code"], w_codes)
        )
        repair_prompt = (
            (_PROMPTS / "grounding_repair_prompt.txt").read_text(encoding="utf-8")
            + "\n\n<H CODEBOOK>\n"
            + definitions_for("H")
            + "\n</H CODEBOOK>\n<R CODEBOOK>\n"
            + definitions_for("R")
            + "\n</R CODEBOOK>\n<DETECTED H>\n"
            + json.dumps(h, ensure_ascii=False)
            + "\n</DETECTED H>\n"
            + _format_w_repair_context(w)
            + "\n"
            + format_repair_examples(retrieved)
            + "\n<BASE GUIDE PROMPT>\n"
            + evidence
            + "\n</BASE GUIDE PROMPT>"
        )
        repair = parse_repair(_call_json(self._llm, repair_prompt, 800))
        if not repair:
            return self._no_repair(h, w, w_codes)
        r_codes = repair["applied_r_codes"]
        deferred_r_codes = repair["deferred_r_codes"]
        return {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "repair_steps": repair["repair_steps"],
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
            "needs_base_reply": False,
            "suppress_base_reply": False,
        }

    @staticmethod
    def _no_repair(
        h: dict[str, Any],
        w: dict[str, Any],
        w_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "repair_steps": (),
            "r": None,
            "r_codes": (),
            "deferred_r_codes": (),
            "m": "M5",
            "m_codes": ("M5",),
            "reply": "",
            "needs_base_reply": True,
            "normal_guide_action": (
                h.get("next_guide_action", "normal_or_wait") if h else "normal_or_wait"
            ),
            "suppress_base_reply": False,
        }
