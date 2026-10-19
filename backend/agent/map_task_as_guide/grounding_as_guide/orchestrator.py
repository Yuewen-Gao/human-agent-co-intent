"""Parallel H/W diagnosis followed by data-informed R selection."""
from __future__ import annotations

import json
from dataclasses import replace
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
from services.llm_trace_logger import TraceContext, record_trace_result, trace_chat_completion


_PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts"
_CURRENT_GAME_STATE_MARKER = "\n\n<CURRENT GAME STATE>\n"

# GPT-5.6 Luna may use completion tokens for reasoning before emitting the
# required JSON. Leave enough budget for both the diagnosis and its payload.
JOINT_DIAGNOSIS_MAX_TOKENS = 4096
REPAIR_MAX_TOKENS = 2048


# The Follower's historical execute-or-ask contract remains a compatibility
# boundary, but its validation belongs to orchestration rather than a separate
# "clarification gate" policy module. It validates only the public payload;
# semantic ambiguity and route geometry are decided elsewhere.
_VALID_FOLLOWER_DECISIONS = frozenset({"execute", "ask_clarification"})


def _validated_follower_path_points(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        return None
    points: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            row, col = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return None
        points.append([row, col])
    return points


def _decision_text(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    return candidate.strip() if isinstance(candidate, str) else ""


def parse_follower_action_decision(raw: str) -> dict[str, Any]:
    """Validate the legacy Follower public action envelope.

    This is deliberately not a clarification policy: the model chooses its
    action, while the orchestrator admits only one well-formed public action.
    Route geometry and latest-message checks remain server-side at execution.
    """
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}

    decision = value.get("decision")
    if decision not in _VALID_FOLLOWER_DECISIONS:
        return {}

    if decision == "execute":
        path_points = _validated_follower_path_points(value.get("path_points"))
        if not path_points:
            return {}
        return {
            "decision": "execute",
            "source_message_id": _decision_text(value, "source_message_id"),
            "segment": _decision_text(value, "segment"),
            "completion_criterion": _decision_text(value, "completion_criterion"),
            "message": _decision_text(value, "message"),
            "path_points": path_points,
            "question": "",
        }

    question = _decision_text(value, "question")
    if not question:
        return {}
    return {
        "decision": "ask_clarification",
        "source_message_id": "",
        "segment": "",
        "completion_criterion": "",
        "message": "",
        "path_points": [],
        "question": question,
    }


def _joint_diagnosis_prompt(evidence: str) -> str:
    """Combine the canonical H/W rules with one non-conflicting response contract."""
    h_instruction = (_PROMPTS / "grounding_detection_prompt.txt").read_text(encoding="utf-8")
    w_instruction = (_PROMPTS / "grounding_object_prompt.txt").read_text(encoding="utf-8")
    return (
        "<H DETECTION INSTRUCTIONS>\n"
        + h_instruction
        + "\n</H DETECTION INSTRUCTIONS>\n\n<H CODEBOOK>\n"
        + definitions_for("H")
        + "\n</H CODEBOOK>\n\n<W DETECTION INSTRUCTIONS>\n"
        + w_instruction
        + "\n</W DETECTION INSTRUCTIONS>\n\n<W CODEBOOK>\n"
        + definitions_for("W")
        + "\n</W CODEBOOK>\n\n<H FEW-SHOT REFERENCE ONLY>\n"
        + "Examples are reference-only labeled patterns, not part of the current episode. "
        + "Do not treat an example utterance, landmark, route, or expected output as current evidence. "
        + "Do not copy example content into H, W, evidence, or repair-focus fields. "
        + "Classify only facts inside the later <CURRENT OBSERVABLE EVIDENCE> block.\n"
        + format_h_fewshots()
        + "\n</H FEW-SHOT REFERENCE ONLY>\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n"
        + "<JOINT H/W RESPONSE CONTRACT>\n"
        + "Apply the preceding H and W instructions independently to the same current evidence. "
        + "Their individual JSON descriptions specify the required contents of the `h` and `w` fields below; "
        + "return one JSON object only, not two separate objects.\n"
        + '{"h":{"h_code":"H1..H7","confidence":"low|medium|high",'
        + '"evidence":"short observable detection cue",'
        + '"trajectory_verdict":"correct|incorrect|insufficient_evidence|not_applicable",'
        + '"trajectory_assessment":{"fresh_trace_relation":"advances|completes|departs|not_assessable|not_applicable",'
        + '"observable_departure":"specific landmark-relative departure or empty"},'
        + '"next_guide_action":"repair|give_next_segment|normal_or_wait"},'
        + '"w":{"w_codes":["W1..W13"],"assessments":[{"w_code":"W1..W13",'
        + '"evidence":"short observable support",'
        + '"repair_focus":"neutral sentence describing what the reply should clarify"}]}}\n'
        + "</JOINT H/W RESPONSE CONTRACT>"
    )


def _call_json(llm_client: Any, prompt: str, max_tokens: int, *, trace_context: TraceContext | None = None, stage: str = "") -> tuple[str, Any | None]:
    messages = [{"role": "user", "content": prompt}]
    if trace_context is None:
        return llm_client.chat_completions_create(messages=messages, temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"}), None
    completion = trace_chat_completion(llm_client, replace(trace_context, stage=stage), messages=messages, temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"})
    return completion.response, completion


def _format_w_repair_context(w: dict[str, Any]) -> str:
    """Give R selected W definitions and neutral repair focus, never route facts."""
    codes = tuple(w.get("w_codes") or ())
    definitions = definitions_for_codes("W", codes)
    assessments = w.get("assessments") or ()
    return (
        "<W REPAIR CONTEXT>\n"
        + (definitions or "- no content-level W diagnosis\n")
        + "\n<CURRENT W ASSESSMENTS>\n"
        + json.dumps(assessments, ensure_ascii=False)
        + "\n</CURRENT W ASSESSMENTS>\n"
        + "</W REPAIR CONTEXT>"
    )


def _split_guide_background(evidence: str) -> tuple[str, str]:
    """Separate the rendered static Guide prompt from its current observations."""
    background, marker, observations = evidence.partition(_CURRENT_GAME_STATE_MARKER)
    if not marker:
        return evidence, ""
    return background, marker + observations


def _repair_prompt(
    evidence: str,
    h: dict[str, Any],
    w: dict[str, Any],
    retrieved: tuple[dict[str, Any], ...],
) -> str:
    """Build the R request with fixed Guide rules before repair-specific context."""
    guide_background, observable_evidence = _split_guide_background(evidence)
    selected_h_definition = definitions_for_codes("H", (str(h.get("h_code") or ""),))
    repair_instruction = (_PROMPTS / "grounding_repair_prompt.txt").read_text(encoding="utf-8")
    return (
        "<MAP TASK GUIDE BACKGROUND>\n"
        + guide_background
        + "\n</MAP TASK GUIDE BACKGROUND>\n\n<REPAIR INSTRUCTIONS>\n"
        + repair_instruction
        + "\n</REPAIR INSTRUCTIONS>\n\n<SELECTED H DEFINITION>\n"
        + (selected_h_definition or "- no selected H definition\n")
        + "\n</SELECTED H DEFINITION>\n<R CODEBOOK>\n"
        + definitions_for("R")
        + "\n</R CODEBOOK>\n<DETECTED H>\n"
        + json.dumps(h, ensure_ascii=False)
        + "\n</DETECTED H>\n"
        + _format_w_repair_context(w)
        + "\n"
        + format_repair_examples(retrieved)
        + "\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + observable_evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>"
    )


def safe_decide(orchestrator: Any, evidence: str, *, trace_context: TraceContext | None = None) -> dict[str, Any] | None:
    """Return no grounding decision when diagnosis cannot complete.

    The caller can then use the normal Guide reply path instead of dropping a
    participant-visible turn because a treatment-only LLM call failed.
    """
    try:
        return orchestrator.decide(evidence) if trace_context is None else orchestrator.decide(evidence, trace_context=trace_context)
    except Exception:
        return None


class GroundingOrchestrator:
    def __init__(self, llm_client: Any):
        self._llm = llm_client

    def decide(self, evidence: str, *, trace_context: TraceContext | None = None) -> dict[str, Any]:
        """Jointly diagnose H/W, then select and realize repairs in one LLM call."""
        diagnosis_prompt = _joint_diagnosis_prompt(evidence)
        diagnosis_raw, diagnosis_trace = _call_json(
            self._llm,
            diagnosis_prompt,
            JOINT_DIAGNOSIS_MAX_TOKENS,
            trace_context=trace_context,
            stage="grounding.h_w",
        )
        diagnosis = parse_joint_diagnosis(diagnosis_raw)
        if diagnosis_trace:
            record_trace_result(diagnosis_trace, diagnosis)
        h = diagnosis["h"]
        w = diagnosis["w"]

        w_codes = w.get("w_codes", ())
        # H1-H6 are already operational repair triggers. W enriches repair
        # selection/retrieval, but an empty W diagnosis must not suppress a
        # concrete, observable H diagnosis.
        if not h or h.get("h_code") == "H7":
            return self._no_repair(h, w, w_codes)

        retrieved = retrieve_repair_examples(
            evidence,
            h["h_code"],
            w_codes,
            associated_repair_codes(h["h_code"], w_codes),
        )
        repair_prompt = _repair_prompt(evidence, h, w, retrieved)
        repair_raw, repair_trace = _call_json(
            self._llm,
            repair_prompt,
            REPAIR_MAX_TOKENS,
            trace_context=trace_context,
            stage="grounding.repair",
        )
        repair = parse_repair(repair_raw)
        if not repair:
            if repair_trace:
                record_trace_result(repair_trace, repair, error="invalid_or_empty_repair")
            return self._no_repair(h, w, w_codes)
        r_codes = repair["applied_r_codes"]
        deferred_r_codes = repair["deferred_r_codes"]
        decision = {
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
        if repair_trace:
            decision["_trace_repair_completion"] = repair_trace
            if not str(repair.get("reply") or "").strip():
                record_trace_result(repair_trace, repair)
        return decision

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
