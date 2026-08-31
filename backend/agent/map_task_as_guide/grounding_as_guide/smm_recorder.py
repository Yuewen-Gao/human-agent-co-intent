"""Independent, auditable SMM/annotation recorder for Guide-Agent turns."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .codebook import definitions_for
from .mental_model_update import MENTAL_MODEL_FIELD_KEYS

_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts" / "smm_recorder_prompt.txt"
_ANNOTATION_KEYS = ("explanation_transcription", "task_model_q1", "partner_model_q2", "self_model_q3", "alignment_q4")


def parse_recorder_output(raw: str, revision: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        value = {}
        valid = False
    else:
        valid = isinstance(value, dict)
    patch_changes = value.get("changes") if isinstance(value, dict) else {}
    changes: dict[str, dict[str, str]] = {}
    if isinstance(patch_changes, Mapping):
        for key in MENTAL_MODEL_FIELD_KEYS:
            candidate = patch_changes.get(key)
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("value"), str):
                continue
            text = candidate["value"].strip()
            if text:
                changes[key] = {"value": text, "confidence": str(candidate.get("confidence") or "low")}
    raw_annotation = value.get("annotation") if isinstance(value, dict) else {}
    annotation = {
        key: str(raw_annotation.get(key, "not_observable")).strip() if isinstance(raw_annotation, Mapping) else "not_observable"
        for key in _ANNOTATION_KEYS
    }
    return {"base_revision": revision, "changes": changes, "annotation": annotation, "valid": valid}


def record_prompt(evidence: str, model: Mapping[str, Any], final_reply: str) -> str:
    instruction = _PROMPT.read_text(encoding="utf-8").strip()
    return (
        f"{instruction}\n\n<W CODEBOOK>\n{definitions_for('W')}\n</W CODEBOOK>\n"
        f"<CURRENT SMM>\n{json.dumps(model, ensure_ascii=False)}\n</CURRENT SMM>\n"
        f"<FINAL GUIDE REPLY>\n{final_reply}\n</FINAL GUIDE REPLY>\n"
        f"<NEUTRAL VISIBLE EVIDENCE>\n{evidence}\n</NEUTRAL VISIBLE EVIDENCE>"
    )


def assess(llm_client: Any, evidence: str, model: Mapping[str, Any], final_reply: str) -> dict[str, Any]:
    response = llm_client.chat_completions_create(
        messages=[{"role": "user", "content": record_prompt(evidence, model, final_reply)}],
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return parse_recorder_output(response, int(model.get("revision", 0) or 0))
