"""Independent, auditable SMM/annotation recorder for Guide-Agent turns."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .codebook import definitions_for
from .mental_model_update import MENTAL_MODEL_FIELD_KEYS

_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts" / "smm_recorder_prompt.txt"
_ANNOTATION_KEYS = ("explanation_transcription", "task_model_q1", "partner_model_q2", "self_model_q3", "alignment_q4")


def parse_recorder_output(raw: str, revision: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        value = {}
        valid = False
        parse_error = str(error)
    else:
        valid = isinstance(value, dict)
        parse_error = None if valid else "Recorder response was valid JSON but not an object"
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
    return {
        "base_revision": revision,
        "changes": changes,
        "annotation": annotation,
        "valid": valid,
        "parse_error": parse_error,
    }


def _debug_logging_enabled() -> bool:
    return os.getenv("SMM_RECORDER_DEBUG_LOGGING", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def write_invalid_output_debug_record(
    session_id: str,
    *,
    raw_response: str,
    parse_error: str | None,
    logs_base_dir: str | None = None,
) -> str | None:
    """Append the failed raw recorder response only when debug logging is enabled."""
    if not _debug_logging_enabled():
        return None
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    if logs_base_dir is None:
        from services.action_logger import LOGS_BASE_DIR

        logs_base_dir = LOGS_BASE_DIR

    session_dir = os.path.join(logs_base_dir, session_id.strip())
    os.makedirs(session_dir, exist_ok=True)
    output_path = os.path.join(session_dir, "smm_recorder_debug.jsonl")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id.strip(),
        "parse_error": parse_error or "unknown",
        "raw_response": raw_response if isinstance(raw_response, str) else str(raw_response),
    }
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def record_prompt(evidence: str, model: Mapping[str, Any], final_reply: str) -> str:
    instruction = _PROMPT.read_text(encoding="utf-8").strip()
    return (
        f"{instruction}\n\n<W CODEBOOK>\n{definitions_for('W')}\n</W CODEBOOK>\n"
        f"<CURRENT SMM>\n{json.dumps(model, ensure_ascii=False)}\n</CURRENT SMM>\n"
        f"<FINAL GUIDE REPLY>\n{final_reply}\n</FINAL GUIDE REPLY>\n"
        f"<NEUTRAL VISIBLE EVIDENCE>\n{evidence}\n</NEUTRAL VISIBLE EVIDENCE>"
    )


def assess(
    llm_client: Any,
    evidence: str,
    model: Mapping[str, Any],
    final_reply: str,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    response = llm_client.chat_completions_create(
        messages=[{"role": "user", "content": record_prompt(evidence, model, final_reply)}],
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    assessment = parse_recorder_output(response, int(model.get("revision", 0) or 0))
    if not assessment["valid"] and session_id:
        write_invalid_output_debug_record(
            session_id,
            raw_response=response,
            parse_error=assessment.get("parse_error"),
        )
    return assessment
