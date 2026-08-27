"""Prompt construction and validation for Map Task interpretation updates.

This module deliberately represents only an inspectable, task-relevant
interpretation. It must not be used to request or expose hidden model reasoning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MENTAL_MODEL_FIELD_KEYS = (
    "taskGoal",
    "taskSpecification",
    "procedure",
    "constraint",
    "decisionPriority",
    "taskState",
    "operationalCapability",
    "informationAccess",
    "roleResponsibility",
    "coordinationProtocol",
    "partnerKnowledge",
    "partnerNextAction",
)

_ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "mental_model_prompt.txt"


def _as_revision(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _normalize_fields(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, str] = {}
    for key in MENTAL_MODEL_FIELD_KEYS:
        field_value = value.get(key)
        if isinstance(field_value, Mapping):
            field_value = field_value.get("value")
        if isinstance(field_value, str) and field_value.strip():
            normalized[key] = field_value.strip()
    return normalized


def append_mental_model_update_prompt(prompt: str, mental_model: Mapping[str, Any] | None) -> str:
    """Append current visible state and the incremental-update contract to a prompt."""
    model = mental_model if isinstance(mental_model, Mapping) else {}
    state = {
        "revision": _as_revision(model.get("revision")) or 0,
        "fields": _normalize_fields(model.get("fields", model)),
        "locked_fields": [
            key
            for key, field in (model.get("fields", {}) or {}).items()
            if key in MENTAL_MODEL_FIELD_KEYS and isinstance(field, Mapping) and field.get("locked") is True
        ],
    }
    instruction = _PROMPT_PATH.read_text(encoding="utf-8").strip()
    state_json = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return (
        f"{prompt.rstrip()}\n\n"
        "<EXPLICIT_WORKING_INTERPRETATION>\n"
        f"{state_json}\n"
        "</EXPLICIT_WORKING_INTERPRETATION>\n\n"
        f"{instruction}\n"
    )


def parse_mental_model_patch(response: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a validated, sparse patch from an agent JSON response.

    Unknown fields, empty values, non-string evidence ids, and unsupported
    confidence values are discarded before the patch reaches persisted state.
    """
    if not isinstance(response, Mapping):
        return {"base_revision": None, "changes": {}}

    raw_patch = response.get("mental_model_patch")
    if not isinstance(raw_patch, Mapping):
        return {"base_revision": None, "changes": {}}

    changes: dict[str, dict[str, Any]] = {}
    raw_changes = raw_patch.get("changes")
    if isinstance(raw_changes, Mapping):
        for key in MENTAL_MODEL_FIELD_KEYS:
            candidate = raw_changes.get(key)
            if not isinstance(candidate, Mapping):
                continue

            value = candidate.get("value")
            if not isinstance(value, str) or not value.strip():
                continue

            change: dict[str, Any] = {"value": value.strip()}
            evidence_ids = candidate.get("evidence_message_ids")
            if isinstance(evidence_ids, list):
                ids = [item.strip() for item in evidence_ids if isinstance(item, str) and item.strip()]
                if ids:
                    change["evidence_message_ids"] = ids

            confidence = candidate.get("confidence")
            if isinstance(confidence, str) and confidence.lower() in _ALLOWED_CONFIDENCE:
                change["confidence"] = confidence.lower()

            changes[key] = change

    return {
        "base_revision": _as_revision(raw_patch.get("base_revision")),
        "changes": changes,
    }
