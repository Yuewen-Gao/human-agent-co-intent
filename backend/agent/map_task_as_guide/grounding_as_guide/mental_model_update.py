"""Prompt construction and validation for Map Task interpretation updates.

This module deliberately represents only an inspectable, task-relevant
interpretation. It must not be used to request or expose hidden model reasoning.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from config.grounding_treatment import grounding_treatment_enabled

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
    "communicationProtocol",
    "partnerKnowledge",
    "partnerNextAction",
)

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


def append_mental_model_reply_context(prompt: str, mental_model: Mapping[str, Any] | None) -> str:
    """Expose SMM to the treatment reply path without requesting a model patch.

    The independent recorder owns updates.  Keeping patch instructions out of
    the reply agent prevents a normal response from silently becoming an SMM
    mutation mechanism.
    """
    if not grounding_treatment_enabled():
        return prompt
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
    return (
        f"{prompt.rstrip()}\n\n<EXPLICIT_WORKING_INTERPRETATION>\n"
        f"{json.dumps(state, ensure_ascii=False, sort_keys=True)}\n"
        "</EXPLICIT_WORKING_INTERPRETATION>\n"
        "Use this shared, editable record only for continuity. Current visible maps, "
        "trajectory, and conversation override it. Do not output a mental_model_patch; "
        "a separate recorder updates this record.\n"
    )

