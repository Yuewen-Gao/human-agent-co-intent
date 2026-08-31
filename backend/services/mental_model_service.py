"""Session-scoped storage and conflict handling for visible interpretation state."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping

from config.grounding_treatment import grounding_treatment_enabled
from agent.map_task_as_guide.grounding_as_guide.mental_model_update import (
    MENTAL_MODEL_FIELD_KEYS,
)


UNKNOWN_SMM_VALUE = "Not yet observable from current collaboration."
_ANNOTATION_KEYS = (
    "explanation_transcription",
    "task_model_q1",
    "partner_model_q2",
    "self_model_q3",
    "alignment_q4",
)


def _revision(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _model(session: dict[str, Any]) -> dict[str, Any]:
    candidate = session.get("mental_model")
    if not isinstance(candidate, dict):
        candidate = {"revision": 0, "fields": {}}
        session["mental_model"] = candidate
    candidate["revision"] = _revision(candidate.get("revision"))
    if not isinstance(candidate.get("fields"), dict):
        candidate["fields"] = {}
    return candidate


def _unknown_field() -> dict[str, Any]:
    return {
        "value": UNKNOWN_SMM_VALUE,
        "status": "not_observable",
        "confidence": "low",
        "source": "system",
        "locked": False,
    }


def ensure_complete_mental_model(session: dict[str, Any]) -> dict[str, Any]:
    """Ensure all W-backed SMM fields exist without inventing task facts.

    This is deliberately treatment-independent: it supports background research
    records in control sessions.  Visibility and participant edits remain gated
    by the public treatment-facing functions below.
    """
    model = _model(session)
    for key in MENTAL_MODEL_FIELD_KEYS:
        existing = model["fields"].get(key)
        if isinstance(existing, Mapping):
            entry = dict(existing)
            value = entry.get("value")
            if not isinstance(value, str) or not value.strip():
                entry = _unknown_field()
            else:
                entry["value"] = value.strip()
                entry.setdefault(
                    "status",
                    "not_observable" if entry["value"] == UNKNOWN_SMM_VALUE else "inferred",
                )
                entry.setdefault("confidence", "low")
                entry.setdefault("source", "legacy")
                entry.setdefault("locked", False)
            model["fields"][key] = entry
        elif isinstance(existing, str) and existing.strip():
            model["fields"][key] = {
                "value": existing.strip(),
                "status": "inferred",
                "confidence": "low",
                "source": "legacy",
                "locked": False,
            }
        else:
            model["fields"][key] = _unknown_field()
    return model


def public_mental_model(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI-safe values and display metadata, never internal locks/source."""
    raw = session.get("mental_model") if isinstance(session, Mapping) else None
    if not isinstance(raw, Mapping):
        return {"revision": 0, "fields": {}}
    fields = raw.get("fields")
    result: dict[str, dict[str, str]] = {}
    if isinstance(fields, Mapping):
        for key in MENTAL_MODEL_FIELD_KEYS:
            item = fields.get(key)
            value = item.get("value") if isinstance(item, Mapping) else item
            if isinstance(value, str):
                result[key] = {
                    "value": value,
                    "status": (
                        str(item.get("status"))
                        if isinstance(item, Mapping) and isinstance(item.get("status"), str)
                        else ("not_observable" if value == UNKNOWN_SMM_VALUE else "inferred")
                    ),
                    "confidence": (
                        str(item.get("confidence"))
                        if isinstance(item, Mapping) and isinstance(item.get("confidence"), str)
                        else "low"
                    ),
                }
    return {"revision": _revision(raw.get("revision")), "fields": result}


def should_publish_mental_model_update(changed: bool) -> bool:
    """Keep background control records from becoming a participant UI effect."""
    return bool(changed) and grounding_treatment_enabled()


def _smm_job_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def enqueue_smm_recording_job(
    session: dict[str, Any],
    *,
    agent_participant_id: str,
    action_id: str | None,
    message_id: str | None,
    evidence: str,
    reply: str,
    trajectory_grid_at_reply: str | None = None,
) -> dict[str, Any]:
    """Persist one Guide turn for later, ordered SMM recording."""
    jobs = session.setdefault("smm_recording_jobs", [])
    identity = action_id if isinstance(action_id, str) and action_id else message_id
    for job in jobs:
        if isinstance(job, dict) and job.get("identity") == identity:
            return job
    job = {
        "identity": identity,
        "agent_participant_id": agent_participant_id,
        "action_id": action_id,
        "message_id": message_id,
        "evidence": evidence,
        "reply": reply,
        "trajectory_grid_at_reply": trajectory_grid_at_reply,
        "status": "pending",
        "attempts": 0,
        "queued_at": _smm_job_timestamp(),
    }
    jobs.append(job)
    return job


def claim_next_smm_recording_job(
    session: dict[str, Any], agent_participant_id: str
) -> dict[str, Any] | None:
    """Claim the oldest pending job for one Guide runner."""
    for job in session.get("smm_recording_jobs", []):
        if (
            isinstance(job, dict)
            and job.get("agent_participant_id") == agent_participant_id
            and job.get("status") == "pending"
        ):
            job["status"] = "processing"
            job["attempts"] = int(job.get("attempts", 0) or 0) + 1
            job["started_at"] = _smm_job_timestamp()
            return job
    return None


def resume_smm_recording_jobs(session: dict[str, Any], agent_participant_id: str) -> int:
    """Requeue jobs interrupted by a backend restart before completion."""
    resumed = 0
    for job in session.get("smm_recording_jobs", []):
        if (
            isinstance(job, dict)
            and job.get("agent_participant_id") == agent_participant_id
            and job.get("status") == "processing"
        ):
            job["status"] = "pending"
            job.pop("started_at", None)
            resumed += 1
    return resumed


def finish_smm_recording_job(job: dict[str, Any], *, error: str | None = None) -> None:
    job["status"] = "failed" if error else "complete"
    job["finished_at"] = _smm_job_timestamp()
    if error:
        job["error"] = error


def append_agent_turn_record(
    session: dict[str, Any],
    *,
    action_id: str | None,
    message_id: str | None,
    annotation: Mapping[str, Any] | None,
    agent_participant_id: str | None = None,
    trajectory_grid_at_reply: str | None = None,
    recording_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Append one post-turn, exportable Guide-Agent record.

    ``action_id`` is the action logger UUID and therefore the primary key used
    in exports.  A deep copy prevents later SMM mutation from rewriting history.
    """
    model = ensure_complete_mental_model(session)
    normalized_annotation = {
        key: str((annotation or {}).get(key, "")).strip()
        for key in _ANNOTATION_KEYS
    }
    record = {
        "action_id": action_id if isinstance(action_id, str) and action_id else None,
        "message_id": message_id if isinstance(message_id, str) and message_id else None,
        "annotation": normalized_annotation,
        "current_smm": deepcopy(model),
        "agent_participant_id": (
            agent_participant_id
            if isinstance(agent_participant_id, str) and agent_participant_id
            else None
        ),
        "trajectory_grid_at_reply": (
            trajectory_grid_at_reply.strip()
            if isinstance(trajectory_grid_at_reply, str) and trajectory_grid_at_reply.strip()
            else None
        ),
        "recording_errors": [
            error for error in (recording_errors or [])
            if isinstance(error, str) and error
        ],
    }
    record["recording_status"] = "failed" if record["recording_errors"] else "complete"
    session.setdefault("agent_turn_records", []).append(record)
    return record


def _format_agent_annotation_timeline(records: Any) -> list[dict[str, Any]]:
    """Return local Agent records in the same annotation shape used by exports."""
    timeline: list[dict[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        annotation = record.get("annotation")
        annotation = annotation if isinstance(annotation, dict) else {}
        timeline.append(
            {
                "action_id": record.get("action_id") or "",
                "action_timestamp": record.get("action_timestamp") or "",
                "action_type": record.get("action_type") or "",
                "action_content": record.get("action_content") or "",
                "message_id": record.get("message_id") or "",
                "recording_status": record.get("recording_status") or "complete",
                "recording_errors": record.get("recording_errors") or [],
                "annotation": {
                    key: annotation.get(key) or "" for key in _ANNOTATION_KEYS
                },
                "current_smm": record.get("current_smm") or {},
            }
        )
    return timeline


def write_agent_annotation_timeline_file(
    session: Mapping[str, Any],
    session_id: str,
    *,
    logs_base_dir: str | None = None,
) -> str:
    """Atomically mirror persisted Agent annotations to the session log directory."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")

    if logs_base_dir is None:
        from services.action_logger import LOGS_BASE_DIR

        logs_base_dir = LOGS_BASE_DIR
    session_dir = os.path.join(logs_base_dir, session_id.strip())
    os.makedirs(session_dir, exist_ok=True)
    output_path = os.path.join(session_dir, "post_annotations_agent.json")
    fd, temporary_path = tempfile.mkstemp(
        dir=session_dir, prefix=".post_annotations_agent_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _format_agent_annotation_timeline(session.get("agent_turn_records")),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output_path


def apply_user_mental_model_update(
    session: dict[str, Any], base_revision: Any, fields: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply participant-authored fields and lock each changed field against agents."""
    if not grounding_treatment_enabled():
        return {"changed": False, "reason": "grounding_treatment_disabled", "mental_model": public_mental_model(session)}
    model = _model(session)
    if base_revision != model["revision"]:
        return {"changed": False, "reason": "stale_revision", "mental_model": public_mental_model(session)}
    if not isinstance(fields, Mapping):
        return {"changed": False, "reason": "invalid_fields", "mental_model": public_mental_model(session)}

    changed = False
    for key in MENTAL_MODEL_FIELD_KEYS:
        value = fields.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        previous = model["fields"].get(key)
        if (
            isinstance(previous, Mapping)
            and previous.get("value") == normalized
            and previous.get("source") == "user"
            and previous.get("locked") is True
        ):
            continue
        model["fields"][key] = {
            "value": normalized,
            "status": "confirmed",
            "confidence": "high",
            "source": "user",
            "locked": True,
        }
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}


def _apply_agent_mental_model_patch(
    session: dict[str, Any], patch: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply a validated agent patch after the caller chooses its visibility policy."""
    model = _model(session)
    if not isinstance(patch, Mapping):
        return {"changed": False, "reason": "invalid_patch", "mental_model": public_mental_model(session)}
    if patch.get("base_revision") != model["revision"]:
        return {"changed": False, "reason": "stale_revision", "mental_model": public_mental_model(session)}

    raw_changes = patch.get("changes")
    if not isinstance(raw_changes, Mapping):
        return {"changed": False, "reason": "invalid_patch", "mental_model": public_mental_model(session)}

    changed = False
    for key in MENTAL_MODEL_FIELD_KEYS:
        change = raw_changes.get(key)
        if not isinstance(change, Mapping) or not isinstance(change.get("value"), str):
            continue
        previous = model["fields"].get(key)
        if isinstance(previous, Mapping) and previous.get("locked") is True:
            continue

        value = change["value"].strip()
        if not value or (isinstance(previous, Mapping) and previous.get("value") == value):
            continue
        entry = {
            "value": value,
            "status": "inferred",
            "source": "agent",
            "locked": False,
        }
        if isinstance(change.get("confidence"), str):
            entry["confidence"] = change["confidence"]
        if isinstance(change.get("evidence_message_ids"), list):
            entry["evidence_message_ids"] = change["evidence_message_ids"]
        model["fields"][key] = entry
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}


def apply_internal_agent_mental_model_patch(
    session: dict[str, Any], patch: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply a background recorder patch without enabling treatment UI/prompt effects."""
    return _apply_agent_mental_model_patch(session, patch)
