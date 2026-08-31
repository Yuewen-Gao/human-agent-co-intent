"""Factory for the public, single-participant Map Task session template."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from config.experiments import get_experiment_by_id


_MAPS = (
    ("follower", "ALMANAC_follower_map.png"),
    ("guide", "ALMANAC_guide_map.png"),
)


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_public_maptask_session(session_key: str) -> dict[str, Any]:
    """Build one new public Map Task session for a non-empty session key."""
    normalized_key = session_key.strip() if isinstance(session_key, str) else ""
    if not normalized_key:
        raise ValueError("session_key is required")

    experiment_config = deepcopy(get_experiment_by_id("maptask") or {})
    participant_settings = experiment_config.setdefault("participant_settings", {})
    participant_settings["auto_start"] = True

    now = _utc_now_iso_z()
    session_id = str(uuid4())
    maps = [
        {
            "role": role,
            "filename": filename,
            "original_filename": filename,
            "file_path": f"/api/maps/{filename}",
        }
        for role, filename in _MAPS
    ]
    return {
        "session_id": session_id,
        "session_name": normalized_key,
        "experiment_type": "maptask",
        "experiment_config": experiment_config,
        "status": "waiting",
        "config": {},
        "params": {"duration": 30, "maps": maps},
        "interaction": {
            "communicationLevel": "Private Messaging",
            "communicationMedia": ["text"],
            "typeIndicator": "enabled",
            "awarenessDashboard": {
                "enabled": True,
                "items": ["Participant.name", "Participant.map_progress"],
            },
            "messageLength": False,
            "agentPerceptionTimeWindow": 15,
            "rationales": "step_wise",
        },
        "created_at": now,
        "started_at": None,
        "duration_minutes": 30,
        "remaining_seconds": None,
        "participants": [
            {
                "id": str(uuid4()),
                "name": "Human",
                "participant_name": "Human",
                "type": "human",
                "role": "follower",
                "status": "offline",
                "experiment_params": {},
            },
            {
                "id": str(uuid4()),
                "name": "Agent",
                "participant_name": "Agent",
                "type": "ai",
                "role": "guide",
                "status": "offline",
                "experiment_params": {},
            },
        ],
        "pending_offers": [],
        "completed_trades": [],
    }
