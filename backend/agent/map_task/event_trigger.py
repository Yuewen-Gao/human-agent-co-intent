"""Stable event keys for waking a Map Task guide agent."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional


def follower_trajectory_event_key(progress: Mapping[str, Any]) -> Optional[str]:
    """Return a stable key for a follower drawing, or ``None`` without one.

    Display-only metadata and scoring ratios do not change the key. This keeps
    resize/re-render updates from waking the guide while preserving every real
    trajectory change.
    """
    if not isinstance(progress, Mapping):
        return None
    trajectory = {
        key: progress.get(key)
        for key in ("canvasDataUrl", "grid_text", "filledCells")
        if progress.get(key) is not None
    }
    if not trajectory:
        return None
    encoded = json.dumps(trajectory, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"follower_trajectory:{digest}"
