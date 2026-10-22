"""Map images for the grounding stages that judge route direction (H and the reply).

Reading route direction from the text grids was unstable across calls; with the
Guide map image and the Follower's drawing on the Follower map, the stated
target stayed consistent in replay. Images are an addition: any failure returns
no images and the stage proceeds on text alone.
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from agent.map_task_as_guide.agent_session_context import _follower_progress

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_MAP_DIRS = (_BACKEND_ROOT / "uploads" / "maps", _BACKEND_ROOT / "map_assets")

IMAGE_NOTE = (
    "\n\n<MAP IMAGES>\nImage 1: the Guide map; the dashed line is the target route from START to Finish. "
    "Image 2: the Follower's map with the Follower's current drawing. Landmark positions can differ between "
    "the two maps; describe targets relative to landmarks as they appear on the Follower's map.\n</MAP IMAGES>"
)


def _map_path(session: Mapping[str, Any], role: str) -> Path | None:
    """Resolve the session's map file for a role the same way /api/maps serves it."""
    maps = ((session.get("params") or {}).get("maps") or []) if isinstance(session, Mapping) else []
    for entry in maps:
        if not isinstance(entry, Mapping) or str(entry.get("role", "")).lower() != role:
            continue
        name = str(entry.get("filename") or "")
        if not name or os.path.basename(name) != name:
            return None
        for directory in _MAP_DIRS:
            path = directory / name
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=8)
def _file_data_url(path: str, mtime: float) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _follower_overlay(follower_map: Path, canvas_data_url: str | None) -> str:
    from PIL import Image

    base = Image.open(follower_map).convert("RGBA")
    if isinstance(canvas_data_url, str) and canvas_data_url.startswith("data:image/") and "," in canvas_data_url:
        drawing = Image.open(io.BytesIO(base64.b64decode(canvas_data_url.split(",", 1)[1]))).convert("RGBA")
        base = Image.alpha_composite(base, drawing.resize(base.size))
    buffer = io.BytesIO()
    base.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def map_image_parts(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return [Guide map, Follower map + drawing] as image_url parts, or [] if unavailable."""
    try:
        guide_map, follower_map = _map_path(session, "guide"), _map_path(session, "follower")
        if not guide_map or not follower_map:
            return []
        progress = _follower_progress(session) or {}
        return [
            {"type": "image_url", "image_url": {"url": _file_data_url(str(guide_map), guide_map.stat().st_mtime)}},
            {"type": "image_url", "image_url": {"url": _follower_overlay(follower_map, progress.get("canvasDataUrl"))}},
        ]
    except Exception:  # Images only add context; never block the turn.
        return []


def user_content(prompt: str, images: list[dict[str, Any]] | None) -> str | list[dict[str, Any]]:
    """Plain text without images; otherwise a text part followed by the image parts."""
    if not images:
        return prompt
    return [{"type": "text", "text": prompt + IMAGE_NOTE}, *images]
