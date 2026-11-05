"""Map images for the stages that judge route direction (H and the reply).

A Follower agent gets one image: its own map with its own drawing, never the Guide map.

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

from agent.map_task_as_guide.agent_session_context import _follower_progress, guide_sees_follower_drawing

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_MAP_DIRS = (_BACKEND_ROOT / "uploads" / "maps", _BACKEND_ROOT / "map_assets")

IMAGE_NOTE = (
    "\n\n<MAP IMAGES>\nImage 1: the Guide map; the dashed line is the target route from START to Finish. "
    "Image 2: the Follower's map with the Follower's current drawing.\n</MAP IMAGES>"
)
GUIDE_ONLY_IMAGE_NOTE = (
    "\n\n<MAP IMAGES>\nImage 1: the Guide map; the dashed line is the target route from START to Finish. "
    "The Follower's map and drawing are not available.\n</MAP IMAGES>"
)
FOLLOWER_IMAGE_NOTE = (
    "\n\n<MAP IMAGES>\nImage 1: your Follower map with your current drawing in black. It shows the same "
    "landmarks as your Follower workspace grid.\n</MAP IMAGES>"
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


def _follower_overlay(follower_map: Path, canvas_data_url: str | None, grid_text: str | None = None) -> str:
    """The Follower map with its drawing: the human's canvas, or else the drawn grid cells (an AI Follower)."""
    from PIL import Image, ImageDraw

    base = Image.open(follower_map).convert("RGBA")
    if isinstance(canvas_data_url, str) and canvas_data_url.startswith("data:image/") and "," in canvas_data_url:
        drawing = Image.open(io.BytesIO(base64.b64decode(canvas_data_url.split(",", 1)[1]))).convert("RGBA")
        base = Image.alpha_composite(base, drawing.resize(base.size))
    elif isinstance(grid_text, str) and "x" in grid_text:
        rows = grid_text.strip().splitlines()
        cell_w, cell_h = base.width / len(rows[0]), base.height / len(rows)
        pen = ImageDraw.Draw(base)
        for r, line in enumerate(rows):
            for c, value in enumerate(line):
                if value == "x":
                    pen.rectangle([(c + 0.25) * cell_w, (r + 0.25) * cell_h, (c + 0.75) * cell_w, (r + 0.75) * cell_h],
                                  fill=(0, 0, 0, 255))
    buffer = io.BytesIO()
    base.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def map_image_parts(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return [Guide map, Follower map + drawing] as image_url parts, or [] if unavailable.

    When the Guide cannot see the Follower's drawing, only the Guide map is returned.
    """
    try:
        guide_map, follower_map = _map_path(session, "guide"), _map_path(session, "follower")
        if not guide_map:
            return []
        guide_part = {"type": "image_url", "image_url": {"url": _file_data_url(str(guide_map), guide_map.stat().st_mtime)}}
        if not guide_sees_follower_drawing(session):
            return [guide_part]
        if not follower_map:
            return []
        progress = _follower_progress(session) or {}
        return [
            guide_part,
            {"type": "image_url", "image_url": {"url": _follower_overlay(follower_map, progress.get("canvasDataUrl"))}},
        ]
    except Exception:  # Images only add context; never block the turn.
        return []


def follower_image_parts(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return [Follower map + the Follower's own drawing] for a Follower agent, or [] if unavailable."""
    try:
        follower_map = _map_path(session, "follower")
        if not follower_map:
            return []
        progress = _follower_progress(session) or {}
        overlay = _follower_overlay(follower_map, progress.get("canvasDataUrl"), progress.get("grid_text"))
        return [{"type": "image_url", "image_url": {"url": overlay}}]
    except Exception:  # Images only add context; never block the turn.
        return []


def user_content(
    prompt: str, images: list[dict[str, Any]] | None, note: str | None = None,
) -> str | list[dict[str, Any]]:
    """Plain text without images; otherwise a text part followed by the image parts.

    Without `note`, the Guide's notes are used (one image: Guide map only).
    """
    if not images:
        return prompt
    if note is None:
        note = GUIDE_ONLY_IMAGE_NOTE if len(images) == 1 else IMAGE_NOTE
    return [{"type": "text", "text": prompt + note}, *images]
