#!/usr/bin/env python3
"""
map_formatting.py

For each session in data_preprocessed/, find the follower's last map_draw_stop action,
load the brush-trace map_image (RGBA, transparent background + ink),
project it onto a reference grid (from --grid-json), detect inked cells,
and output per-session:

    formatted_maps/{session}_{participant}_route.json
        {"session": ..., "participant": ..., "grid_size": [rows, cols],
         "route_cells": [[r, c], ...]}

    formatted_maps/{session}_{participant}_route.txt
        ASCII art grid: x = inked cell, . = empty, borders #

Also saves the resized 810×1180 PNG for reference.

Usage:
    python map_formatting.py
    python map_formatting.py --grid-json path/to/map.json --ink-threshold 10 --min-fraction 0.05
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

DATA_DIR        = HERE / "data_preprocessed"
SESSION_FILES   = HERE / "session_files" / "logs"
OUTPUT_DIR      = HERE / "formatted_maps"
TARGET_W, TARGET_H = 810, 1180

DEFAULT_GRID_JSON = (
    HERE.parent / "follower_agent_simulation" / "maps" / "map3.map.json"
)

SESSION_FOLLOWER_NAMES: dict[str, set[str]] = {
    "study1": {"Grace"},
    "study2": {"Cameron"},
    "study3": {"Jesse"},
}
PILOT_SESSIONS = {"pilot1"}

# ---------------------------------------------------------------------------
# Grid ink detection
# ---------------------------------------------------------------------------

def ink_mask_resized(
    img: Image.Image,
    ink_alpha_threshold: int = 10,
) -> tuple[np.ndarray, int, int]:
    """
    Resize to TARGET_W×TARGET_H (same canvas as ground-truth grid mapping), return bool ink mask (H,W).
    """
    resized = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    arr = np.array(resized)

    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3].astype(np.float32)
    else:
        gray = np.array(resized.convert("L")).astype(np.float32)
        alpha = 255 - gray

    ink_mask = alpha > ink_alpha_threshold
    h, w = ink_mask.shape
    return ink_mask, h, w


def detect_route_cells_trajectory(
    img: Image.Image,
    rows: int,
    cols: int,
    ink_alpha_threshold: int = 10,
) -> list[list[int]]:
    """
    Occupancy grid aligned with ``grid_size`` [rows, cols] from map JSON (e.g. 60×40).

    A cell [r, c] is marked if the brush stroke **passes through** it: any ink pixel
    (alpha > threshold) falls inside that cell after projecting the image onto the
    same TARGET_W×TARGET_H canvas used for ground truth — i.e. treat the cell as
    filled whenever the trajectory intersects it, not a minimum ink fraction.
    """
    ink_mask, h, w = ink_mask_resized(img, ink_alpha_threshold)
    route_cells: list[list[int]] = []
    for r in range(rows):
        r0 = int(r * h / rows)
        r1 = int((r + 1) * h / rows)
        for c in range(cols):
            c0 = int(c * w / cols)
            c1 = int((c + 1) * w / cols)
            patch = ink_mask[r0:r1, c0:c1]
            if patch.size and np.any(patch):
                route_cells.append([r, c])
    return route_cells


def detect_route_cells(
    img: Image.Image,
    rows: int,
    cols: int,
    ink_alpha_threshold: int = 10,
    min_ink_fraction: float = 0.05,
) -> list[list[int]]:
    """
    Return list of [row, col] cells where brush ink fraction >= min_ink_fraction.
    Ink pixels = alpha > ink_alpha_threshold.
    Image is first resized to TARGET_W x TARGET_H before cell analysis.
    """
    ink_mask, h, w = ink_mask_resized(img, ink_alpha_threshold)

    route_cells: list[list[int]] = []
    for r in range(rows):
        r0 = int(r * h / rows)
        r1 = int((r + 1) * h / rows)
        for c in range(cols):
            c0 = int(c * w / cols)
            c1 = int((c + 1) * w / cols)
            patch = ink_mask[r0:r1, c0:c1]
            if patch.size == 0:
                continue
            frac = patch.sum() / patch.size
            if frac >= min_ink_fraction:
                route_cells.append([r, c])

    return route_cells


# ---------------------------------------------------------------------------
# Txt grid builder
# ---------------------------------------------------------------------------

ROUTE_CHAR = "x"
EMPTY_CHAR = "."


def route_grid_from_canvas_data_url(
    canvas_data_url: str,
    rows: int = 60,
    cols: int = 40,
    ink_alpha_threshold: int = 10,
) -> str:
    """Convert a live follower canvas PNG data URL into an ``x``/``.`` grid.

    The browser syncs ``canvasDataUrl`` into the follower's ``map_progress``.
    This function uses the same 810x1180 projection and trajectory-cell rule as
    the offline ALMANAC formatting script, so the live drawing grid shares the
    coordinate system of ``almanac_maps/map_guide.json``.
    """
    if not isinstance(canvas_data_url, str) or not canvas_data_url.startswith("data:image/"):
        return ""
    try:
        _header, payload = canvas_data_url.split(",", 1)
        raw = base64.b64decode(payload, validate=True)
        if not raw:
            return ""
        with Image.open(BytesIO(raw)) as img:
            route_cells = detect_route_cells_trajectory(
                img, rows, cols, ink_alpha_threshold=ink_alpha_threshold
            )
    except (ValueError, OSError, base64.binascii.Error):
        return ""

    route_set = {(row, col) for row, col in route_cells}
    return "\n".join(
        "".join(ROUTE_CHAR if (row, col) in route_set else EMPTY_CHAR for col in range(cols))
        for row in range(rows)
    )


def build_route_txt(
    rows: int,
    cols: int,
    route_cells: list[list[int]],
    session: str,
    participant: str,
) -> str:
    route_set = {(rc[0], rc[1]) for rc in route_cells}
    grid = [[EMPTY_CHAR] * cols for _ in range(rows)]
    for r, c in route_set:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = ROUTE_CHAR

    border = "#" + "-" * cols + "#"
    lines = [border]
    for row in grid:
        lines.append("#" + "".join(row) + "#")
    lines.append(border)
    lines.append("")
    lines.append(f"# session: {session}  participant: {participant}")
    lines.append(f"# grid_size (rows, cols): {rows}, {cols}")
    lines.append(f"# {ROUTE_CHAR} = brush route cell ({len(route_cells)} cells)")
    lines.append(f"# {EMPTY_CHAR} = empty")
    lines.append("")
    lines.append(f"# route_cells: {json.dumps(route_cells, separators=(',', ':'))}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def find_map_image_path(session_id: str, map_image_field: str) -> Path | None:
    filename = Path(map_image_field).name
    candidate = SESSION_FILES / session_id / "files" / filename
    return candidate if candidate.is_file() else None


def process_session(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    session_name: str = data.get("session_name", json_path.stem)
    session_id: str = data.get("session_id", "")
    actions: list[dict] = data.get("human_actions_all", [])

    follower_names = SESSION_FOLLOWER_NAMES.get(session_name)
    is_pilot = session_name in PILOT_SESSIONS

    by_participant: dict[str, list[dict]] = {}
    for a in actions:
        if a.get("action_type") != "map_draw_stop":
            continue
        pname: str = a.get("participant_name", "unknown")
        if not is_pilot and follower_names and pname not in follower_names:
            continue
        by_participant.setdefault(pname, []).append(a)

    results = []
    for pname, acts in by_participant.items():
        acts_sorted = sorted(acts, key=lambda x: (x.get("timestamp") or "", x.get("action_id") or ""))
        last = acts_sorted[-1]
        map_image_field: str = last.get("map_image", "")
        if not map_image_field:
            print(f"  [warn] {session_name}/{pname}: no map_image field", file=sys.stderr)
            continue
        img_path = find_map_image_path(session_id, map_image_field)
        if img_path is None:
            print(f"  [warn] {session_name}/{pname}: file not found: {map_image_field}", file=sys.stderr)
            continue
        results.append({
            "session_name": session_name,
            "session_id": session_id,
            "participant_name": pname,
            "map_image_path": img_path,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument(
        "--grid-json", type=Path, default=DEFAULT_GRID_JSON,
        help="map.json providing grid_size [rows, cols] (default: map3.map.json)",
    )
    parser.add_argument(
        "--ink-threshold", type=int, default=10,
        help="Alpha value above which a pixel counts as ink (default: 10)",
    )
    parser.add_argument(
        "--min-fraction", type=float, default=0.05,
        help="Min fraction of ink pixels per cell to mark as route (default: 0.05)",
    )
    args = parser.parse_args()

    if not args.grid_json.is_file():
        sys.exit(f"Grid JSON not found: {args.grid_json}")

    grid_data = json.loads(args.grid_json.read_text(encoding="utf-8"))
    gs = grid_data.get("grid_size")
    if not isinstance(gs, list) or len(gs) != 2:
        sys.exit("grid.json missing grid_size: [rows, cols]")
    rows, cols = int(gs[0]), int(gs[1])
    print(f"Grid: {rows}×{cols}  (from {args.grid_json.name})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(DATA_DIR.glob("*_actions_and_annotations.json"))
    if not json_files:
        sys.exit(f"No JSON files found in {DATA_DIR}")

    total = 0
    for jf in json_files:
        for entry in process_session(jf):
            session   = entry["session_name"]
            pname     = entry["participant_name"]
            src_path: Path = entry["map_image_path"]
            stem      = f"{session}_{pname}"

            img = Image.open(src_path)

            # --- Route detection ---
            route_cells = detect_route_cells(
                img, rows, cols,
                ink_alpha_threshold=args.ink_threshold,
                min_ink_fraction=args.min_fraction,
            )

            # --- Save JSON ---
            json_out = OUTPUT_DIR / f"{stem}_route.json"
            json_out.write_text(
                json.dumps({
                    "session": session,
                    "participant": pname,
                    "grid_size": [rows, cols],
                    "grid_json": str(args.grid_json),
                    "route_cells": route_cells,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # --- Save TXT ---
            txt_out = OUTPUT_DIR / f"{stem}_route.txt"
            txt_out.write_text(
                build_route_txt(rows, cols, route_cells, session, pname),
                encoding="utf-8",
            )

            # --- Save resized PNG (reference) ---
            png_out = OUTPUT_DIR / f"{stem}_map.png"
            img.resize((TARGET_W, TARGET_H), Image.LANCZOS).save(png_out, format="PNG")

            print(f"  {stem}: {len(route_cells)} route cells  →  {json_out.name} / {txt_out.name}")
            total += 1

    print(f"\nDone. {total} session(s) processed → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
