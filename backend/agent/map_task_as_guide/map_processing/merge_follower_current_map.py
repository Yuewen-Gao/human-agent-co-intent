"""Build a text-only Follower workspace from landmarks and the current trace.

The Guide target map and the Follower workspace intentionally use separate
coordinate systems.  This module only overlays two sources that are already in
the Follower coordinate system: the Follower landmark map and the Follower's
current drawing trajectory.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _grid_size(follower_map: Mapping[str, Any]) -> tuple[int, int]:
    raw_size = follower_map.get("grid_size")
    if not isinstance(raw_size, list) or len(raw_size) != 2:
        raise ValueError("follower map must contain grid_size: [rows, cols]")
    try:
        rows, cols = int(raw_size[0]), int(raw_size[1])
    except (TypeError, ValueError) as error:
        raise ValueError("follower map grid_size must contain integers") from error
    if rows <= 0 or cols <= 0:
        raise ValueError("follower map grid_size must be positive")
    return rows, cols


def _valid_cells(raw_cells: Any, rows: int, cols: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if not isinstance(raw_cells, list):
        return cells
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 2:
            continue
        try:
            row, col = int(raw_cell[0]), int(raw_cell[1])
        except (TypeError, ValueError):
            continue
        if 0 <= row < rows and 0 <= col < cols:
            cells.add((row, col))
    return cells


def _landmark_cells(
    follower_map: Mapping[str, Any], rows: int, cols: int
) -> dict[str, set[tuple[int, int]]]:
    raw_landmarks = follower_map.get("landmarks")
    if not isinstance(raw_landmarks, Mapping):
        return {}

    result: dict[str, set[tuple[int, int]]] = {}
    for raw_name, raw_landmark in raw_landmarks.items():
        if not isinstance(raw_name, str) or not isinstance(raw_landmark, Mapping):
            continue
        summary = raw_landmark.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        cells = _valid_cells(raw_landmark.get("cells"), rows, cols)
        if not cells:
            cells = _valid_cells(summary.get("boundary_cells"), rows, cols)
        result[raw_name] = cells
    return result


def _trajectory_cells(grid_text: str, rows: int, cols: int) -> set[tuple[int, int]]:
    if not isinstance(grid_text, str):
        raise ValueError("follower trajectory grid must be text")
    lines = grid_text.strip().splitlines()
    if len(lines) != rows:
        raise ValueError(f"follower trajectory grid expected {rows} rows, got {len(lines)}")

    cells: set[tuple[int, int]] = set()
    for row, line in enumerate(lines):
        if len(line) != cols:
            raise ValueError(
                f"follower trajectory grid row {row} expected {cols} columns, got {len(line)}"
            )
        invalid = set(line) - {".", "x"}
        if invalid:
            raise ValueError("follower trajectory grid may contain only '.' and 'x'")
        cells.update((row, col) for col, value in enumerate(line) if value == "x")
    return cells


def _landmark_summary_lines(follower_map: Mapping[str, Any]) -> list[str]:
    raw_landmarks = follower_map.get("landmarks")
    if not isinstance(raw_landmarks, Mapping):
        return ["- none"]

    lines: list[str] = []
    for name in sorted(raw_landmarks):
        landmark = raw_landmarks[name]
        if not isinstance(name, str) or not isinstance(landmark, Mapping):
            continue
        summary = landmark.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        centroid = summary.get("centroid")
        bbox = summary.get("bbox")
        lines.append(
            f"- {name}: centroid={tuple(centroid) if isinstance(centroid, list) else 'unknown'}, "
            f"bbox={dict(bbox) if isinstance(bbox, Mapping) else 'unknown'}"
        )
    return lines or ["- none"]


def merge_follower_current_map(
    follower_map: Mapping[str, Any], trajectory_grid_text: str
) -> dict[str, Any]:
    """Overlay a Follower trajectory on the Follower landmark map.

    Overlay legend: ``.`` empty, ``#`` landmark, ``x`` trajectory, and ``!`` a
    trajectory cell that overlaps a landmark.  The returned grid is exclusively
    in the Follower coordinate system.
    """
    if not isinstance(follower_map, Mapping):
        raise ValueError("follower map must be an object")
    rows, cols = _grid_size(follower_map)
    trajectory_cells = _trajectory_cells(trajectory_grid_text, rows, cols)
    landmarks = _landmark_cells(follower_map, rows, cols)
    landmark_cells = set().union(*landmarks.values()) if landmarks else set()

    overlay = [["." for _ in range(cols)] for _ in range(rows)]
    for row, col in landmark_cells:
        overlay[row][col] = "#"
    for row, col in trajectory_cells:
        overlay[row][col] = "!" if (row, col) in landmark_cells else "x"

    overlaps = {
        name: [list(cell) for cell in sorted(cells & trajectory_cells)]
        for name, cells in sorted(landmarks.items())
        if cells & trajectory_cells
    }
    return {
        "grid_size": [rows, cols],
        "grid_text": "\n".join("".join(row) for row in overlay),
        "trace_cell_count": len(trajectory_cells),
        "trace_landmark_overlaps": overlaps,
        "landmark_summaries": _landmark_summary_lines(follower_map),
    }


def build_follower_current_workspace_context(
    follower_map: Mapping[str, Any], trajectory_grid_text: str
) -> str:
    """Render the merged Follower workspace as an auditable prompt block."""
    merged = merge_follower_current_map(follower_map, trajectory_grid_text)
    rows, cols = merged["grid_size"]
    overlap_summary = merged["trace_landmark_overlaps"] or "none"
    landmarks_text = "\n".join(merged["landmark_summaries"])
    return (
        "<FOLLOWER CURRENT WORKSPACE>\n"
        "coordinate_system: follower map only; do not compare coordinates directly with the Guide map\n"
        f"grid_size: {rows} rows x {cols} columns\n"
        "legend: . = empty, # = follower landmark, x = follower trajectory, "
        "! = follower trajectory overlaps a landmark cell\n"
        "landmarks:\n"
        f"{landmarks_text}\n"
        f"trace_cell_count: {merged['trace_cell_count']}\n"
        f"trace_landmark_overlaps: {overlap_summary}\n"
        "grid:\n"
        f"{merged['grid_text']}\n"
        "</FOLLOWER CURRENT WORKSPACE>"
    )
