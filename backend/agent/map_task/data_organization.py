#!/usr/bin/env python3
"""
Organize one session export into follower vs guide JSON timelines.

Reads merged JSON by default from ``data_processed/``:

    data_processed/{session_name}_actions_and_annotations.json
    [optional] data_processed/{session_name}.json   (same schema)

Map snapshots from:

    data_processed/{session_name}/files/

Fallbacks (still checked via resolve logic):

    data/sessions/{session_id}/files/
    session_files/logs/{session_id}/files/

Projects follower snapshots onto the same canvas as ``map_formatting`` (810×1180) and the
grid from ``--grid-json`` (``grid_size`` = [rows, cols], e.g. 60×40 → 40 columns × 60 rows).
Stroke occupancy uses **trajectory cells**: any ink pixel in a cell marks that cell (aligned with
GT); use ``--grid-cell-mode density`` to restore the old minimum-ink-fraction rule.

``drawing_accuracy`` is the **mean quality** over ink cells: on GT route = 100%, Chebyshev
distance 1 to route = 2/3 (~66.67%), distance 2 = 1/3, farther = 0%.

Also writes score_board.txt (route = 3, distance-1 band = 2, distance-2 = 1, else .).

Usage:
    python data_organization.py study4 --canvas-visibility true
    # Older merged JSON still under ./data:
    python data_organization.py study1 --canvas-visibility true --data-dir ./data
    python data_organization.py study8 --canvas-visibility false \\
        --grid-json ../follower_agent_simulation/maps/map3.map.json \\
        --route-json ../follower_agent_simulation/maps/map3.map_route.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from map_formatting import SESSION_FILES, detect_route_cells, detect_route_cells_trajectory

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data_processed"
DEFAULT_LEGACY_SESSIONS = HERE / "data" / "sessions"
DEFAULT_GRID_JSON = HERE.parent / "follower_agent_simulation" / "maps" / "map3.map.json"
DEFAULT_ROUTE_JSON = HERE.parent / "follower_agent_simulation" / "maps" / "map3.map_route.json"

SESSION_FOLLOWER_NAMES: dict[str, frozenset[str]] = {
    "study1": frozenset({"Grace"}),
    "study2": frozenset({"Cameron"}),
    "study3": frozenset({"Jesse"}),
}


def cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def find_session_json(data_dir: Path, session_name: str) -> Path:
    """Prefer explicit filenames under data_dir (data_preprocess layout), then glob."""
    merged = data_dir / f"{session_name}_actions_and_annotations.json"
    if merged.is_file():
        try:
            blob = json.loads(merged.read_text(encoding="utf-8"))
            if str(blob.get("session_name") or "") == session_name:
                return merged
        except json.JSONDecodeError:
            pass

    short = data_dir / f"{session_name}.json"
    if short.is_file():
        try:
            blob = json.loads(short.read_text(encoding="utf-8"))
            if str(blob.get("session_name") or "") == session_name:
                return short
        except json.JSONDecodeError:
            pass

    matches: list[Path] = []
    suffix = "_actions_and_annotations.json"
    for p in sorted(data_dir.glob(f"*{suffix}")):
        stem = p.name[: -len(suffix)]
        if stem != session_name and not stem.startswith(session_name + "_"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(data.get("session_name") or "") != session_name:
            continue
        matches.append(p)
    if not matches:
        sys.exit(f"No merged JSON for session_name={session_name!r} under {data_dir}")
    if len(matches) > 1:
        sys.exit(f"Ambiguous JSON for session_name={session_name!r}: {matches}")
    return matches[0]


def follower_names_for_session(data: Mapping[str, Any]) -> frozenset[str]:
    sname = str(data.get("session_name") or "")
    named = SESSION_FOLLOWER_NAMES.get(sname)
    if named:
        return named
    # Heuristic: participant with the most map_draw_stop acts as follower.
    counts: dict[str, int] = {}
    for a in data.get("human_actions_all") or []:
        if a.get("action_type") != "map_draw_stop":
            continue
        pname = str(a.get("participant_name") or "")
        counts[pname] = counts.get(pname, 0) + 1
    if not counts:
        sys.exit(f"Cannot infer follower for session {sname}: no map_draw_stop actions.")
    max_n = max(counts.values())
    leaders = {n for n, c in counts.items() if c == max_n}
    if len(leaders) != 1:
        sys.exit(f"Ambiguous follower vote for session {sname}: {counts}")
    return frozenset(leaders)


def participant_blob_for_name(by_participant: Mapping[str, Any], display_name: str) -> dict[str, Any]:
    if display_name not in by_participant:
        sys.exit(f"Missing by_participant[{display_name!r}]")
    blob = by_participant[display_name]
    if not isinstance(blob, dict):
        sys.exit(f"by_participant[{display_name!r}] is not an object")
    return blob


def mental_model_from_annotation(ann: Mapping[str, Any]) -> dict[str, str]:
    return {
        "rationale": str(ann.get("explanation_transcription") or ""),
        "team_goal": str(ann.get("task_model_q1") or ""),
        "partner_intent": str(ann.get("partner_model_q2") or ""),
        "self_reasoning": str(ann.get("self_model_q3") or ""),
    }


def classify_follower_action(a: Mapping[str, Any]) -> tuple[str, str] | None:
    """Returns (action_type, action_content text) or None to skip."""
    at = str(a.get("action_type") or "")
    ac = str(a.get("action_content") or "")
    if at == "send_message":
        return "message", ac
    if at == "map_draw_stop":
        meta = a.get("metadata") or {}
        tool = meta.get("stroke_tool")
        if ac == "brush_release" or tool == "brush":
            return "draw", ""
        if ac == "eraser_release" or tool == "eraser":
            return "erase", ""
        return "draw", ""
    if at == "map_tool_click":
        if ac == "undo":
            return "undo", ""
        if ac == "reset":
            return "reset", ""
    return None


def map_canvas_cells_from_image(
    img_path: Path,
    rows: int,
    cols: int,
    ink_threshold: int,
    grid_cell_mode: str,
    min_fraction: float,
) -> list[list[int]]:
    img = Image.open(img_path)
    if grid_cell_mode == "trajectory":
        return detect_route_cells_trajectory(img, rows, cols, ink_alpha_threshold=ink_threshold)
    if grid_cell_mode == "density":
        return detect_route_cells(
            img, rows, cols, ink_alpha_threshold=ink_threshold, min_ink_fraction=min_fraction
        )
    sys.exit(f"Unknown --grid-cell-mode: {grid_cell_mode!r} (use trajectory or density)")


def canvas_string_from_cells(cells: list[list[int]], rows: int, cols: int) -> str:
    ink = {(r[0], r[1]) for r in cells if len(r) >= 2}
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for r, c in ink:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = "x"
    return "\n".join("".join(row) for row in grid)


def resolve_map_image_file(session_id: str, map_image_field: str, roots: list[Path]) -> Path | None:
    if not map_image_field:
        return None
    name = Path(map_image_field).name
    rel = str(map_image_field).lstrip("./")
    tried: list[Path] = []
    for root in roots:
        candidates = [
            root / session_id / "files" / name,
            root / session_id / rel,
            root / "files" / name,
            root / name,
        ]
        for c in candidates:
            tried.append(c)
            if c.is_file():
                return c
    return None


def ink_cell_quality_fraction(cell: tuple[int, int], route_set: set[tuple[int, int]]) -> float:
    """
    Per-cell contribution toward drawing_accuracy (mean over ink cells):
    on GT route → 1.0 (100%); Chebyshev distance 1 → 2/3 (~66.67%); distance 2 → 1/3; else 0.
    """
    if cell in route_set:
        return 1.0
    mind = min(cheb(cell, r) for r in route_set)
    if mind == 1:
        return 2.0 / 3.0
    if mind == 2:
        return 1.0 / 3.0
    return 0.0


def drawing_accuracy_value(ink_cells: list[list[int]], route_set: set[tuple[int, int]]) -> float | None:
    """Mean of ink_cell_quality_fraction over all ink cells."""
    cells = [(c[0], c[1]) for c in ink_cells if len(c) >= 2]
    if not cells:
        return None
    total = sum(ink_cell_quality_fraction(c, route_set) for c in cells)
    return round(total / len(cells), 6)


def render_score_board(rows: int, cols: int, route_set: set[tuple[int, int]]) -> str:
    lines: list[str] = []
    for r in range(rows):
        row_chars: list[str] = []
        for c in range(cols):
            cell = (r, c)
            if cell in route_set:
                row_chars.append("3")
                continue
            mind = min(cheb(cell, x) for x in route_set)
            if mind == 1:
                row_chars.append("2")
            elif mind == 2:
                row_chars.append("1")
            else:
                row_chars.append(".")
        lines.append("".join(row_chars))
    header = (
        "# score_board: 3=route cell, 2=Chebyshev distance 1 to route, "
        "1=distance 2, .=other\n"
    )
    return header + "\n".join(lines) + "\n"


def load_route_cells(route_json: Path) -> list[list[int]]:
    data = json.loads(route_json.read_text(encoding="utf-8"))
    rc = data.get("route_cells")
    if not isinstance(rc, list):
        sys.exit(f"{route_json} missing route_cells list")
    out: list[list[int]] = []
    for pair in rc:
        if isinstance(pair, list) and len(pair) == 2:
            out.append([int(pair[0]), int(pair[1])])
    return out


def sorted_actions_all(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    ha = list(data.get("human_actions_all") or [])
    ha.sort(key=lambda x: (str(x.get("timestamp") or ""), str(x.get("action_id") or "")))
    return ha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    parser.add_argument("session_name", help="session_name field, e.g. study1")
    parser.add_argument(
        "--canvas-visibility",
        required=True,
        choices=("true", "false"),
        help="Whether guide rows include follower canvas between consecutive guide messages.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"preprocessed bundle root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--grid-json", type=Path, default=DEFAULT_GRID_JSON)
    parser.add_argument("--route-json", type=Path, default=DEFAULT_ROUTE_JSON)
    parser.add_argument("--ink-threshold", type=int, default=10)
    parser.add_argument(
        "--grid-cell-mode",
        choices=("trajectory", "density"),
        default="trajectory",
        help="trajectory: cell on if any ink pixel (stroke passes through); "
        "density: old rule using --min-fraction (default 0.05).",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.05,
        help="Only for --grid-cell-mode density: min ink fraction per cell.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: {data-dir}/organized/{session_name}",
    )
    args = parser.parse_args()
    canvas_visibility = args.canvas_visibility == "true"

    if not args.grid_json.is_file():
        sys.exit(f"grid json not found: {args.grid_json}")
    if not args.route_json.is_file():
        sys.exit(f"route json not found: {args.route_json}")

    grid_meta = json.loads(args.grid_json.read_text(encoding="utf-8"))
    gs = grid_meta.get("grid_size")
    if not isinstance(gs, list) or len(gs) != 2:
        sys.exit("grid json missing grid_size [rows, cols]")
    rows, cols = int(gs[0]), int(gs[1])

    route_cells_ref = load_route_cells(args.route_json)
    route_set: set[tuple[int, int]] = {(r, c) for r, c in route_cells_ref}

    json_path = find_session_json(args.data_dir, args.session_name)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    session_id = str(data.get("session_id") or "").strip()
    if not session_id:
        sys.exit("session_id missing in merged JSON")

    follower_set = follower_names_for_session(data)
    by_p = data.get("by_participant") or {}
    follower_name = next(iter(follower_set))
    guide_names = [n for n in by_p if n not in follower_set]
    if len(guide_names) != 1:
        sys.exit(f"Expected one guide; follower={follower_name}, participants={list(by_p.keys())}")
    guide_name = guide_names[0]

    follower_blob = participant_blob_for_name(by_p, follower_name)
    guide_blob = participant_blob_for_name(by_p, guide_name)

    f_actions = follower_blob.get("human_actions") or []
    f_timeline = follower_blob.get("post_session_annotations_timeline") or []
    if len(f_actions) != len(f_timeline):
        sys.exit("follower human_actions and post_session_annotations_timeline length mismatch")

    g_actions_full = guide_blob.get("human_actions") or []
    g_timeline = guide_blob.get("post_session_annotations_timeline") or []
    if len(g_actions_full) != len(g_timeline):
        sys.exit("guide human_actions and post_session_annotations_timeline length mismatch")

    # Primary: data_processed/<session_name>/files/ (from data_preprocess.py)
    asset_roots = [
        args.data_dir / args.session_name,
        DEFAULT_LEGACY_SESSIONS,
        SESSION_FILES,
    ]

    out_dir = args.out_dir or (args.data_dir / "organized" / args.session_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_board_path = out_dir / "score_board.txt"
    score_board_path.write_text(render_score_board(rows, cols, route_set), encoding="utf-8")

    follower_participant_id = str(f_actions[0].get("participant_id") or "") if f_actions else ""

    follower_out: list[dict[str, Any]] = []
    for i, act in enumerate(f_actions):
        classified = classify_follower_action(act)
        if classified is None:
            continue
        atype, content = classified
        ann = (f_timeline[i].get("annotation") or {}) if i < len(f_timeline) else {}
        row: dict[str, Any] = {
            "role": "follower",
            "timestamp": act.get("timestamp") or "",
            "action_type": atype,
            "action_content": content if atype == "message" else "",
            "map_canvas": "",
            "drawing_accuracy": "",
            "mental_model": mental_model_from_annotation(ann),
        }
        if atype in {"draw", "erase", "undo", "reset"}:
            mp = resolve_map_image_file(session_id, str(act.get("map_image") or ""), asset_roots)
            if mp and mp.is_file():
                cells = map_canvas_cells_from_image(
                    mp,
                    rows,
                    cols,
                    args.ink_threshold,
                    args.grid_cell_mode,
                    args.min_fraction,
                )
                if cells:
                    row["map_canvas"] = canvas_string_from_cells(cells, rows, cols)
                    acc = drawing_accuracy_value(cells, route_set)
                    row["drawing_accuracy"] = float(acc) if acc is not None else ""
                else:
                    row["map_canvas"] = ""
                    row["drawing_accuracy"] = ""
            else:
                row["map_canvas"] = ""
                row["drawing_accuracy"] = ""
        follower_out.append(row)

    # Guide messages with optional follower canvas between consecutive guide sends.
    all_sorted = sorted_actions_all(data)
    follower_pid = follower_participant_id

    def is_follower_map_action(a: Mapping[str, Any]) -> bool:
        if str(a.get("participant_id") or "") != follower_pid:
            return False
        cl = classify_follower_action(a)
        return bool(cl and cl[0] in {"draw", "erase", "undo", "reset"})

    guide_out: list[dict[str, Any]] = []
    prev_guide_ts: str | None = None

    for gi, act in enumerate(g_actions_full):
        if act.get("action_type") != "send_message":
            continue
        ann = (g_timeline[gi].get("annotation") or {}) if gi < len(g_timeline) else {}
        ts = str(act.get("timestamp") or "")
        row: dict[str, Any] = {
            "role": "guide",
            "timestamp": ts,
            "action_type": "message",
            "action_content": str(act.get("action_content") or ""),
            "map_canvas": "",
            "drawing_accuracy": "",
            "mental_model": mental_model_from_annotation(ann),
        }

        if canvas_visibility and prev_guide_ts is not None:
            interval_actions = [
                a
                for a in all_sorted
                if str(a.get("timestamp") or "") > prev_guide_ts
                and str(a.get("timestamp") or "") < ts
                and is_follower_map_action(a)
            ]
            if interval_actions:
                last_a = max(
                    interval_actions,
                    key=lambda x: (str(x.get("timestamp") or ""), str(x.get("action_id") or "")),
                )
                mp = resolve_map_image_file(session_id, str(last_a.get("map_image") or ""), asset_roots)
                if mp and mp.is_file():
                    cells = map_canvas_cells_from_image(
                        mp,
                        rows,
                        cols,
                        args.ink_threshold,
                        args.grid_cell_mode,
                        args.min_fraction,
                    )
                    if cells:
                        row["map_canvas"] = canvas_string_from_cells(cells, rows, cols)
                        acc = drawing_accuracy_value(cells, route_set)
                        row["drawing_accuracy"] = float(acc) if acc is not None else ""

        guide_out.append(row)
        prev_guide_ts = ts

    follower_path = out_dir / f"{args.session_name}_follower_timeline.json"
    guide_path = out_dir / f"{args.session_name}_guide_timeline.json"
    follower_path.write_text(json.dumps(follower_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    guide_path.write_text(json.dumps(guide_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {follower_path} ({len(follower_out)} actions)")
    print(f"Wrote {guide_path} ({len(guide_out)} actions)")
    print(f"Wrote {score_board_path}")


if __name__ == "__main__":
    main()
