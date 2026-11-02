"""
Stage 2 of the study data pipeline: turn one exported session bundle into ALMANAC per-role timelines.

Input: the bundle written by ``scripts/export_session_data.py`` (``*_actions_and_annotations.json``) and the
session's ``logs/<session_id>/`` folder. Output, in a folder named after the bundle next to it (or in ``--out-dir``), as ALMANAC's ``organized/<session>/``:
  - ``<name>_guide_timeline.json`` and ``<name>_follower_timeline.json``: one row per action, in ALMANAC's format
    (``role``, ``timestamp``, ``action_type``, ``action_content``, ``map_canvas``, ``drawing_accuracy``,
    ``mental_model``), for human and agent alike;
  - ``score_board.txt``: the route scoring grid.

``mental_model`` holds the participant's post-session annotation of that action, or the agent's
(``scripts/annotate_agent_actions.py``), plus ``alignment`` (Q4).

The Follower's canvas comes from its saved drawing image (a human Follower) or from replaying the route edits the
agent made (an agent Follower: each ``send_follower_update`` is matched to its turn and the turn's
``keep_until`` / ``add`` is applied in order). Guide rows carry the Follower's latest canvas only in c2, where the
Guide could see the drawing.

  docker compose exec backend python -m scripts.build_study_timelines exports/<file>_actions_and_annotations.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from agent.map_task_as_follower.trajectory_snapshot import apply_route_edit  # noqa: E402
from agent.map_task_as_guide.map_processing.data_organization import (  # noqa: E402
    canvas_string_from_cells,
    classify_follower_action,
    drawing_accuracy_value,
    map_canvas_cells_from_image,
    mental_model_from_annotation,
    render_score_board,
    resolve_map_image_file,
)
from config.map_sets import MAP_SETS  # noqa: E402
from scripts.annotate_agent_actions import _read_jsonl, turn_for_action  # noqa: E402

BUNDLE_SUFFIX = '_actions_and_annotations.json'
INK_THRESHOLD = 10  # data_organization.py's default


def _is_agent(action: Mapping[str, Any]) -> bool:
    return action.get('is_human') in (False, 'False', 'false')


def annotations_by_action(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every annotation in the bundle (participants' and the agent's), by action_id."""
    out: Dict[str, Dict[str, Any]] = {}
    for key in ('post_session_annotations', 'agent_post_session_annotations'):
        for timeline in (bundle.get(key) or {}).values():
            for row in timeline or []:
                if row.get('action_id'):
                    out[row['action_id']] = row.get('annotation') or {}
    return out


def mental_model(annotation: Mapping[str, Any]) -> Dict[str, str]:
    return {**mental_model_from_annotation(annotation), 'alignment': str(annotation.get('alignment_q4') or '')}


def route_edit(parsed: Any) -> Optional[tuple[Any, Any]]:
    """(keep_until, add) of a turn's final result: the grounding Follower's or the plain Follower's format."""
    if not isinstance(parsed, dict):
        return None
    if 'keep_until' in parsed and (parsed.get('execute') or parsed.get('decision') == 'execute'):
        return parsed['keep_until'], parsed.get('add') or []
    for action in parsed.get('actions') or []:
        if isinstance(action, dict) and action.get('type') == 'send_follower_update' and 'keep_until' in action:
            return action['keep_until'], action.get('add') or []
    return None


def load_grids(map_set: Optional[str]) -> tuple[int, int, set[tuple[int, int]], Path]:
    """Grid size and target route (Guide grid) and the Follower grid path of the session's map set."""
    if map_set not in MAP_SETS:
        sys.exit(f'Unknown map set {map_set!r}; re-export the session so the bundle has its condition.')
    guide = json.loads(Path(MAP_SETS[map_set]['guide'][1]).read_text(encoding='utf-8'))
    rows, cols = (int(v) for v in guide['grid_size'])
    route = {(int(r), int(c)) for r, c in guide['route_cells']}
    return rows, cols, route, Path(MAP_SETS[map_set]['follower'][1])


def follower_rows(
    bundle: Mapping[str, Any],
    follower: Mapping[str, Any],
    logs_dir: Path,
    grids: tuple[int, int, set[tuple[int, int]], Path],
    annotations: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows, cols, route, follower_grid = grids
    session_id = str(bundle.get('session_id') or '')
    actions = [a for a in bundle.get('actions_all') or [] if a.get('participant_id') == follower.get('id')]
    agent = str(follower.get('type') or '').lower() in ('ai', 'ai_agent')
    trace_path = logs_dir / session_id / 'llm_requests.jsonl'
    trace = _read_jsonl(str(trace_path)) if agent and trace_path.exists() else []
    progress: Dict[str, Any] = {}

    def row(action: Mapping[str, Any], action_type: str, content: str = '', cells: Optional[list] = None) -> Dict[str, Any]:
        accuracy = drawing_accuracy_value(cells, route) if cells else None
        return {
            'role': 'follower',
            'timestamp': action.get('timestamp') or '',
            'action_type': action_type,
            'action_content': content,
            'map_canvas': canvas_string_from_cells(cells, rows, cols) if cells else '',
            'drawing_accuracy': float(accuracy) if accuracy is not None else '',
            'mental_model': mental_model(annotations.get(action.get('action_id') or '', {})),
        }

    out: List[Dict[str, Any]] = []
    for action in actions:
        if not agent:
            classified = classify_follower_action(action)
            if classified is None:
                continue
            action_type, content = classified
            cells = None
            if action_type != 'message':
                image = resolve_map_image_file(session_id, str(action.get('map_image') or ''), [logs_dir])
                cells = map_canvas_cells_from_image(image, rows, cols, INK_THRESHOLD, 'trajectory', 0.05) if image else None
                if not image:
                    print(f'warning: drawing image not found for {action.get("action_id")}', file=sys.stderr)
            out.append(row(action, action_type, content if action_type == 'message' else '', cells))
        elif action.get('action_type') == 'message':
            out.append(row(action, 'message', str(action.get('action_content') or '')))
        elif action.get('action_type') == 'send_follower_update':
            # The route edit, then the message that went with it (one action, two ALMANAC rows).
            turn_id = turn_for_action(dict(action), trace) if trace else None
            final = [r for r in trace if r.get('event') == 'result' and r.get('turn_id') == turn_id
                     and r.get('stage') in ('grounding.follower_turn', 'agent.action')]
            edit = route_edit(final[-1].get('parsed_result')) if final else None
            if edit is None:
                print(f'warning: route edit not found for {action.get("action_id")}', file=sys.stderr)
            else:
                try:
                    progress, _ = apply_route_edit(progress, edit[0], edit[1], follower_grid)
                except ValueError as error:
                    print(f'warning: route edit for {action.get("action_id")} not replayed: {error}', file=sys.stderr)
            cells = [[int(v) for v in cell.split(',')] for cell in progress.get('filledCells') or []]
            out.append(row(action, 'erase' if edit is not None and not edit[1] else 'draw', cells=cells))
            out.append(row(action, 'message', str(action.get('action_content') or '')))
    return out


def guide_rows(
    bundle: Mapping[str, Any],
    guide: Mapping[str, Any],
    follower_timeline: List[Dict[str, Any]],
    canvas_visible: bool,
    annotations: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    for action in bundle.get('actions_all') or []:
        if action.get('participant_id') != guide.get('id') or action.get('action_type') not in ('send_message', 'send_map_guidance'):
            continue
        timestamp = str(action.get('timestamp') or '')
        row = {
            'role': 'guide',
            'timestamp': timestamp,
            'action_type': 'message',
            'action_content': str(action.get('action_content') or ''),
            'map_canvas': '',
            'drawing_accuracy': '',
            'mental_model': mental_model(annotations.get(action.get('action_id') or '', {})),
        }
        if canvas_visible and previous is not None:
            # As in ALMANAC: the Follower's latest drawing since the Guide's previous message.
            drawn = [r for r in follower_timeline
                     if r['map_canvas'] and previous < r['timestamp'] < timestamp]
            if drawn:
                row['map_canvas'], row['drawing_accuracy'] = drawn[-1]['map_canvas'], drawn[-1]['drawing_accuracy']
        out.append(row)
        previous = timestamp
    return out


def build(bundle: Mapping[str, Any], logs_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """The Guide timeline, the Follower timeline and the score board of one session bundle."""
    condition = bundle.get('condition') or {}
    participants = bundle.get('participants') or []
    by_role = {str(p.get('role') or '').lower(): p for p in participants}
    if 'guide' not in by_role or 'follower' not in by_role:
        sys.exit('The bundle has no guide/follower participants; re-export the session.')
    grids = load_grids(condition.get('map_set'))
    annotations = annotations_by_action(bundle)
    follower = follower_rows(bundle, by_role['follower'], logs_dir, grids, annotations)
    guide = guide_rows(bundle, by_role['guide'], follower, condition.get('canvas_condition') == 'c2', annotations)
    rows, cols, route, _ = grids
    return guide, follower, render_score_board(rows, cols, route)


def main() -> int:
    parser = argparse.ArgumentParser(description='Build ALMANAC per-role timelines from one exported session bundle')
    parser.add_argument('bundle', type=Path, help='*_actions_and_annotations.json from export_session_data.py')
    parser.add_argument('--logs-dir', type=Path, default=Path(_BACKEND_ROOT) / 'logs')
    parser.add_argument('--out-dir', type=Path, default=None, help='default: <bundle folder>/<bundle name>/')
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
    guide, follower, score_board = build(bundle, args.logs_dir)
    name = args.bundle.name[:-len(BUNDLE_SUFFIX)] if args.bundle.name.endswith(BUNDLE_SUFFIX) else args.bundle.stem
    out_dir = args.out_dir or args.bundle.parent / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (('guide_timeline', guide), ('follower_timeline', follower)):
        path = out_dir / f'{name}_{suffix}.json'
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Wrote {path} ({len(rows)} rows)')
    (out_dir / 'score_board.txt').write_text(score_board, encoding='utf-8')
    print(f'Wrote {out_dir / "score_board.txt"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
