"""
Export session data from PostgreSQL.

1) User (human) actions only — full payloads as stored in action_logs.
2) Full bundle — human actions plus in-session and post-session annotations.

Resolve session by:
  - ``--session-id`` (UUID): use directly.
  - ``--session-name``: lookup ``research_sessions.session_name``; if that table is missing
    or empty, fall back to ``action_logs.payload->>'session_name'``.

Run locally (point DATABASE_URL at Postgres, e.g. host port mapped from Docker):
  cd backend && python -m scripts.export_session_data --session-name "test1"

Run inside the backend container (same DB as the app; recommended when DB only exists in Docker):
  docker compose exec backend python -m scripts.export_session_data --session-name "test1"

Copy exports from container to host (default output: /app/exports in the image):
  docker compose cp backend:/app/exports ./backend/exports

Requires DATABASE_URL or PG* env. Loads backend/.env if python-dotenv is installed.

SSH tunnel to Docker Postgres on the server:

- Option A: ``export DATABASE_URL='postgresql://USER:PASS@127.0.0.1:LOCALPORT/humanagent'``
  before running (overrides docker-only ``postgres`` hostname in ``.env``).
- Option B: keep ``.env`` as ``...@postgres:5432/...`` and run
  ``export EXPORT_PG_TUNNEL_PORT=15432`` (match ``ssh -L 15432:127.0.0.1:5432 ...``);
  this script rewrites host to ``127.0.0.1`` and port to that value.

For localhost / tunnel targets, this script sets ``PGSSLMODE=disable`` and forces
``sslmode=disable`` on ``DATABASE_URL`` as well (URI sslmode overrides env for libpq).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_BACKEND_ROOT, '.env'))
except ImportError:
    pass


def _rewrite_docker_postgres_for_ssh_tunnel() -> None:
    """
    .env for Compose uses host ``postgres``; that name does not resolve on a laptop.
    With ``EXPORT_PG_TUNNEL_PORT`` set (local end of ``ssh -L LOCAL:127.0.0.1:5432``),
    rewrite URL or PG* so connections go through the tunnel.
    """
    port_s = (os.environ.get('EXPORT_PG_TUNNEL_PORT') or '').strip()
    if not port_s:
        return
    try:
        tunnel_port = int(port_s)
    except ValueError:
        return

    raw_url = (os.environ.get('DATABASE_URL') or '').strip()
    if raw_url:
        try:
            from sqlalchemy.engine.url import make_url

            u = make_url(raw_url)
            if u.host and str(u.host).lower() == 'postgres':
                u = u.set(host='127.0.0.1', port=tunnel_port)
                os.environ['DATABASE_URL'] = u.render_as_string(hide_password=False)
        except Exception:
            pass

    if (os.environ.get('PGHOST') or '').strip().lower() == 'postgres':
        os.environ['PGHOST'] = '127.0.0.1'
        os.environ['PGPORT'] = str(tunnel_port)


_rewrite_docker_postgres_for_ssh_tunnel()


def _disable_ssl_for_local_db_host() -> None:
    """
    .env often sets PGSSLMODE=require and sslmode=require on DATABASE_URL for RDS.
    Docker Postgres has no TLS; libpq prefers URI sslmode over PGSSLMODE, so both must
    be cleared for SSH -L to 127.0.0.1 or exports get "server closed the connection".
    """
    url = (os.environ.get('DATABASE_URL') or '').strip()
    if url:
        try:
            from sqlalchemy.engine.url import make_url

            u = make_url(url)
            h = (str(u.host or '')).lower()
            if h in ('127.0.0.1', 'localhost', '::1'):
                os.environ['PGSSLMODE'] = 'disable'
                u = u.update_query_dict({'sslmode': 'disable', 'gssencmode': 'disable'})
                os.environ['DATABASE_URL'] = u.render_as_string(hide_password=False)
                return
        except Exception:
            normalized = url.replace('postgresql+psycopg2://', 'postgresql://', 1)
            try:
                parsed = urlparse(normalized)
                h = (parsed.hostname or '').lower()
                if h in ('127.0.0.1', 'localhost', '::1'):
                    os.environ['PGSSLMODE'] = 'disable'
                    sep = '&' if parsed.query else '?'
                    low = url.lower()
                    if 'sslmode=' not in low:
                        os.environ['DATABASE_URL'] = url + sep + 'sslmode=disable'
                    return
            except ValueError:
                pass
    host = (os.environ.get('PGHOST') or '').strip().lower()
    if host in ('127.0.0.1', 'localhost', '::1'):
        os.environ['PGSSLMODE'] = 'disable'


_disable_ssl_for_local_db_host()


def _safe_filename_part(name: str) -> str:
    s = re.sub(r'[^\w\-]+', '_', name.strip())[:80]
    return s or 'session'


def _resolve_session_ids(session_name: str) -> List[str]:
    from sqlalchemy.exc import ProgrammingError

    from services.db import (
        find_session_ids_by_name,
        find_session_ids_by_name_from_action_logs,
    )

    name = session_name.strip()
    ids: List[str] = []
    try:
        ids = find_session_ids_by_name(name)
    except ProgrammingError as e:
        err = str(e).lower()
        if 'research_sessions' in err or 'does not exist' in err:
            ids = []
        else:
            raise
    if not ids:
        ids = find_session_ids_by_name_from_action_logs(name)
    return ids


def _build_participant_name_map(participant_ids: List[str], human_actions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map participant_id -> readable participant name; keep keys unique."""
    base_names: Dict[str, str] = {}
    for pid in participant_ids:
        base_names[pid] = ''
    for e in human_actions:
        pid = e.get('participant_id')
        if not pid or pid not in base_names:
            continue
        nm = (e.get('participant_name') or '').strip()
        if nm and not base_names[pid]:
            base_names[pid] = nm

    unnamed_index = 1
    for pid in participant_ids:
        if not base_names[pid]:
            base_names[pid] = f'participant {unnamed_index}'
            unnamed_index += 1

    # De-duplicate same visible names by appending a short id suffix.
    counts = collections.Counter(base_names.values())
    out: Dict[str, str] = {}
    for pid in participant_ids:
        nm = base_names[pid]
        if counts[nm] > 1:
            out[pid] = f'{nm} ({pid[:8]})'
        else:
            out[pid] = nm
    return out


def _format_in_session_by_participant(
    in_session_all: List[Dict[str, Any]], participant_name_by_id: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in in_session_all:
        pid = r.get('participant_id')
        if not pid:
            continue
        participant_name = participant_name_by_id.get(pid, pid)
        checkpoint_key = f"checkpoint {r.get('checkpoint_index')}"
        out.setdefault(participant_name, {})[checkpoint_key] = r.get('transcription') or ''
    return out


def _format_post_timeline_for_participant(
    actions: List[Dict[str, Any]],
    annotations_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return timeline sorted by action timestamp with compact annotation fields."""
    def _pick_q_value(ann: Dict[str, Any], q_key: str) -> str:
        label_id = (ann.get(f'{q_key}LabelId') or '').strip().lower()
        label_value = (ann.get(q_key) or '').strip()
        if label_id == 'other' or label_value.lower() == 'other':
            other_text = (ann.get(f'{q_key}OtherText') or '').strip()
            if other_text:
                return other_text
        return label_value

    timeline: List[Dict[str, Any]] = []
    for a in actions:
        action_id = a.get('action_id')
        ann = annotations_payload.get(action_id, {}) if action_id else {}
        if not isinstance(ann, dict):
            ann = {}
        timeline.append(
            {
                'action_id': action_id or '',
                'action_timestamp': a.get('timestamp') or '',
                'action_type': a.get('action_type') or '',
                'action_content': a.get('action_content') or '',
                'annotation': {
                    'explanation_transcription': ann.get('momentExplanationTranscript') or '',
                    'task_model_q1': _pick_q_value(ann, 'q1'),
                    'partner_model_q2': _pick_q_value(ann, 'q2'),
                    'self_model_q3': _pick_q_value(ann, 'q3'),
                    'alignment_q4': ann.get('q4') or '',
                },
            }
        )
    return timeline


def _session_participants(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each participant's id, name, type (human / ai) and role (guide / follower) from the session record."""
    return [
        {k: p.get(k) for k in ('id', 'name', 'type', 'role')}
        for p in record.get('participants') or []
        if isinstance(p, dict)
    ]


def _session_condition(record: Dict[str, Any]) -> Dict[str, Any]:
    """The study condition of one session, from its own record.

    `canvas_condition` follows ALMANAC: c1 = the Guide cannot see the Follower's drawing, c2 = it can.
    Sessions created before the condition was recorded fall back to what the record shows (`None` = unknown).
    """
    from config.map_sets import MAP_SETS

    interaction = record.get('interaction') or {}
    agent = next((p for p in record.get('participants') or []
                  if isinstance(p, dict) and str(p.get('type') or '').lower() in ('ai', 'ai_agent')), {})
    agent_role = str(agent.get('role') or '').lower() or None
    grounding = interaction.get('groundingTreatment')
    if not isinstance(grounding, bool):
        grounding = True if record.get('grounding_decisions') else None
    dashboard = interaction.get('awarenessDashboard') or {}
    dashboard_on = bool(dashboard.get('enabled'))
    guide_sees = interaction.get('guideSeesFollowerDrawing')
    if agent_role == 'guide':  # the agent Guide's view
        visible = guide_sees if isinstance(guide_sees, bool) else None
    else:  # a human Guide sees the drawing only through the Awareness Dashboard
        visible = dashboard_on and 'Participant.map_progress' in (dashboard.get('items') or [])
    maps = {str(m.get('role')): m.get('filename') for m in (record.get('params') or {}).get('maps') or []
            if isinstance(m, dict)}
    map_set = next((name for name, files in MAP_SETS.items() if files['guide'][0] == maps.get('guide')), None)
    return {
        'agent_role': agent_role,
        'agent_type': {True: 'grounding', False: 'plain'}.get(grounding, 'unknown'),
        'canvas_condition': {True: 'c2', False: 'c1'}.get(visible, 'unknown'),
        'guide_sees_follower_drawing': guide_sees if isinstance(guide_sees, bool) else None,
        'awareness_dashboard_enabled': dashboard_on,
        'map_set': map_set,
        'maps': maps,
    }


def _agent_post_annotations(session_id: str, all_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The AI agent's actions with the annotations written by scripts/annotate_agent_actions.py, by agent name."""
    path = os.path.join(_BACKEND_ROOT, 'logs', session_id, 'post_annotations_agent.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    agent_actions = [e for e in all_entries if e.get('is_human') is False]
    out: Dict[str, Any] = {}
    for pid in sorted({e.get('participant_id') for e in agent_actions if e.get('participant_id')}):
        actions = [e for e in agent_actions if e.get('participant_id') == pid]
        name = next((e.get('participant_name') for e in actions if e.get('participant_name')), pid)
        out[name] = _format_post_timeline_for_participant(actions, payload)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Export session actions and annotations')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--session-name',
        help='Exact session_name (research_sessions or action_logs.payload.session_name)',
    )
    g.add_argument(
        '--session-id',
        help='Session UUID (skips name lookup)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(_BACKEND_ROOT, 'exports'),
        help='Directory for JSON files (created if missing)',
    )
    args = parser.parse_args()

    from services.db import (
        is_db_configured,
        list_distinct_participant_ids,
        load_all_in_session_rows_for_session,
        load_all_post_session_rows_for_session,
        load_research_session,
        load_session_logs,
    )

    if not is_db_configured():
        print('export_session_data: DATABASE_URL or PG* not set; cannot read PostgreSQL.', file=sys.stderr)
        return 1

    if args.session_id:
        session_id = args.session_id.strip()
        label = session_id
        name_for_json = ''
    else:
        name = args.session_name.strip()
        ids = _resolve_session_ids(name)
        if not ids:
            print(
                f'export_session_data: no session found for session_name={name!r} '
                '(research_sessions and action_logs.payload.session_name).',
                file=sys.stderr,
            )
            return 2
        if len(ids) > 1:
            print(
                f'export_session_data: multiple session_ids for name {name!r}: {ids}. '
                'Pass --session-id to pick one.',
                file=sys.stderr,
            )
            return 3
        session_id = ids[0]
        label = name
        name_for_json = name

    all_entries = load_session_logs(session_id)
    record = load_research_session(session_id)
    if not name_for_json and all_entries:
        name_for_json = (all_entries[0].get('session_name') or '')[:512]
    human_actions = [e for e in all_entries if e.get('is_human') is True]

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = _safe_filename_part(label) + '_' + session_id[:8]

    actions_path = os.path.join(args.output_dir, f'{prefix}_user_actions.json')
    bundle_path = os.path.join(args.output_dir, f'{prefix}_actions_and_annotations.json')

    with open(actions_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'session_name': name_for_json,
                'session_id': session_id,
                'count': len(human_actions),
                'human_actions': human_actions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    in_session_all = load_all_in_session_rows_for_session(session_id)
    post_all = load_all_post_session_rows_for_session(session_id)

    pid_set = set(list_distinct_participant_ids(session_id))
    for r in in_session_all:
        pid = r.get('participant_id')
        if pid:
            pid_set.add(pid)
    for r in post_all:
        pid = r.get('participant_id')
        if pid:
            pid_set.add(pid)
    participant_ids = sorted(pid_set)
    participant_name_by_id = _build_participant_name_map(participant_ids, all_entries)
    participant_names = [participant_name_by_id[pid] for pid in participant_ids]

    by_participant: Dict[str, Dict[str, Any]] = {}
    for pid in participant_ids:
        p_name = participant_name_by_id[pid]
        p_actions = [e for e in human_actions if e.get('participant_id') == pid]
        p_post_payload = next(
            (r.get('payload') or {} for r in post_all if r.get('participant_id') == pid),
            {},
        )
        by_participant[p_name] = {
            'human_actions': [e for e in human_actions if e.get('participant_id') == pid],
            'in_session_annotations': [r for r in in_session_all if r.get('participant_id') == pid],
            'post_session_annotations_timeline': _format_post_timeline_for_participant(
                p_actions, p_post_payload
            ),
        }

    bundle: Dict[str, Any] = {
        'session_name': name_for_json,
        'session_id': session_id,
        'participant_ids': participant_ids,
        'participant_names': participant_names,
        'participant_name_by_id': participant_name_by_id,
        'condition': _session_condition(record),
        'participants': _session_participants(record),
        'actions_all': sorted(all_entries, key=lambda e: (e.get('timestamp') or '', e.get('action_id') or '')),
        'human_actions_all': human_actions,
        'in_session_annotations_all': _format_in_session_by_participant(
            in_session_all, participant_name_by_id
        ),
        'post_session_annotations': {
            participant_name_by_id[pid]: _format_post_timeline_for_participant(
                [e for e in human_actions if e.get('participant_id') == pid],
                next((r.get('payload') or {} for r in post_all if r.get('participant_id') == pid), {}),
            )
            for pid in participant_ids
        },
        'by_participant': by_participant,
        'agent_post_session_annotations': _agent_post_annotations(session_id, all_entries),
    }

    with open(bundle_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print('Wrote', actions_path)
    print('Wrote', bundle_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
