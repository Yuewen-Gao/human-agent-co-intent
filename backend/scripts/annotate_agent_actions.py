"""
Annotate an AI agent's actions in one session the way participants annotate their own (ALMANAC post-session schema).

Offline and separate from the agent: nothing here runs during a session or feeds back into the agent. The script reads
``logs/<session_id>/`` (the agent's action log and ``llm_requests.jsonl``) and, for each action the agent actually took
(a turn that sent nothing has no action and is not annotated), makes one LLM call with:
  - the agent's task instructions and what it could see at that turn (the logged request it worked from),
  - the action it took.
The answers are written to ``logs/<session_id>/post_annotations_agent.json``, keyed by action_id, in the same saved
format as the participants' post-session annotations, so the export reads both the same way.

An action is matched to its turn by the action_id the turn's result records (the Guide), else by the turn whose final
result immediately precedes the action (the Follower: same timestamp).

Run inside the backend container (it has the API key):
  docker compose exec backend python -m scripts.annotate_agent_actions --session-id <id> --dry-run
  docker compose exec backend python -m scripts.annotate_agent_actions --session-id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

PROMPT_FILE = os.path.join(_BACKEND_ROOT, 'agent', 'prompts', 'agent_annotation_prompt.txt')
OUTPUT_NAME = 'post_annotations_agent.json'
FINAL_STAGES = ('agent.action', 'grounding.repair', 'grounding.follower_turn')
GAME_STATE_MARKER = '\n\n<CURRENT GAME STATE>\n'
EVIDENCE_OPEN, EVIDENCE_CLOSE = '<CURRENT OBSERVABLE EVIDENCE>', '</CURRENT OBSERVABLE EVIDENCE>'
MAX_FOLLOWER_GAP_SECONDS = 10.0

# The participants' options (frontend/src/views/PostSessionAnnotation.vue), id -> text as saved.
LABELS = {
    'q1': {'t1': 'Still figuring out what we needed to do', 't2': 'Working toward a shared understanding',
           't3': 'Clear on what to do and working on it', 't4': 'Something was unclear and we were working it out',
           'other': 'Other'},
    'q2': {'p1': 'Understood the situation and we were on the same page',
           'p2': 'Probably understood our situation but I was not fully sure',
           'p3': 'Is waiting for more information to understand the situation',
           'p4': 'Misunderstood and we were not aligned', 'p5': 'Gave no clear signal either way', 'other': 'Other'},
    'q3': {'r1': 'Executing a plan we already agreed on', 'r2': 'Exploring on my own to gather information',
           'r3': 'Confirming the situation with my partner',
           'r4': 'Grounding by sharing or requesting information to align',
           'r5': 'Repairing a mistake or misunderstanding', 'r6': 'Waiting for more information', 'other': 'Other'},
}
Q4_REASONS = {'n1': 'Different understanding of the task goal', 'n2': 'Different understanding of the current state',
              'n3': 'One of us was missing key information', 'n4': 'Communication was unclear or ambiguous',
              'n5': 'Technical or interface issue got in the way', 'other': 'Other'}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_session(session_dir: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The agent's actions (oldest first) and the session's LLM trace rows."""
    actions: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(session_dir)):
        if name.endswith('.jsonl') and name != 'llm_requests.jsonl':
            actions += [row for row in _read_jsonl(os.path.join(session_dir, name)) if row.get('is_human') is False]
    trace_path = os.path.join(session_dir, 'llm_requests.jsonl')
    trace = _read_jsonl(trace_path) if os.path.exists(trace_path) else []
    return sorted(actions, key=lambda a: a.get('timestamp') or ''), trace


def turn_for_action(action: Dict[str, Any], trace: List[Dict[str, Any]]) -> Optional[str]:
    """The turn that produced the action: by recorded action_id, else the final result just before it."""
    action_id = action.get('action_id')
    for row in trace:
        if row.get('event') != 'result':
            continue
        executed = row.get('execution_result')
        if isinstance(executed, dict) and any(
                isinstance(s, dict) and s.get('action_id') == action_id for s in executed.get('successful') or ()):
            return row.get('turn_id')
    at = _time(action['timestamp'])
    earlier = [row for row in trace
               if row.get('event') == 'result' and row.get('stage') in FINAL_STAGES
               and row.get('participant_id') == action.get('participant_id') and _time(row['timestamp']) <= at]
    if not earlier:
        return None
    last = max(earlier, key=lambda row: row['timestamp'])
    return last.get('turn_id') if (at - _time(last['timestamp'])).total_seconds() <= MAX_FOLLOWER_GAP_SECONDS else None


def _text(request: Dict[str, Any]) -> str:
    content = (request.get('request_messages') or [{}])[-1].get('content')
    if isinstance(content, str):
        return content
    return '\n'.join(part.get('text', '') for part in content or () if isinstance(part, dict) and part.get('type') == 'text')


def turn_context(turn_id: str, trace: List[Dict[str, Any]], role: str = 'Guide') -> Dict[str, Any]:
    """Task instructions and what the agent saw in this turn.

    The grounding agent's intermediate stage results are left out, so both agent types are annotated from the same
    information.

    The plain agent's own planning, action space and output format are left out of the task instructions, as the
    grounding stages leave them out (`guide_evidence` / `follower_evidence`): they ask for a different JSON.
    """
    from agent.map_task_as_guide.grounding_as_guide.orchestrator import follower_evidence, guide_evidence
    rows = [row for row in trace if row.get('turn_id') == turn_id]
    requests = {row.get('stage'): row for row in rows if row.get('event') == 'request'}
    if 'grounding.h' in requests:  # the grounding stages share one evidence block
        text = _text(requests['grounding.h'])
        start, end = text.find(EVIDENCE_OPEN), text.find(EVIDENCE_CLOSE)
        evidence = text[start + len(EVIDENCE_OPEN):end].strip('\n') if start >= 0 and end > start else text
    else:
        final = next((requests[s] for s in FINAL_STAGES if s in requests), None)
        evidence = _text(final) if final else ''
    evidence = (follower_evidence if role == 'Follower' else guide_evidence)(evidence)
    task, marker, seen = evidence.partition(GAME_STATE_MARKER)
    return {'task': task.strip(), 'seen': (marker + seen).strip() if marker else ''}


def _role(action: Dict[str, Any], trace: List[Dict[str, Any]], turn_id: str) -> str:
    state = ((action.get('session_status') or {}).get('private_state') or {}).get('map') or {}
    role = str(state.get('role') or '').lower()
    if role in ('guide', 'follower'):
        return role.capitalize()
    stages = {row.get('stage') for row in trace if row.get('turn_id') == turn_id}
    return 'Follower' if 'grounding.follower_turn' in stages else 'Guide'


def build_prompt(role: str, context: Dict[str, Any], action: Dict[str, Any]) -> str:
    with open(PROMPT_FILE, encoding='utf-8') as f:
        instruction = f.read().replace('{role}', role)
    taken = {'action_type': action.get('action_type'), 'content': action.get('action_content'),
             'details': action.get('metadata') or {}}
    return (
        '<TASK INSTRUCTIONS>\n' + context['task'] + '\n</TASK INSTRUCTIONS>\n\n'
        + '<WHAT YOU COULD SEE AT THIS MOMENT>\n' + (context['seen'] or 'not recorded') + '\n</WHAT YOU COULD SEE AT THIS MOMENT>\n\n'
        + '<THE ACTION YOU TOOK>\n' + json.dumps(taken, ensure_ascii=False) + '\n</THE ACTION YOU TOOK>\n\n'
        + '<ANNOTATION>\n' + instruction + '\n</ANNOTATION>'
    )


def to_saved_format(answer: Dict[str, Any]) -> Dict[str, Any]:
    """The participants' saved fields (as PostSessionAnnotation.vue stores them)."""
    saved: Dict[str, Any] = {'momentExplanationTranscript': str(answer.get('explanation') or '').strip()}
    for q, labels in LABELS.items():
        label_id = str(answer.get(q) or '').strip().lower()
        label_id = label_id if label_id in labels else 'other'
        saved[q] = labels[label_id]
        saved[f'{q}LabelId'] = label_id
        saved[f'{q}OtherText'] = str(answer.get(f'{q}_other') or '').strip() if label_id == 'other' else ''
    same_page = str(answer.get('q4') or '').strip().lower()
    saved['q4'] = same_page if same_page in ('yes', 'no') else ''
    reasons = [str(r).strip().lower() for r in answer.get('q4_reasons') or ()] if saved['q4'] == 'no' else []
    saved['q4Reasons'] = [Q4_REASONS[r] for r in reasons if r in Q4_REASONS]
    saved['q4OtherText'] = str(answer.get('q4_other') or '').strip() if 'other' in reasons else ''
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description='Annotate an agent\'s actions with the participants\' post-session schema')
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--logs-dir', default=os.path.join(_BACKEND_ROOT, 'logs'))
    parser.add_argument('--limit', type=int, help='annotate at most this many new actions')
    parser.add_argument('--dry-run', action='store_true', help='print the first prompt and the counts; no LLM call')
    args = parser.parse_args()

    session_dir = os.path.join(args.logs_dir, args.session_id)
    actions, trace = load_session(session_dir)
    out_path = os.path.join(session_dir, OUTPUT_NAME)
    done: Dict[str, Any] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            done = json.load(f)
    todo = []
    for action in actions:
        if action.get('action_id') in done:
            continue
        turn_id = turn_for_action(action, trace)
        if not turn_id:
            print('no turn found for action', action.get('action_id'), action.get('action_type'), file=sys.stderr)
            continue
        todo.append((action, turn_id))
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f'agent actions {len(actions)}, already annotated {len(done)}, to annotate {len(todo)} (one LLM call each)')
    if args.dry_run:
        if todo:
            action, turn_id = todo[0]
            role = _role(action, trace, turn_id)
            print(build_prompt(role, turn_context(turn_id, trace, role), action))
        return 0

    from agent.llm_client import create_llm_client
    client = create_llm_client()
    model = getattr(client, 'default_model', None)
    for action, turn_id in todo:
        role = _role(action, trace, turn_id)
        prompt = build_prompt(role, turn_context(turn_id, trace, role), action)
        raw = client.chat_completions_create([{'role': 'user', 'content': prompt}], max_tokens=4096,
                                             response_format={'type': 'json_object'})
        try:
            answer = json.loads(raw)
        except (TypeError, ValueError):
            print('unparsed answer for', action.get('action_id'), file=sys.stderr)
            continue
        done[action['action_id']] = {**to_saved_format(answer), 'turn_id': turn_id, 'annotator_model': model}
        with open(out_path, 'w', encoding='utf-8') as f:  # after each action, so an interrupted run can resume
            json.dump(done, f, ensure_ascii=False, indent=2)
        print('annotated', action.get('action_id'), done[action['action_id']]['q1LabelId'],
              done[action['action_id']]['q2LabelId'], done[action['action_id']]['q3LabelId'], done[action['action_id']]['q4'])
    print('wrote', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
