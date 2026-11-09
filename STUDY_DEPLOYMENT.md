# Map Task Study: Deployment Handover

For whoever deploys the study site: how to set the conditions, which APIs to configure, where each session's data is stored, and how to export it. Current as of 2026-11-06 (map set H2).

## 1. Study setup

Each session pairs one human participant with one AI agent on the Map Task, text chat only. Two things are fixed and two are switched.

- **Fixed:** visibility condition **C1** (the Guide cannot see the Follower's drawing), with the **Awareness Dashboard off**.
- **Fixed:** map set `H2` (images `backend/map_assets/H2_*.png`, grids `backend/map_grids/H2_*.json`).
- **Switched:** whether the agent is the **Guide or the Follower**, and whether it is the **grounding or the plain** agent.

| Condition | `MAPTASK_AGENT_ROLE` | `GROUNDING_TREATMENT` | Human plays |
| --- | --- | --- | --- |
| Agent Guide, grounding | `"guide"` | `True` | Follower |
| Agent Guide, plain | `"guide"` | `False` | Follower |
| Agent Follower, grounding | `"follower"` | `True` | Guide |
| Agent Follower, plain | `"follower"` | `False` | Guide |

**What everyone is told about the maps.** Both agent prompts state, in their rules (line 8 of `backend/agent/prompts/maptask_agent_prompt.txt` and `maptask_follower_agent_prompt.txt`):

> The two maps may differ in three landmarks, including misplaced, misnamed, and duplicate landmarks.

Read this sentence to every participant, word for word, during onboarding, so the human and the agent start with the same information. The app's Rules button shows nothing, so onboarding is the only place participants get it. The sentence describes map set H2; if the maps change, change it in both prompts and in the onboarding script.

## 2. Must do before deploying

- [x] **Awareness Dashboard off.** Done in the code: the public session template, `backend/services/public_session_template.py`, sets `"awarenessDashboard": {"enabled": False, "items": []}`. Keep it off: with it on, a human Guide sees the Follower's drawing in the Info Dashboard, which breaks C1.
- [ ] **Turn on the LLM trace log:** set `LLM_TRACE_LOGGING=1` in `backend/.env`. It is off by default; without it there is no `llm_requests.jsonl`, and the agent's post-session annotation cannot be generated.
- [ ] **Keep and back up `backend/logs/`.** The LLM trace and the agent's annotations exist only there, not in the database. Docker mounts it to the host as `./backend/logs:/app/logs`.
- [ ] **Leave `GUIDE_SEES_FOLLOWER_DRAWING = False`** in `backend/config/grounding_treatment.py`. This is the agent side of C1.
- [x] **In-session spoken checkpoints are off.** They fire at a share of the target route length, and H2 has no target length in `route_pixel_ratio_from_map_filename` (`backend/functions.py`), so they never fire. Do not add one for H2. Do not turn off the in-session annotation switch in the session's experiment config either: it also makes the session untimed, ends the task when the participant presses **Submit task** (the agent does not press it), and limits the agent's periodic checks.

## 3. Switching conditions

The condition is a server-wide setting in `backend/config/grounding_treatment.py`; change it and rebuild the backend.

```python
ENABLE_GROUNDING_TREATMENT = True     # keep True
MAPTASK_AGENT_ROLE = "guide"          # "guide" or "follower": the agent's role; the human gets the other
GUIDE_SEES_FOLLOWER_DRAWING = False   # C1, fixed
GROUNDING_TREATMENT = True            # True: grounding agent; False: plain agent
```

```bash
docker compose up -d --build backend
```

- The setting applies to every new session on that server, so only one condition runs at a time. Run conditions in blocks, or deploy one instance per condition.
- When the human joins a session, the server assigns roles from `MAPTASK_AGENT_ROLE`.
- A single session can override grounding on or off with `interaction.groundingTreatment` (a boolean); the role can only be set on the server.
- The map set is `ACTIVE_MAP_SET = "H2"` in `backend/config/map_sets.py`. Do not change it.

## 4. APIs and environment variables

All go in `backend/.env`, which Docker reads through `env_file`. Never commit keys to the repository.

| Purpose | Variables | Notes |
| --- | --- | --- |
| Agent model | `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-5.6-sol` | Official OpenAI API. `OPENAI_REASONING_EFFORT` is optional (unset = model default); `OPENAI_BASE_URL` is normally unset. The agent-annotation script uses the same settings. |
| Speech transcription (spoken post-session rationale) | `OPENAI_API_KEY` | Uses OpenAI `whisper-1`. Azure also works: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_WHISPER_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`; when set, Azure is used first. |
| Database | `DATABASE_URL` (or `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`), `PGSCHEMA` | PostgreSQL. Defaults to the Postgres bundled in docker compose; set `DATABASE_URL` for an external one such as RDS. `PGSCHEMA` defaults to `humanagent_collab`. |
| LLM trace log | `LLM_TRACE_LOGGING=1` | Required; see section 2. |
| Flask session key | `FLASK_SECRET_KEY` | Set a random string, not the default. |
| Screenshots in S3 (optional) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET` | Action screenshots and page snapshots for annotation. Without S3 they stay in `backend/logs/<session_id>/files/`. |

## 5. Deployment

The study needs both a database and the local `backend/logs/` folder: the database holds actions, sessions and annotations; the folder holds the LLM trace and the agent's annotations.

```bash
# All in one: bundled Postgres + backend + frontend
docker compose up -d --build

# External database (e.g. AWS RDS), with DATABASE_URL in .env
docker compose -f docker-compose.rds.yml --env-file .env up --build
```

Ports: backend `${BACKEND_PORT:-5001}` → container 5000; frontend `${FRONTEND_PORT:-8080}` → container 80.

## 6. Running each session

- **Onboarding:** explain the rules and the interface, and read the map-difference sentence from section 1 word for word.
- **Task link (Prolific):** set the study's task URL to `https://<host>/login?PROLIFIC_PID={{%PROLIFIC_PID%}}` and turn on Prolific's URL parameters. The login page then logs the participant in with their Prolific ID as the session key, so they type nothing; the session's name is their Prolific ID, and reopening the link returns to the same session. Without `PROLIFIC_PID` in the URL, the page shows the manual session-key form (for testing).
- **Participant ID:** the pre- and post-questionnaires (Qualtrics) start with a participant ID question, answered with the Prolific ID. Record each participant ID together with the session ID and the condition, so the questionnaires can be joined to the session data.
- **End of the task:** the participant presses **Submit task**. This ends the task and opens the post-session annotation. The session's status reads `paused` either way; whether Submit was pressed is stored in the session's `maptask_submit_confirmed_ids`.
- **Post-session annotation:** the participant then annotates their own actions (spoken rationale plus Q1–Q4).

## 7. Data each session produces

Files are under `backend/logs/<session_id>/`.

| Data | Database table | File |
| --- | --- | --- |
| Every action: message, draw, erase, undo, reset (human and agent) | `action_logs` | `<participant_id>.jsonl` |
| Session configuration (condition, maps, participants) | `research_sessions` | — |
| In-session spoken checkpoints | `in_session_annotations` | none under H2 (off, see section 2) |
| Human post-session annotation (spoken rationale + Q1–Q4) | `post_session_annotations` | `post_annotations_<participant_id>.json` |
| Action screenshots and page snapshots | — | `files/` (or S3) |
| Every agent LLM call (request, response, intermediate judgements) | — | `llm_requests.jsonl` (needs `LLM_TRACE_LOGGING=1`) |
| Agent post-session annotation (offline script, same format as the human's) | — | `post_annotations_agent.json` |

## 8. After data collection

Run these four steps for each session once the participant has finished their post-session annotation. They do not need to run right after the session: the agent's annotation is generated offline, from `backend/logs/<session_id>/` alone, so it can run in one batch after data collection, as long as that folder is kept.

1. **Generate the agent's post-session annotation.** Run with `--dry-run` first to see how many actions it will annotate, without any model call. The real run makes one model call per action, about 15–40 per session. It annotates only actions the agent actually took, so 15-second checks that sent nothing are skipped. A rerun skips actions already annotated.

    ```bash
    docker compose exec backend python -m scripts.annotate_agent_actions --session-id <session_id> --dry-run
    docker compose exec backend python -m scripts.annotate_agent_actions --session-id <session_id>
    ```

2. **Export.** Output goes to `backend/exports/`: `<name>_<first 8 of id>_user_actions.json` (all human actions) and `<name>_<first 8 of id>_actions_and_annotations.json`. The second file holds:

    | Key | Contents |
    | --- | --- |
    | `condition` | `agent_role` (guide / follower), `agent_type` (grounding / plain), `canvas_condition` (c1 / c2), `map_set`, `maps`, plus the raw flags `guide_sees_follower_drawing` and `awareness_dashboard_enabled` |
    | `participants` | Each participant's `id`, `name`, `type` (human / ai) and `role` |
    | `actions_all` | Every action, human and agent, in time order |
    | `human_actions_all` | Human actions only |
    | `post_session_annotations` | Human post-session annotations, by participant name |
    | `agent_post_session_annotations` | The agent's annotations from step 1 |
    | `in_session_annotations_all` | Spoken checkpoints (empty under H2) |

    The condition is read from the session's own record, which stores it when the session is created. Sessions created before 2026-11-02 did not store it, so their `canvas_condition` can show `unknown`.

    ```bash
    docker compose exec backend python -m scripts.export_session_data --session-id <session_id>
    # or by name: --session-name "<session name>"
    ```

3. **Build the per-role timelines** (ALMANAC format). This reads the file from step 2 and `backend/logs/<session_id>/`, and writes the results to a folder named after the export file: `<name>_guide_timeline.json`, `<name>_follower_timeline.json` and `score_board.txt`. It makes no model calls.

    ```bash
    docker compose exec backend python -m scripts.build_study_timelines exports/<name>_<first 8 of id>_actions_and_annotations.json
    ```

    Each row is one action, with the same fields for human and agent:

    | Field | Meaning |
    | --- | --- |
    | `role` | guide / follower |
    | `timestamp` | When the action happened |
    | `action_type` | message / draw / erase / undo / reset |
    | `action_content` | The message text (empty for drawing actions) |
    | `map_canvas` | The Follower's drawing as a 60×40 grid of `x` and `.` |
    | `drawing_accuracy` | Mean score of the drawn cells: 1 on the route, 2/3 one cell away, 1/3 two cells away, 0 farther |
    | `mental_model` | The post-session annotation of this action: `rationale`, `team_goal`, `partner_intent`, `self_reasoning`, `alignment` |

    How the drawing is filled in:
    - **Human Follower:** read from the saved drawing image.
    - **Agent Follower:** rebuilt by replaying its route edits from `llm_requests.jsonl`. One agent route update becomes two rows, a draw row and the message sent with it.
    - **Guide rows:** carry the Follower's latest drawing only in c2. In C1, the study setting, they carry none.

4. **Back up the raw folder.** Copy the whole `backend/logs/<session_id>/`. The export does not include `llm_requests.jsonl`; that file exists only here.

Each post-session annotation has the same fields for human and agent, matching the ALMANAC data:

| Field | Meaning |
| --- | --- |
| `explanation_transcription` | Spoken rationale (transcribed) |
| `task_model_q1` | Q1 team goal |
| `partner_model_q2` | Q2 partner intent |
| `self_model_q3` | Q3 self-reasoning |
| `alignment_q4` | Q4 on the same page: yes / no |
