"""Static approved few-shot retrieval; never invokes offline analysis at runtime."""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts"
_H_FEWSHOTS_JSON = _PROMPT_DIR / "h_fewshots.json"
_REPAIR_FEWSHOTS_JSON = _PROMPT_DIR / "repair_fewshots.json"
_REPAIR_ASSOCIATIONS_JSON = _PROMPT_DIR / "repair_retrieval_associations.json"

_MAP_EVIDENCE_FIELDS = (
    "relevant_landmarks",
    "current_intended_segment",
    "target_relation",
    "current_trajectory_relation",
    "observable_action_result",
)
_TASK_STATE_FIELDS = (
    "visible_task_phase",
    "completed_route_state",
    "unresolved_task_state",
)


def _project_visible_turns(
    example: dict[str, Any], *, include_flag: str = "used_for_detection"
) -> tuple[dict[str, str], ...]:
    """Keep only utterances available at the corresponding runtime stage."""
    projected = []
    for turn in example.get("visible_conversation") or ():
        if not isinstance(turn, dict) or turn.get(include_flag) is not True:
            continue
        phase = turn.get("phase")
        role = str(turn.get("role", "")).lower()
        utterance = str(turn.get("utterance", "")).strip()
        # A Guide detector runs before producing its current response. Therefore
        # all future turns and Guide-authored focal repairs are unavailable.
        if phase == "after" or (phase == "focal" and role == "guide"):
            continue
        if phase not in {"before", "focal"} or role not in {"guide", "follower"} or not utterance:
            continue
        projected.append({"phase": phase, "role": role, "utterance": utterance})
    return tuple(projected)


def _project_map_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    projected = {key: value[key] for key in _MAP_EVIDENCE_FIELDS if value.get(key) not in (None, "", [])}
    return projected or None


def _project_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    projected = {key: value[key] for key in fields if value.get(key) not in (None, "", [])}
    return projected or None


@lru_cache(maxsize=1)
def runtime_h_examples() -> tuple[dict[str, Any], ...]:
    """Project the reviewed registry into the minimal H-detector view.

    Audit metadata remains in ``h_fewshots.json`` but never crosses this
    runtime boundary. An example may be absent for any H category.
    """
    if not _H_FEWSHOTS_JSON.is_file():
        return ()
    raw_examples = json.loads(_H_FEWSHOTS_JSON.read_text(encoding="utf-8"))
    if not isinstance(raw_examples, list):
        raise ValueError("h_fewshots.json must contain a JSON array")

    projected_examples = []
    for raw in raw_examples:
        if not isinstance(raw, dict):
            continue
        h_code = str(raw.get("h_code", ""))
        expected = raw.get("expected_detector_output")
        if not re.fullmatch(r"H[1-7]", h_code) or not isinstance(expected, dict):
            continue
        expected_h_code = str(expected.get("h_code", ""))
        if expected_h_code != h_code:
            raise ValueError(f"few-shot label mismatch for {raw.get('example_id', '<unknown>')}")
        projected_examples.append({
            "h_code": h_code,
            "visible_conversation": _project_visible_turns(raw),
            "guide_map_evidence": _project_map_evidence(raw.get("guide_map_evidence")),
            "follower_workspace_evidence": _project_map_evidence(
                raw.get("follower_workspace_evidence")
            ),
            "expected_detector_output": {
                "h_code": h_code,
                "confidence": str(expected.get("confidence", "")),
                "evidence": str(expected.get("evidence", "")),
            },
        })
    return tuple(projected_examples)


def format_h_fewshots() -> str:
    """Render only observable context plus the reviewed target H output."""
    rendered = ["<H FEW-SHOT EXAMPLES>"]
    for example in runtime_h_examples():
        rendered.extend(("<H FEW-SHOT EXAMPLE>", "Observable context:"))
        conversation = example["visible_conversation"]
        if conversation:
            rendered.append("<VISIBLE CONVERSATION>")
            rendered.extend(
                f"[{turn['phase']}][{turn['role'].title()}] {turn['utterance']}"
                for turn in conversation
            )
            rendered.append("</VISIBLE CONVERSATION>")
        for tag, field in (
            ("GUIDE MAP CONTEXT", "guide_map_evidence"),
            ("FOLLOWER WORKSPACE CONTEXT", "follower_workspace_evidence"),
        ):
            value = example[field]
            if value:
                rendered.extend((
                    f"<{tag}>",
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    f"</{tag}>",
                ))
        rendered.extend((
            f"This observable context is labeled {example['h_code']}.",
            "Expected output:",
            json.dumps(example["expected_detector_output"], ensure_ascii=False, sort_keys=True),
            "</H FEW-SHOT EXAMPLE>",
        ))
    rendered.append("</H FEW-SHOT EXAMPLES>")
    return "\n".join(rendered)


@lru_cache(maxsize=1)
def runtime_repair_examples() -> tuple[dict[str, Any], ...]:
    """Return safe, single-R repair examples, deduplicated within an episode."""
    if not _REPAIR_FEWSHOTS_JSON.is_file():
        return ()
    raw_examples = json.loads(_REPAIR_FEWSHOTS_JSON.read_text(encoding="utf-8"))
    if not isinstance(raw_examples, list):
        raise ValueError("repair_fewshots.json must contain a JSON array")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_examples:
        if not isinstance(raw, dict) or raw.get("eligible") is not True:
            continue
        selection = raw.get("repair_selection_context")
        expected = raw.get("expected_repair_output")
        if not isinstance(selection, dict) or not isinstance(expected, dict):
            continue
        r_code = str(selection.get("primary_r_code", ""))
        if not re.fullmatch(r"R[1-7]", r_code) or expected.get("r_code") != r_code:
            continue
        source_episode_id = str(raw.get("source_episode_id", "")).strip()
        example_id = str(raw.get("example_id", "")).strip()
        if not source_episode_id or not example_id:
            continue
        quality = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
        candidate = {
            "example_id": example_id,
            "source_episode_id": source_episode_id,
            "r_code": r_code,
            "h_code": str(selection.get("h_code", "")),
            "w_codes": tuple(code for code in selection.get("w_codes", ()) if isinstance(code, str)),
            "visible_conversation": _project_visible_turns(raw, include_flag="used_for_selection"),
            "guide_map_evidence": _project_map_evidence(raw.get("guide_map_evidence")),
            "follower_workspace_evidence": _project_map_evidence(raw.get("follower_workspace_evidence")),
            "current_task_state_evidence": _project_fields(
                raw.get("current_task_state_evidence"), _TASK_STATE_FIELDS
            ),
            "repair_structure": str((selection.get("label_evidence") or {}).get("r", "")).strip(),
            "reply": str(expected.get("reply", "")).strip(),
            "retrieval_text": str(raw.get("retrieval_text", "")).strip(),
            "quality": float(quality.get("overall", 0) or 0),
        }
        if not candidate["reply"]:
            continue
        key = (source_episode_id, r_code)
        if key not in selected or candidate["quality"] > selected[key]["quality"]:
            selected[key] = candidate
    return tuple(sorted(selected.values(), key=lambda item: item["example_id"]))


def _token_overlap_score(current_evidence: str, example: dict[str, Any]) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", current_evidence.lower()))
    if not tokens:
        return 0.0
    reference = set(re.findall(r"[a-z0-9]+", example["retrieval_text"].lower()))
    return len(tokens & reference) / len(tokens | reference) + 0.05 * example["quality"]


def retrieve_repair_examples(
    current_evidence: str, associated_r_codes: tuple[str, ...], *, max_examples: int = 5
) -> tuple[dict[str, Any], ...]:
    """Cover H/W-associated R examples, then fill by text similarity."""
    examples = runtime_repair_examples()
    scored = sorted(
        ((-_token_overlap_score(current_evidence, example), example) for example in examples),
        key=lambda item: (item[0], item[1]["example_id"]),
    )
    chosen: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r_code in associated_r_codes:
        match = next((example for _, example in scored if example["r_code"] == r_code), None)
        if match:
            chosen.append(match)
            seen_ids.add(match["example_id"])
    for _, example in scored:
        if len(chosen) >= max_examples:
            break
        if example["example_id"] not in seen_ids:
            chosen.append(example)
            seen_ids.add(example["example_id"])
    return tuple(chosen[:max_examples])


def format_repair_examples(examples: tuple[dict[str, Any], ...]) -> str:
    """Render safe context and repair structure, never source-session audit data."""
    rendered = ["<RETRIEVED REPAIR EXAMPLES>"]
    for example in examples:
        rendered.append(f'<RETRIEVED REPAIR EXAMPLE id="{example["example_id"]}">')
        conversation = example["visible_conversation"]
        if conversation:
            rendered.append("<VISIBLE CONVERSATION>")
            rendered.extend(
                f"[{turn['phase']}][{turn['role'].title()}] {turn['utterance']}"
                for turn in conversation
            )
            rendered.append("</VISIBLE CONVERSATION>")
        for tag, field in (
            ("GUIDE MAP CONTEXT", "guide_map_evidence"),
            ("FOLLOWER WORKSPACE CONTEXT", "follower_workspace_evidence"),
            ("CURRENT TASK STATE", "current_task_state_evidence"),
        ):
            if example[field]:
                rendered.extend((f"<{tag}>", json.dumps(example[field], ensure_ascii=False, sort_keys=True), f"</{tag}>"))
        rendered.extend((
            "<REPAIR CONTEXT>",
            f"H={example['h_code']}; W={list(example['w_codes'])}; primary R={example['r_code']}",
            "</REPAIR CONTEXT>",
            "<REPAIR STRUCTURE>", example["repair_structure"], "</REPAIR STRUCTURE>",
            "<HISTORICAL REPLY EXCERPT>", example["reply"], "</HISTORICAL REPLY EXCERPT>",
            "</RETRIEVED REPAIR EXAMPLE>",
        ))
    rendered.append("</RETRIEVED REPAIR EXAMPLES>")
    return "\n".join(rendered)


@lru_cache(maxsize=1)
def repair_association_counts() -> dict[str, dict[tuple[str, ...], int]]:
    """Load offline-derived H/W-to-R supports used only for retrieval ranking.

    The data file is generated by ``grounding_analysis`` from the original
    annotations.  Reviewed few-shots remain the retrieval pool and are never
    used as the statistical sample.  Counts do not enter any LLM prompt.
    """
    if not _REPAIR_ASSOCIATIONS_JSON.is_file():
        raise FileNotFoundError(
            "Missing static repair association export: "
            f"{_REPAIR_ASSOCIATIONS_JSON}"
        )
    raw = json.loads(_REPAIR_ASSOCIATIONS_JSON.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("repair_retrieval_associations.json has an unsupported schema")

    fields_by_family = {
        "H+W": ("h_code", "w_code", "r_code"),
        "H": ("h_code", "r_code"),
        "W": ("w_code", "r_code"),
    }
    result: dict[str, dict[tuple[str, ...], int]] = {}
    for family, fields in fields_by_family.items():
        rows = raw.get(family)
        if not isinstance(rows, list):
            raise ValueError(f"repair association export is missing {family} rows")
        counts: dict[tuple[str, ...], int] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"repair association {family} row must be an object")
            key = tuple(str(row.get(field, "")) for field in fields)
            valid_codes = all(
                re.fullmatch(rf"{code[0]}[1-9][0-9]*", code) for code in key
            )
            support = row.get("support")
            if not valid_codes or not isinstance(support, int) or support < 1:
                raise ValueError(f"invalid repair association row in {family}: {row!r}")
            if key in counts:
                raise ValueError(f"duplicate repair association row in {family}: {key!r}")
            counts[key] = support
        result[family] = counts
    return result
