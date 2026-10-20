"""Grounding-act vocabulary and the act block handed to the reply stage.

Definitions paraphrase Traum's grounding acts in task-general wording. The
act-to-R mapping is a soft hint for the repair stage, not a rule.
"""
from __future__ import annotations

from typing import Any

ACTS = ("Initiate", "Continue", "Acknowledge", "Repair", "Request Repair", "Request Acknowledge", "Cancel")
# Acts realized by the repair (R) stage; the others go to the normal Guide reply.
REPAIR_STAGE_ACTS = frozenset({"Repair", "Request Repair", "Request Acknowledge", "Cancel"})

PENDING_CONTENT = (
    "Pending content: content that one party has presented (an instruction, proposal, description, question, "
    "or task result) but that has not yet been shown to be understood by the other party."
)
DEFINITIONS = {
    "Initiate": "Present new content that starts a new step, topic, or proposal, rather than adding to or fixing content that is still pending.",
    "Continue": "Add more of your own still-pending content: the next part of the same instruction or explanation, without changing topic and without correcting anything.",
    "Acknowledge": "Show or state that you understood the partner's previous content or result, e.g., a positive reply, an accurate restatement, or approval of what they did.",
    "Repair": "Correct an actual or possible misunderstanding of content already presented by either party, by replacing it, revising it, or adding the detail needed to fix it.",
    "Request Repair": "Signal that you did not understand the partner's content, and ask them to clarify, repeat, or specify it.",
    "Request Acknowledge": "Ask the partner to show whether they understood or accepted your content or your result.",
    "Cancel": "Explicitly withdraw content that was still pending, so that it will no longer be established.",
}
READING_NOTES = (
    "Classify an act by what the utterance does in the current context, not by its surface form. One message "
    "may perform several acts, e.g., Acknowledge then Initiate. Request Repair means you did not understand the "
    "partner; Request Acknowledge asks the partner to show that they understood you."
)
REPLY_TODO = {
    "Initiate": "Present the next new step or new content the partner has not yet received. Do not repeat or correct earlier content.",
    "Continue": "Add the next part of your own instruction or explanation that is still being worked on, staying on the same step. Do not start a new step and do not correct anything.",
    "Acknowledge": "Briefly confirm that what the partner just said or did matches your understanding. Do not add a new instruction or a correction unless another selected act asks for it.",
    "Repair": "Correct the misunderstanding described by the target item: replace, revise, or add the missing detail.",
    "Request Repair": "Say what you did not understand in the partner's content and ask them to clarify.",
    "Request Acknowledge": "Ask for a confirmation the partner can answer by replying or acting; do not introduce a new correction.",
    "Cancel": "Explicitly withdraw the earlier content so the partner stops acting on it; give any replacement as the next act.",
}
R_HINT = {
    "Repair": "R4, R5, R6",
    "Request Repair": "R1, R2, R3",
    "Request Acknowledge": "R2, R3, R5, R7",
    "Cancel": "usually together with R4",
}
NO_GRID = (
    "The partner cannot see any grid, coordinates, or cell counts. Never describe distances as numbers of spaces, "
    "cells, rows, or columns; describe positions only by relations to visible landmarks and to the partner's own drawing."
)


def definitions_text() -> str:
    lines = [PENDING_CONTENT, ""]
    lines += [f"- {act}: {DEFINITIONS[act]}" for act in ACTS]
    return "\n".join(lines + ["", READING_NOTES])


def r7_allowed(target_item: dict[str, Any] | None) -> bool:
    """R7 checks an already established repair, so it needs a prior repair for the item."""
    return bool(target_item and target_item.get("repair_attempts"))


def act_block(
    acts: list[str],
    targets: list[str],
    reason: str,
    target_item: dict[str, Any] | None,
    item_status: dict[str, Any] | None,
    *,
    for_repair: bool,
) -> str:
    lines = ["<SELECTED GROUNDING ACT>"]
    for act, target in zip(acts, targets):
        lines.append(f"Your reply must perform: {act}" + (f" (target item: {target})" if target else ""))
        lines.append(f"Meaning: {DEFINITIONS[act]}")
        lines.append(f"What the reply should do: {REPLY_TODO[act]}")
        if for_repair and act in R_HINT:
            lines.append(f"Compatible repair strategies (guidance, not a rule): {R_HINT[act]}.")
    if len(acts) > 1:
        lines.append("Perform the selected acts in the order listed.")
    if target_item and target_item.get("repair_attempts"):
        attempts = target_item["repair_attempts"]
        lines.append(f'Item history: repairs sent: {len(attempts)} (last: "{attempts[-1]["reply"]}").')
        if for_repair:
            lines.append("R7 is allowed because a repair for this item was already sent.")
    if item_status:
        lines.append(f"Item status from this turn's decision: {item_status.get('outcome')} "
                     f"{item_status.get('progress', '')}: {item_status.get('evidence', '')}".rstrip())
    lines.append(NO_GRID)
    lines.append(f"Reason (from act selection): {reason}")
    lines.append("Do not mention the act name, codes, or this block to the participant.")
    lines.append("</SELECTED GROUNDING ACT>")
    return "\n".join(lines)
