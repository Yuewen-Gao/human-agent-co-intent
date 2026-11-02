"""Grounding-act vocabulary and the act block handed to the reply stage.

The act block is the grounding Guide's own reply instruction; the plain Guide
never sees it. It says what each selected act must achieve in the next message
and, when the repair stage writes the message, how each repair strategy (R) is
carried out. The R codebook only defines the strategies (it is also the
annotation scheme); what a repair may fix belongs to the repair prompt; task
conventions (landmarks, no grid numbers) belong to the shared task prompt.
Definitions paraphrase Traum's grounding acts in task-general wording. The
act-to-R mapping is a soft hint, not a rule.
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
    "Acknowledge": "Show or state that you understood the partner's previous content or result, e.g., a positive reply, a restatement in your own terms, or approval of what they did.",
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
# What each act must achieve in this message (not how to word it).
REPLY_TODO = {
    "Initiate": "Give the next new step or content the partner has not yet received, without repeating or correcting earlier content.",
    "Continue": "Add the next part of the step still being worked on, without starting a new step or correcting anything.",
    "Acknowledge": "Acknowledge what the partner just said or did, claiming no more than your evidence shows, without adding a new instruction or correction unless another selected act asks for it.",
    "Repair": "Fix the misunderstanding described by the target item.",
    "Request Repair": "Get the partner to clarify the specific part of their content you could not interpret.",
    "Request Acknowledge": "Get the partner to show whether your pending content or result matches what they meant, or how they understood it. When the partner cannot observe your result, the message has to make it available for them to check. Do not introduce a new correction.",
    "Cancel": "Withdraw the earlier content so the partner stops acting on it; give any replacement as the next act.",
}
R_HINT = {
    "Repair": "R4, R5, R6",
    "Request Repair": "R1, R2, R3",
    "Request Acknowledge": "R1, or R7 when a repair for the item was already sent, to check the partner's understanding of your content; R3 to put your own result or interpretation to the partner",
    "Cancel": "R4",
}
# How a reply carries out each repair strategy (definitions are in the R codebook).
R_REALIZATION = {
    "R1": "Ask an open question that lets the partner say, in their own terms, what they understood or where their "
          "result is. Do not offer candidates, and do not supply the reference the answer should use.",
    "R2": "Ask about the one missing or ambiguous component only, in terms the partner can check on their own side: "
          "a reference both have confirmed, or the partner's own words.",
    "R3": "Offer one candidate built from established references and ask the partner to confirm or correct it; a bare "
          "\"yes\" to it is weak evidence.",
    "R4": "Replace the problematic expression with one built from references established as shared, or from the "
          "partner's own terms.",
    "R5": "State the candidate as provisional and invite the partner to accept, reject, or revise it.",
    "R6": "State the target relative to a named reference and the relation to it (for example, which side of it), "
          "not only \"near\" it; prefer references established as shared.",
    "R7": "Ask the partner to show, in their own terms, how they carried out the repair; do not restate the repair "
          "as a yes/no question.",
}
EVIDENCE_NOTE = (
    "Evidence of understanding, with the least collaborative effort: when the partner's understanding of something "
    "still matters, let this message bring the evidence where it can (for example, through the step itself), and do "
    "not ask again for what the partner has just told you."
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
    lines = ["<SELECTED GROUNDING ACT>", "Sending a message has already been decided: send one message that performs the selected act(s)."]
    for act, target in zip(acts, targets):
        lines.append(f"Your reply must perform: {act}" + (f" (target item: {target})" if target else ""))
        lines.append(f"Meaning: {DEFINITIONS[act]}")
        lines.append(f"What the reply should achieve: {REPLY_TODO[act]}")
        if for_repair and act in R_HINT:
            lines.append(f"Compatible repair strategies (guidance, not a rule): {R_HINT[act]}.")
    if len(acts) > 1:
        lines.append("Perform the selected acts in the order listed.")
    if target_item and target_item.get("repair_attempts"):
        attempts = target_item["repair_attempts"]
        lines.append(f'Item history: repairs sent: {len(attempts)} (what was said last; check it against the map before reusing it: "{attempts[-1]["reply"]}").')
    if item_status:
        lines.append(f"Item status from this turn's decision: {item_status.get('outcome')}, evidence "
                     f"{item_status.get('strength') or '?'}: {item_status.get('evidence', '')}".rstrip())
    lines.append(EVIDENCE_NOTE)
    if for_repair:
        lines.append("How to carry out each repair strategy (R):")
        lines += [f"- {code}: {text}" for code, text in R_REALIZATION.items()]
    lines.append(f"Reason (from act selection): {reason}")
    lines.append("Do not mention the act name, codes, or this block to the participant.")
    lines.append("</SELECTED GROUNDING ACT>")
    return "\n".join(lines)
