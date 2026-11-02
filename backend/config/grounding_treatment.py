"""One server-owned switch for the Map Task grounding treatment.

This controls the H/W/R grounding pipeline. The public function returns only a
read-only capability flag for clients.
"""
from __future__ import annotations


ENABLE_GROUNDING_TREATMENT = True
MAPTASK_AGENT_ROLE = "guide"
# Map Task Guide without the grounding treatment: also send the two map images
# (Guide map, Follower map + drawing) with the normal reply.
PLAIN_GUIDE_MAP_IMAGES = True
# Defaults when a session does not set them (the session's own value still wins).
GUIDE_SEES_FOLLOWER_DRAWING = False  # C2: the Guide (human or agent) sees the Follower's drawing (False: C1)
# For the agent in MAPTASK_AGENT_ROLE. True: grounding agent; False: plain agent;
# None: Awareness Dashboard rule (Guide only; a Follower agent then stays plain).
GROUNDING_TREATMENT = True



def grounding_treatment_enabled() -> bool:
    return bool(ENABLE_GROUNDING_TREATMENT)


def maptask_agent_role() -> str:
    """Return the only supported server-owned AI role for a Map Task session."""
    role = str(MAPTASK_AGENT_ROLE).strip().lower()
    return role if role in {"guide", "follower"} else "guide"


def public_feature_flags() -> dict[str, bool]:
    return {"grounding_treatment_enabled": grounding_treatment_enabled()}
