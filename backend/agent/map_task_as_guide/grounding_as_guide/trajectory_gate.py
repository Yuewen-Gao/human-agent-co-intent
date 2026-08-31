"""Deterministic de-duplication keys for drawing-triggered Guide replies."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


_ACTIONABLE_VERDICTS = frozenset({"correct", "incorrect"})


def advice_signature(h: Mapping[str, Any]) -> str | None:
    """Return a stable key for the immediate advice warranted by an H result.

    The observed trace is intentionally excluded. A trace may grow while the
    correction or the next target segment remains the same, and should not
    trigger a duplicate Guide message in that case.
    """
    verdict = h.get("trajectory_verdict")
    action = h.get("next_guide_action")
    mismatch = h.get("misalignment")
    expected = mismatch.get("expected_state") if isinstance(mismatch, Mapping) else None
    if (
        not isinstance(verdict, str)
        or verdict not in _ACTIONABLE_VERDICTS
        or not isinstance(action, str)
        or not isinstance(expected, str)
        or not expected.strip()
    ):
        return None
    payload = json.dumps(
        {"verdict": verdict, "action": action, "expected_state": expected.strip().casefold()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def should_suppress_trajectory_reply(
    previous_signature: str | None, h: Mapping[str, Any]
) -> bool:
    """True when the latest drawing still warrants exactly the same advice."""
    current_signature = advice_signature(h)
    return bool(current_signature and current_signature == previous_signature)
