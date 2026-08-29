"""Validation for public, inspectable LLM stage outputs."""
from __future__ import annotations
import json
from typing import Any

def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value=json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}

def parse_h(raw: str) -> dict[str, str]:
    value=parse_json_object(raw); code=value.get("h_code")
    return {"h_code":code,"confidence":str(value.get("confidence", "low")),"evidence":str(value.get("evidence", ""))} if isinstance(code,str) and code in {f"H{i}" for i in range(1,8)} else {}

def parse_repair(raw: str, allowed: tuple[str,...]) -> dict[str, str]:
    value=parse_json_object(raw); code=value.get("r_code")
    if not isinstance(code,str) or code not in allowed: return {}
    field=value.get("uw_code")
    return {"r_code":code,"reply":str(value.get("reply", "")),"uw_code":field if isinstance(field,str) and field in {f"W{i}" for i in range(1,13)} else "","evidence_is_explicit":str(value.get("evidence_is_explicit", "false")).lower() == "true","adds_missing_detail":str(value.get("adds_missing_detail", "false")).lower()=="true"}
