"""Two-call grounding orchestrator; caller supplies current observable evidence."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .contract import parse_h, parse_repair
from .fewshot_repository import examples_for
from .grounding_rules import may_update_field, repair_candidates, update_mode

_PROMPTS=Path(__file__).resolve().parents[2] / "prompts"

def _fewshot(rows: list[dict[str,str]]) -> str:
    return "\n".join(f"Evidence: {r.get('follower_evidence','')}\nResponse: {r.get('guide_response','')}\nLabels: {r.get('guide_labels','')}" for r in rows)

class GroundingOrchestrator:
    def __init__(self, llm_client: Any): self._llm=llm_client
    def decide(self, evidence: str) -> dict[str, Any]:
        h_prompt=(_PROMPTS/"detect_h_prompt.txt").read_text(encoding="utf-8")+"\nFew-shot:\n"+_fewshot(examples_for("h"))+"\nEvidence:\n"+evidence
        h=parse_h(self._llm.chat_completions_create(messages=[{"role":"user","content":h_prompt}],temperature=0,max_tokens=400,response_format={"type":"json_object"}))
        allowed=repair_candidates(h.get("h_code"))
        if not allowed: return {"h":h,"r":None,"m":"M5","uw":None,"reply":""}
        r_prompt=(_PROMPTS/"select_repair_prompt.txt").read_text(encoding="utf-8")+f"\nDetected H: {h['h_code']}\nAllowed R: {', '.join(allowed)}\nFew-shot:\n"+_fewshot(examples_for("repair",h_code=h["h_code"]))+"\nEvidence:\n"+evidence
        repair=parse_repair(self._llm.chat_completions_create(messages=[{"role":"user","content":r_prompt}],temperature=0,max_tokens=600,response_format={"type":"json_object"}),allowed)
        if not repair: return {"h":h,"r":None,"m":"M5","uw":None,"reply":""}
        return {"h":h,"r":repair["r_code"],"m":update_mode(repair["r_code"],adds_missing_detail=repair["adds_missing_detail"]),"uw":may_update_field(evidence_is_explicit=repair["evidence_is_explicit"],proposed_field=repair["uw_code"]),"reply":repair["reply"]}
