# Map Task Grounding Pipeline

`grounding_analysis/` is an offline-only project. It produces candidate examples, but the running backend never imports or executes it.

The online backend reads only the reviewed static snapshot in `agent/prompts/grounding_fewshots.csv`. It makes two LLM calls: first it classifies `H`; second it selects `R`, drafts the reply, and proposes `UW`. `grounding_rules.py` then deterministically maps `R` to `M` and rejects a proposed `UW` without explicit evidence.

The pipeline records only inspectable evidence and structured labels. It does not request hidden reasoning. If H is absent, R is disallowed, or UW evidence is weak, it returns no update rather than guessing.
