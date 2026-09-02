import sys
sys.path.insert(0, ".")
import scripts.evaluate_ablation as m

q = "Is dementia a normal part of aging?"
for name, fn in m.RETRIEVERS.items():
    chunks = fn(q)
    print(name, "->", len(chunks), "chunks; sources:", [c.get("source_document") for c in chunks])
    print("  chunk_ids:", [c.get("chunk_id") for c in chunks])

# Test strip_reliability_section on a real production prompt
import json
from scripts.rag_chat import prepare_evidence
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision
from scripts.build_grounded_prompt import build_grounded_prompt

chunks = m.RETRIEVERS["hybrid_full"](q)
ev = prepare_evidence(chunks)
rel = evaluate_reliability(query=q, evidence_items=ev)
dec = make_reliability_decision(rel)
prompt = build_grounded_prompt(query=q, evidence_items=ev, reliability=rel, decision=dec,
                               user_profile=None, conversation_context="",
                               response_language="en", assistance_plan=None)
stripped = m.strip_reliability_section(prompt)
print("prompt len:", len(prompt), "stripped len:", len(stripped))
print("RELIABILITY INFORMATION in stripped:", "RELIABILITY INFORMATION" in stripped)
print("SOURCE INFORMATION in stripped:", "SOURCE INFORMATION" in stripped)
print("Decision in stripped:", "Decision:" in stripped)
