"""
ElderDocAI RQ2 Evaluation - 16 gold QA questions.

Evaluates retrieval grounding, reliability, faithfulness, and answer
relevance using ONLY the in-scope `gold_questions` subset of
data/gold_qa_evaluation.json (out-of-scope items are excluded).

Import discipline: we import ONLY `scripts.rag_chat` (which imports
`scripts.hybrid_retriever` once). Avoids the double model-load that hung
`evaluate_faithfulness.py`.
"""

import os
import json
import time
import re

import ollama

from scripts.rag_chat import generate_answer, LLM_MODEL
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_QA_FILE = os.path.join(BASE_DIR, "data", "gold_qa_evaluation.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "evaluation_results", "gold_qa_results.json")
JUDGE_MODEL = "llama3.2:latest"
FAITHFULNESS_PROMPT = """You are a strict RAG faithfulness evaluator.

Your ONLY job:
Check whether the generated answer contains
information that is NOT supported by the retrieved evidence.

IMPORTANT:
Do NOT punish the answer because:
- it is short
- it does not include every evidence detail
- it summarizes evidence
- it omits examples
- it does not use all retrieved documents

ONLY punish:
- invented facts
- unsupported medical claims
- information not present in evidence
- contradictions with evidence

====================================================
QUESTION
====================================================

{question}


====================================================
RETRIEVED EVIDENCE
====================================================

{evidence}


====================================================
GENERATED ANSWER
====================================================

{answer}


====================================================
SCORING
====================================================

1.0

All statements are supported by evidence.
No hallucination.


0.75

Almost completely supported.
Only tiny unsupported wording.


0.50

Some important statements are unsupported.


0.25

Many unsupported statements.


0.0

Answer is unrelated or contradicts evidence.


Return ONLY JSON:

{{
    "score": 1.0,
    "reason": "short explanation"
}}
"""

RELEVANCE_PROMPT = """You are a strict RAG answer relevance evaluator.

Your ONLY task is to evaluate whether the generated answer
directly answers the user's question.

Do NOT evaluate:
- factual correctness
- faithfulness
- missing details
- writing quality
- answer length

====================================================
QUESTION
====================================================

{question}


====================================================
GENERATED ANSWER
====================================================

{answer}


====================================================
SCORING RULES
====================================================

1.0

The answer directly and completely addresses the question.


0.75

The answer answers the main question but misses
minor aspects.


0.50

The answer is partially related but does not fully
answer the question.


0.25

The answer is mostly unrelated.


0.0

The answer does not address the question at all.


IMPORTANT:

A short answer can receive 1.0 if it correctly answers
the question.

Do NOT reduce the score because:
- the answer is concise
- examples are missing
- additional evidence was not included


Return ONLY JSON.

Format:

{{
    "score": 1.0,
    "reason": "short explanation"
}}
"""
def normalize_filename(filename):
    """Normalize a source filename for robust comparison."""
    filename = str(filename).lower().replace(".pdf", "")
    return "".join(ch for ch in filename if ch.isalnum())


SOFT_HYPHEN = "\u00ad"
ZERO_WIDTH = "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"


def normalize_for_span(text):
    """Robust text normalization for supporting-span matching.

    - Merges characters broken by a Unicode soft hyphen (U+00AD), including
      adjacent whitespace: 'demen\\u00ad tia' -> 'dementia'.
    - Removes other zero-width / format control chars.
    - Joins PDF line-break hyphenation artifacts ('word-\\nword').
    - Collapses all remaining non-alphanumerics to a single space.
    """
    if not text:
        return ""
    t = str(text).casefold()
    # Soft hyphen + any adjacent whitespace: the pieces belong to one word.
    t = re.sub(r"\s*" + SOFT_HYPHEN + r"\s*", "", t)
    # Remaining zero-width / format control chars.
    t = re.sub("[" + ZERO_WIDTH + "]", "", t)
    # PDF line-break hyphenation: hyphen at end-of-line joins the word.
    t = re.sub(r"-\s*\n\s*", "", t)
    # Collapse all other punctuation / spacing to a single ASCII space.
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


SPAN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "has", "have", "how", "i", "in", "into", "is",
    "it", "its", "may", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "your",
}


def span_token_coverage(span, evidence_text):
    """Fraction of gold-span content tokens present in the evidence text.

    Robust to light paraphrase, chunk boundaries, and PDF hyphenation.
    """
    span_tokens = [
        tok for tok in normalize_for_span(span).split()
        if tok not in SPAN_STOPWORDS
    ]
    evidence_tokens = set(normalize_for_span(evidence_text).split())
    if not span_tokens:
        return 1.0 if span else 0.0
    hits = sum(1 for tok in span_tokens if tok in evidence_tokens)
    return round(hits / len(span_tokens), 4)


def build_evidence_text(evidence_items):
    """Render retrieved evidence into a text block for the LLM judge."""
    lines = []
    for i, item in enumerate(evidence_items, start=1):
        lines.append(
            "==============================\n"
            f"Evidence {i}\n"
            "==============================\n"
            f"Source: {item.get('source_document')}\n\n"
            f"{item.get('text', '')}"
        )
    return "\n".join(lines)


def judge(question, answer, evidence, prompt_template, system_content):
    """Call the LLM judge with ollama JSON format."""
    text = build_evidence_text(evidence) if evidence else "(no evidence retrieved)"
    prompt = prompt_template.format(question=question, answer=answer, evidence=text)
    response = ollama.chat(
        model=JUDGE_MODEL,
        format="json",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        parsed = json.loads(response["message"]["content"])
        return float(parsed["score"]), str(parsed.get("reason", ""))
    except Exception as exc:  # noqa: BLE001
        return None, f"judge-parse-error: {exc}"


def evaluate_question(q):
    """Run the full evaluation stack for a single gold question."""
    record = {
        "id": q.get("id"),
        "question": q.get("question"),
        "topic": q.get("topic"),
        "category": q.get("category"),
        "expected_source_document": q.get("source_document"),
        "expected_chunk_ids": q.get("chunk_ids", []),
        "target_keys": q.get("target_keys", []),
        "supporting_span": q.get("supporting_span"),
        "reference_answer": q.get("reference_answer"),
        "retrieved_source_documents": [],
        "retrieved_chunk_ids": [],
        "retrieved_evidence_texts": [],
        "generated_answer": None,
        "reliability": None,
        "adaptive_decision": None,
        "answer_relevance": None,
        "faithfulness": None,
        "grounding": None,
        "timings": {},
        "error": None,
    }

    t_start = time.perf_counter()
    try:
        answer, evidence_items = generate_answer(q.get("question"), return_evidence=True)
        record["generated_answer"] = answer
        record["timings"]["generation_seconds"] = round(time.perf_counter() - t_start, 3)

        retrieved_src = [e.get("source_document") for e in evidence_items]
        retrieved_cid = [e.get("chunk_id") for e in evidence_items]
        record["retrieved_source_documents"] = retrieved_src
        record["retrieved_chunk_ids"] = retrieved_cid
        record["retrieved_evidence_texts"] = [e.get("text", "") for e in evidence_items]

        # ---- Grounding metrics (against actual retrieved evidence) ----
        norm_expected = normalize_filename(q.get("source_document", ""))
        norm_retrieved = {normalize_filename(s) for s in retrieved_src}
        source_retrieved = bool(norm_retrieved) and norm_expected in norm_retrieved

        expected_cid = {int(c) for c in (q.get("chunk_ids") or [])}
        retrieved_cid_set = {int(c) for c in retrieved_cid if c is not None}
        chunk_recall = bool(expected_cid & retrieved_cid_set)

        span = q.get("supporting_span") or ""
        evidence_joined = " ".join((e.get("text") or "") for e in evidence_items)
        span_norm_needle = normalize_for_span(span)
        span_exact = bool(span) and span_norm_needle in normalize_for_span(evidence_joined)
        span_coverage = span_token_coverage(span, evidence_joined) if span else 0.0
        # Supporting span is considered supported when the gold span appears
        # verbatim (robust-normalized) OR at least 85% of its content tokens
        # are covered by the retrieved evidence.
        span_supported = bool(span) and (span_exact or span_coverage >= 0.85)

        record["grounding"] = {
            "source_retrieval_correct": source_retrieved,
            "expected_source_normalized": norm_expected,
            "retrieved_sources_normalized": sorted(norm_retrieved),
            "chunk_recall": chunk_recall,
            "expected_chunk_ids": sorted(expected_cid),
            "retrieved_chunk_ids": sorted(retrieved_cid_set),
            "supporting_span_supported": span_supported,
            "supporting_span_exact": span_exact,
            "supporting_span_coverage": span_coverage,
            "target_keys": q.get("target_keys", []),
        }

        # ---- Reliability (existing config + thresholds) ----
        reliability = evaluate_reliability(
            query=q.get("question"),
            evidence_items=evidence_items,
        )
        decision = make_reliability_decision(reliability)
        record["reliability"] = {k: round(float(v), 4) for k, v in reliability.items()}
        record["adaptive_decision"] = decision.get("decision")

        # ---- LLM judges ----
        t_j = time.perf_counter()
        relevance_score, relevance_reason = judge(
            q.get("question"), answer, evidence_items,
            RELEVANCE_PROMPT, "You evaluate RAG answer relevance only.",
        )
        fidelity_score, fidelity_reason = judge(
            q.get("question"), answer, evidence_items,
            FAITHFULNESS_PROMPT, "You evaluate RAG faithfulness only.",
        )
        record["timings"]["judge_seconds"] = round(time.perf_counter() - t_j, 3)
        record["answer_relevance"] = {
            "score": relevance_score, "reason": relevance_reason,
        }
        record["faithfulness"] = {
            "score": fidelity_score, "reason": fidelity_reason,
        }
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["timings"]["generation_seconds"] = round(time.perf_counter() - t_start, 3)

    return record
def main():
    with open(GOLD_QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_questions = data.get("gold_questions", [])
    print("=" * 70)
    print(f"ElderDocAI RQ2 Gold-QA Evaluation ({len(gold_questions)} questions)")
    print("=" * 70)

    results = []
    t_all = time.perf_counter()
    for i, q in enumerate(gold_questions, start=1):
        print(f"\n[{i}/{len(gold_questions)}] {q.get('id')} :: {q.get('question')}")
        rec = evaluate_question(q)
        results.append(rec)
        if rec.get("error"):
            print(f"  ERROR: {rec['error']}")
        else:
            g = rec["grounding"]
            print(f"  source_ok={g['source_retrieval_correct']} chunk_recall={g['chunk_recall']} span_supported={g['supporting_span_supported']}")
            print(f"  reliability={rec['reliability'].get('overall_reliability') if rec['reliability'] else None} decision={rec['adaptive_decision']}")
            print(f"  relevance={rec['answer_relevance'].get('score') if rec['answer_relevance'] else None} faithfulness={rec['faithfulness'].get('score') if rec['faithfulness'] else None}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nWrote JSON results to {OUTPUT_FILE}")

    # ---- Summary ----
    total = len(results)
    gen_ok = sum(1 for r in results if r.get("generated_answer"))
    src_ok = sum(1 for r in results if r.get("grounding") and r["grounding"]["source_retrieval_correct"])
    chunk_ok = sum(1 for r in results if r.get("grounding") and r["grounding"]["chunk_recall"])
    span_ok = sum(1 for r in results if r.get("grounding") and r["grounding"]["supporting_span_supported"])

    rel_scores = [r["reliability"]["overall_reliability"] for r in results if r.get("reliability")]
    rel_avg = float(sum(rel_scores) / len(rel_scores)) if rel_scores else None

    dec_counts = {}
    for r in results:
        d = r.get("adaptive_decision")
        dec_counts[d] = dec_counts.get(d, 0) + 1
    rel_accept = dec_counts.get("ACCEPT", 0)

    def avg_key(key, field="score"):
        vals = []
        for r in results:
            v = r.get(key)
            if v and v.get(field) is not None:
                vals.append(float(v[field]))
        return round(sum(vals) / len(vals), 4) if vals else None

    avg_relevance = avg_key("answer_relevance")
    avg_faith = avg_key("faithfulness")

    print("\n" + "=" * 70)
    print("FINAL RQ2 SUMMARY")
    print("=" * 70)
    print(f"Total questions            : {total}")
    print(f"Successful generations     : {gen_ok}/{total}")
    print(f"Source retrieval accuracy  : {src_ok}/{total}")
    print(f"Evidence/chunk recall      : {chunk_ok}/{total}")
    print(f"Supporting-span support    : {span_ok}/{total}")
    print(f"Average reliability        : {rel_avg:.4f}" if rel_avg is not None else "Average reliability        : n/a")
    print(f"Accept decisions           : {rel_accept}/{total}")
    print(f"Average relevance          : {avg_relevance}" if avg_relevance is not None else "Average relevance          : n/a")
    print(f"Average faithfulness       : {avg_faith}" if avg_faith is not None else "Average faithfulness       : n/a")
    print(f"Decision distribution      : {dec_counts}")
    print(f"Total wall time            : {round(time.perf_counter() - t_all, 1)}s")

    print("\nPer-question results:")
    for r in results:
        rel = r.get("reliability", {})
        g = r.get("grounding", {})
        print(
            f"  {r.get('id')}: gen_ok={bool(r.get('generated_answer'))} "
            f"src={g.get('source_retrieval_correct')} chunk={g.get('chunk_recall')} "
            f"span={g.get('supporting_span_supported')} "
            f"rel={rel.get('overall_reliability')} dec={r.get('adaptive_decision')} "
            f"faith={r.get('faithfulness', {}).get('score')} "
            f"relv={r.get('answer_relevance', {}).get('score')} "
            + (f"ERR={r.get('error')}" if r.get('error') else "")
        )


if __name__ == "__main__":
    main()
