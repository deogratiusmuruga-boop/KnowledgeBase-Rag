# ElderDocAI
### An Evidence–Grounded, Rule-Bounded Adaptive Framework for Elderly Care Assistance
**Manuscript scaffold / Methods–Results focus draft**
*Working title only — finalized with the target journal.*

---

## ABSTRACT (working)

Caregiver-facing AI systems must assist users without overclaiming medical
certainty. Large conversational models risk ungrounded diagnoses or risk
statements. We present **ElderDocAI**, a pipeline that (i) reduces longitudinal
clinical records into *documented care-activity signals* across deterministic
temporal windows, (ii) derives a *safe, rule-based adaptive assistance plan*
constrained by explicit no-diagnosis / no-risk-prediction guardrails, and
(iii) generates an *evidence-grounded response* from a hybrid-retrieval
knowledge base with reliability gating. The central contribution is the
separation of a fully validated deterministic personalization layer from a
grounded LLM response layer, preserving research scope: **the system describes
documented care activity and its temporal change but does not diagnose,
predict, or infer medical risk.**
---

## 1. MANUSCRIPT OUTLINE (Full paper)

1. **Introduction**
   - Background: burden on informal elderly caregivers; risk of ungrounded LLM advice.
   - Problem: harness longitudinal EMR-style data for adaptive assistance without overclaiming.
   - Contribution statement: validated deterministic core, safety guardrails, grounded RAG.
2. **Related Work**
   - Retrieval Augmented Generation for healthcare.
   - Adaptive/reactivity in care technology.
   - Safety frameworks and the "no diagnosis" boundary in computational care.
   - Gap: proving a deterministic safety-bounded layer AND a grounded LLM response together.
3. **Methods** (detailed in Section 2 below)
4. **Results** (detailed in Section 3 below)
5. **Discussion**
   - Meaning of results; why validation matters; significance of `NO_DATA != STABLE`; limitations.
6. **Conclusion**
   - Restate research contribution; future work (prospective data, generalization).
---

## 2. METHODS (DETAILED)

### 2.1 Architectural Overview

ElderDocAI is composed of **two domains**:

1. **Deterministic, validated core** — clinical features → dynamic care state →
continuous temporal windows → care-state transitions → adaptive assistance →
adaptive context → assistance decision → assistance plan.
2. **Grounded response layer** — hybrid retrieval, reliability gating, and
   evidence-grounded LLM generation.

| Layer | Records | Output |
|-------|---------|--------|
| Clinical Features | 178 patients | per-patient feature vectors |
| Dynamic Care State | 178 | aggregate care state |
| Care-State Window | 9,723 | one-year windows |
| Care-State Transitions | 9,723 | state deltas |
| Adaptive Assessment | 9,723 | assistance mode |
| Adaptive Context (junction) | 9,723 | merged per-window context |
| Assistance Decision | 9,723 | strategy + priority |
| Assistance Plan | 9,723 | concrete action set |

### 2.2 Data

- **Synthea-derived FHIR records** for 178 synthetic elderly patients.
- Longitudinal records split into disjoint one-year windows
  (`window_start`, `window_end`).
- **Join key:** `(patient_id, window_start, window_end)` — used across every
  layer (never array position), guaranteeing reproducible alignment.

### 2.3 Adaptive Context

- Each window merges patient profile, care state, transition, changed
  dimensions, adaptive assistance mode/priority, and a context status.
- Care states encode **documented activity intensity**
  (LOW / MODERATE / HIGH; `NO_DATA` when no activity is documented).

### 2.4 Deterministic Decision + Plan

- Strategy selection is **rule-based** with precedence:
  `INITIAL -> ONBOARDING_SUPPORT`
  `NO_DATA / GAP -> DATA_COLLECTION_SUPPORT`
  escalation -> `ENHANCED_CONTEXT_SUPPORT`
  de-escalation -> `ADAPTIVE_DEESCALATION_SUPPORT`
  increasing / decreasing / no-change -> MONITORING / FOLLOW_UP / tiered support.
- Each plan carries a **verbatim safety policy**:
  `NO_DIAGNOSIS`, `NO_MEDICAL_RISK_PREDICTION`,
  `NO_DISEASE_PROGRESSION_INFERENCE`, `DOCUMENTED_ACTIVITY_ONLY`,
  and (for gaps) `NO_DATA_IS_NOT_STABILITY`.

### 2.5 Grounded LLM Response

- **Retrieval:** hybrid dense + BM25 + cross-encoder reranking over a provider
  caregiver corpus.
- **Gate:** reliability evaluator with accept / refine / re-retrieve / reject policy.
- **Generation:** Llama 3.2 (Ollama) with temperature 0, forcing *only-from-SOURCE*
  answers; the validated assistance plan (strategy, actions, safety constraints)
  is injected for **response adaptation**, never as medical evidence.

---

## 3. RESULTS (DETAILED, FROM VALIDATED OUTPUTS)

### 3.1 Pipeline Validation

Every deterministic layer reports **0 errors / 0 warnings / PASS**:
upstream care-state pipeline, adaptive context, assistance decisions,
and assistance plans.

### 3.2 Layer Size & Coverage

| Artifact | Records |
|----------|---------|
| Adaptive Context | 9,723 |
| Assistance Decision | 9,723 |
| Assistance Plan (JSON + CSV) | 9,723 |
| Unique patients | 178 |
| Unique join keys | 9,723 (0 duplicates) |

### 3.3 Context-Status Breakdown

```
Initial      178 (1.8%)
DataGap    6,356 (65.4%)
Active     3,189 (32.8%)
```

### 3.4 Strategy -> Assistance Plan (validated)

| Strategy | Records |
|----------|---------|
| ONBOARDING_SUPPORT | 178 |
| DATA_COLLECTION_SUPPORT | 7,445 |
| CONTEXTUAL_SUPPORT | 476 |
| ENHANCED_SUPPORT | 452 |
| ENHANCED_CONTEXT_SUPPORT | 446 |
| ADAPTIVE_DEESCALATION_SUPPORT | 302 |
| LIGHT_SUPPORT | 202 |
| MONITORING_SUPPORT | 121 |
| FOLLOW_UP_SUPPORT | 101 |

### 3.5 Safety Verification

- `NO_DATA != STABLE` held across all 7,445 DATA_COLLECTION cases; the planner never
  labels a gap as stable, improved, or deteriorated.
- `NO_DATA_IS_NOT_STABILITY` is present on every gap-bound plan.
- The base safety policy is present on all 9,723 records.

### 3.6 Representative -> Action / Safety Cases

- **EScalation** -> `ENHANCED_CONTEXT_SUPPORT` -> targeted info, review, check-in;
  no diagnosis claim.
- **De-escalation** -> `ADAPTIVE_DEESCALATION_SUPPORT` -> reduce intervention intensity.
- **Increasing** -> `MONITORING_SUPPORT` -> acknowledge increased documented activity.
- **NO_DATA** -> `DATA_COLLECTION_SUPPORT` -> check-in / data update / avoid
  personalization; explicit `NOT STABLE`.
- **Initial** -> `ONBOARDING_SUPPORT` -> establish context; no medical inference.

### 3.7 Validator Confidence

Each layer (upstream care-state, adaptive-context, decisions, plans) provides an
automated PASS/FAIL validator — reusable as a reproducibility appendix / table.

---

## 4. NEXT STEPS (FOR FULL PAPER)

1. **Live end-to-end evaluation** — run `evaluate_faithfulness`,
   `evaluate_context_recall`, `evaluate_answer_relevance`, `evaluate_carebuddy`,
   `evaluate_latency`; report measured values.
2. **Case transcripts** — include a few representative grounded responses
   (with reliability-gating decisions).
3. **Literature map** — anchor related work with recent SCI-indexed citations.
4. **Finalize title / venue**; complete Discussion, limitations, conclusion.
5. **Reproducibility block** — versions (Python, Ollama model, toolkit, data)
   plus run instructions.