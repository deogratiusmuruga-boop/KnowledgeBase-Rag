# ElderDocAI Ablation Study Report

> Synthetic longitudinal clinical records (Synthea-derived) and 16 gold QA questions over the ElderDocAI knowledge base.
> **Scope:** architectural / internal-consistency evaluation. This is NOT a clinical validation study and implies no clinical benefit.

## 1. Experimental Objective

Quantify the contribution of each ElderDocAI component across retrieval / evidence grounding, answer quality, reliability assessment, dynamic care-state awareness, and adaptive assistance behavior, via controlled condition-by-condition comparison.

## 2. Experimental Design

A0 (full system) is the reference. Each ablation removes exactly one component while holding inputs constant: same 16 gold questions, same KB index / embedding model, same generation model (llama3.2, temperature 0), same reliability config, same judge prompt/rubric, same patient profile for every condition attaching a patient, and the same 9,723 windows for the care-state/adaptive analysis.

## 3. Conditions

| Cond | Name | Retrieval | Rerank | Reliability gate | Dynamic care state | Adaptive assistance |
|---|---|---|---|---|---|---|
| A0 | Full ElderDocAI | hybrid_full | True | True | True | adaptive |
| A1 | No Dynamic Care State | hybrid_full | True | True | False | none |
| A2 | No Adaptive Assistance | hybrid_full | True | True | True | fixed_default |
| A3 | Dense Retrieval Only | dense_only | False | True | True | adaptive |
| A4 | Hybrid Without Reranking | hybrid_no_rerank | False | True | True | adaptive |
| A5 | No Reliability Gate | hybrid_full | True | False | True | adaptive |
| A6 | Static Care Profile Only | hybrid_full | True | True | False | none |

## 4. Dataset and Number of Cases

- 16 gold QA questions per condition, 9,723 synthetic patient x year windows (178 synthetic patients) for the adaptive-assistance analysis.

## 5. Metrics

- Retrieval: source accuracy, evidence/chunk recall@3, supporting-span support, precision@3, MRR(source).
- Answer: RQ2-judge faithfulness and relevance, successful-generation rate.
- Reliability: mean/min reliability, decision distribution (same config).
- Dynamic/adaptive: assistance-mode diversity, state-driven and transition-driven adaptation rates, exact agreement with the full system.

## 6. Per-Condition Results

### 6.1 Main table

|Condition|Retrieval Accuracy (%)|Evidence Recall@3|Faithfulness|Answer Relevance|Reliability|Adaptive coverage|
|--:|--:|--:|--:|--:|--:|--:|
|Full ElderDocAI|100.0|0.7302|0.9375|0.8906|0.9394|window-level (Sec. 7)|
|No Dynamic Care State|100.0|0.7302|0.9688|0.9531|0.9394|window-level (Sec. 7)|
|No Adaptive Assistance|100.0|0.7302|0.8906|0.9375|0.9394|window-level (Sec. 7)|
|Dense Retrieval Only|87.5|0.6417|0.9219|0.8281|0.9265|window-level (Sec. 7)|
|Hybrid Without Reranking|93.75|0.6573|0.9219|0.7969|0.9293|window-level (Sec. 7)|
|No Reliability Gate|100.0|0.7302|0.8906|0.9375|0.9394|window-level (Sec. 7)|
|Static Care Profile Only|100.0|0.7302|0.9688|0.9531|0.9394|window-level (Sec. 7)|

*Adaptive coverage is a window-level property (Sec. 7), not a 16-question metric.*

### 6.2 Per-condition aggregates

**A0 - Full ElderDocAI**

- retrieval accuracy 100.0% | recall@3 0.7302 | precision@3 0.4375 | MRR 1.0 | span support 87.5%
- faithfulness 0.9375 | relevance 0.8906 | successful gen 100.0%
- reliability mean 0.9394 | decision dist {'ACCEPT': 16}
- runtime 31.9s | errors 0

**A1 - No Dynamic Care State**

- retrieval accuracy 100.0% | recall@3 0.7302 | precision@3 0.4375 | MRR 1.0 | span support 87.5%
- faithfulness 0.9688 | relevance 0.9531 | successful gen 100.0%
- reliability mean 0.9394 | decision dist {'ACCEPT': 16}
- runtime 30.38s | errors 0

**A2 - No Adaptive Assistance**

- retrieval accuracy 100.0% | recall@3 0.7302 | precision@3 0.4375 | MRR 1.0 | span support 87.5%
- faithfulness 0.8906 | relevance 0.9375 | successful gen 100.0%
- reliability mean 0.9394 | decision dist {'ACCEPT': 16}
- runtime 22.09s | errors 0

**A3 - Dense Retrieval Only**

- retrieval accuracy 87.5% | recall@3 0.6417 | precision@3 0.3542 | MRR 0.8438 | span support 75.0%
- faithfulness 0.9219 | relevance 0.8281 | successful gen 100.0%
- reliability mean 0.9265 | decision dist {'ACCEPT': 16}
- runtime 24.65s | errors 0

**A4 - Hybrid Without Reranking**

- retrieval accuracy 93.75% | recall@3 0.6573 | precision@3 0.375 | MRR 0.9062 | span support 81.25%
- faithfulness 0.9219 | relevance 0.7969 | successful gen 100.0%
- reliability mean 0.9293 | decision dist {'ACCEPT': 16}
- runtime 35.06s | errors 0

**A5 - No Reliability Gate**

- retrieval accuracy 100.0% | recall@3 0.7302 | precision@3 0.4375 | MRR 1.0 | span support 87.5%
- faithfulness 0.8906 | relevance 0.9375 | successful gen 100.0%
- reliability mean 0.9394 | decision dist {'ACCEPT': 16}
- runtime 30.76s | errors 0

**A6 - Static Care Profile Only**

- retrieval accuracy 100.0% | recall@3 0.7302 | precision@3 0.4375 | MRR 1.0 | span support 87.5%
- faithfulness 0.9688 | relevance 0.9531 | successful gen 100.0%
- reliability mean 0.9394 | decision dist {'ACCEPT': 16}
- runtime 30.38s | errors 0

## 7. Adaptive-Assistance Comparison (9,723 windows)

|Policy|Distinct modes|Mode distribution|Priority distribution|
|--:|--:|--:|--:|
|full|9|{"WAIT_FOR_DATA": 7445, "CONTEXTUAL_SUPPORT": 476, "ENHANCED_SUPPORT": 452, "ADAPTIVE_ESCALATION": 446, "ADAPTIVE_DEESCALATION": 302, "LIGHT_SUPPORT": 202, "INITIAL_CONTEXT": 178, "MONITORING_SUPPORT": 121, "FOLLOW_UP_SUPPORT": 101}|{"LOW": 7807, "HIGH": 969, "MODERATE": 947}|
|state_only_A2_proxy|5|{"WAIT_FOR_DATA": 6356, "LIGHT_SUPPORT": 1271, "CONTEXTUAL_SUPPORT": 1257, "ENHANCED_SUPPORT": 838, "MAINTENANCE_SUPPORT": 1}|{"LOW": 7628, "MODERATE": 1257, "HIGH": 838}|
|static_patient_A6_proxy|3|{"CONTEXTUAL_SUPPORT": 4313, "ENHANCED_SUPPORT": 4093, "MAINTENANCE_SUPPORT": 1317}|{"MODERATE": 4313, "HIGH": 4093, "LOW": 1317}|
|fixed_default_A1_proxy|1|{"GENERAL_SUPPORT": 9723}|{"LOW": 9723}|

- state-sensitive adaptation rate: **100.0%** (9723 windows changed vs fixed default)
- transition-sensitive adaptation rate: **23.01%** (2237 windows changed vs state-only policy)
- exact agreement full vs state-only (A2 proxy): **76.99%**
- exact agreement full vs static-patient (A6 proxy): **6.14%**
- exact agreement full vs fixed-default (A1 proxy): **0.0%**
- priority-shift rate vs state-only: **5.17%**

Transition sensitivity breakdown:

|Transition|Windows|Mode differs from state-only|%|
|--:|--:|--:|--:|
|INITIAL_STATE|178|178|100.0|
|NO_CHANGE|1130|0|0.0|
|STATE_ESCALATION|446|446|100.0|
|STATE_DEESCALATION|302|302|100.0|
|INCREASING_ACTIVITY|121|121|100.0|
|DECREASING_ACTIVITY|101|101|100.0|
|GAP|7445|1089|14.63|

## 8. Component-Wise Comparison (Delta = Ablated - Full)

|Ablation|Faithfulness (full -> abl)|Relevance (full -> abl)|Reliability (full -> abl)|Retrieval Accuracy (full -> abl)|Interpretation|
|--:|--:|--:|--:|--:|--:|
|No Dynamic Care State|0.9375 -> 0.9688|0.8906 -> 0.9531|0.9394 -> 0.9394|100.0% -> 100.0%|Static profile + plain RAG. Removes dynamic care state, transitions, and state-derived adaptive assistance.|
|No Adaptive Assistance|0.9375 -> 0.8906|0.8906 -> 0.9375|0.9394 -> 0.9394|100.0% -> 100.0%|Dynamic care state is detected, but assistance is a FIXED default strategy (GENERAL_SUPPORT/LOW).|
|Dense Retrieval Only|0.9375 -> 0.9219|0.8906 -> 0.8281|0.9394 -> 0.9265|100.0% -> 87.5%|FAISS dense retrieval only. No BM25, no score fusion, no CrossEncoder reranking.|
|Hybrid Without Reranking|0.9375 -> 0.9219|0.8906 -> 0.7969|0.9394 -> 0.9293|100.0% -> 93.75%|Dense + BM25 hybrid fusion but no CrossEncoder reranking.|
|No Reliability Gate|0.9375 -> 0.8906|0.8906 -> 0.9375|0.9394 -> 0.9394|100.0% -> 100.0%|Reliability computed/recorded, but the reliability+decision section is absent from the prompt; generation always proceeds.|
|Static Care Profile Only|0.9375 -> 0.9688|0.8906 -> 0.9531|0.9394 -> 0.9394|100.0% -> 100.0%|Conventional personalized RAG baseline: static profile + retrieval + grounded answer. Prompt-identical to A1 by construction.|

## 9. Reliability Comparison (A0 vs A5)

- **A0**: avg 0.9394, min 0.8571, decision dist {'ACCEPT': 16, 'REFINE': 0, 'RE-RETRIEVE': 0, 'REJECT': 0}
- **A5**: avg 0.9394, min 0.8571, decision dist {'ACCEPT': 16, 'REFINE': 0, 'RE-RETRIEVE': 0, 'REJECT': 0}
- Reliability calculation and retrieved evidence are identical in A0/A5; only the prompt differs (A5 removes the reliability/decision block).
- On these 16 high-quality in-scope gold questions every decision is ACCEPT; the dataset does not expose gating behavior on low-reliability evidence.

## 10. Retrieval Comparison

- **A0**: acc 100.0% | recall@3 0.7302 | span 87.5% | MRR 1.0
- **A1**: acc 100.0% | recall@3 0.7302 | span 87.5% | MRR 1.0
- **A2**: acc 100.0% | recall@3 0.7302 | span 87.5% | MRR 1.0
- **A3**: acc 87.5% | recall@3 0.6417 | span 75.0% | MRR 0.8438
- **A4**: acc 93.75% | recall@3 0.6573 | span 81.25% | MRR 0.9062
- **A5**: acc 100.0% | recall@3 0.7302 | span 87.5% | MRR 1.0
- **A6**: acc 100.0% | recall@3 0.7302 | span 87.5% | MRR 1.0

## 11. Statistical / Significance Analysis

Paired deltas (ablated - A0) per question, with a paired Wilcoxon signed-rank test where justified. n is small (<=16); results are descriptive and must not be read as proof of significance.

### No Dynamic Care State

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|0.0312|0.0|wilcoxon_signed_rank|0.414216|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.234194|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### No Adaptive Assistance

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.256839|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|0.0469|0.0|wilcoxon_signed_rank|0.496242|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Dense Retrieval Only

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|-0.0156|0.0|wilcoxon_signed_rank|0.705457|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|-0.0625|0.0|wilcoxon_signed_rank|0.517634|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Hybrid Without Reranking

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|-0.0156|0.0|wilcoxon_signed_rank|0.654721|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|-0.0938|0.0|wilcoxon_signed_rank|0.234194|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### No Reliability Gate

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|-0.0469|0.0|wilcoxon_signed_rank|0.317311|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|0.0469|0.0|wilcoxon_signed_rank|0.523609|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

### Static Care Profile Only

|Metric|n paired|Delta mean|Delta median|Test|p-value|Note|
|--:|--:|--:|--:|--:|--:|--:|
|Faithfulness|16|0.0312|0.0|wilcoxon_signed_rank|0.414216|descriptive only; small n; dz undefined/inflated when sd=0|
|Answer Relevance|16|0.0625|0.0|wilcoxon_signed_rank|0.234194|descriptive only; small n; dz undefined/inflated when sd=0|
|Reliability|0|n/a|n/a|not_applicable|n/a|no paired values (judge parse failures or missing rows)|

## 12. Figures

- `fig1_retrieval_ablation.png`
- `fig2_answer_quality_ablation.png`
- `fig3_reliability_ablation.png`
- `fig4_adaptive_assistance_ablation.png`

## 13. Reproducibility Information

- Console log saved to `C:\Users\chosun\Documents\ElderDocAI-System\data\evaluation_results\ablation\ablation_console_output.txt` (213 lines).
- Generation model llama3.2 (temperature 0 / top_p 0.1 / top_k 10); judge llama3.2.
- Reproducible from `scripts/evaluate_ablation.py` with the same indexed KB and gold-QA set.

## 14. Limitations

- 16 gold questions is a small, curated, in-scope set; most are straightforward and high-reliability, creating ceiling effects that limit retrieval and reliability-gate differentiation.
- The records are SYNTHETIC (Synthea-derived). No clinical validity is claimed.
- A1 and A6 share the same prompt by construction and reuse the same generation outputs; they differ only in the reported label.
- A2's window-level signal is a state-only proxy of the fixed-default strategy; the production system ships no non-adaptive fallback to ablate directly.
- A5 removes the reliability section from the prompt; the reliability calculation itself is unchanged and still reported.

## 15. Interpretation

Retrieval/reranking/reliability ablations produce high and internally consistent retrieval and answer-quality scores on the 16 in-scope gold questions, so retrieval/reliability differences are small (ceiling effects) and cannot be over-interpreted. The window-level analysis demonstrates that adaptive assistance is strongly state- and transition-dependent: a fixed-default or static-patient policy matches the full system's assistance decisions on a small fraction of the 9,723 windows, whereas a per-window state-only policy matches a large share. This shows that the dynamic care-state and transition layers meaningfully change assistance behavior, while evidence grounding and reliable generation remain stable when those layers are removed. These are architectural / internal-consistency results on synthetic data, not clinical evidence.
