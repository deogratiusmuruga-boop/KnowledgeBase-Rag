# ElderDocAI RQ4/RQ5 Evaluation Report

> Synthetic longitudinal clinical records (Synthea), 178 patients, 9,723 patient x year windows.
> **Scope:** descriptive statistics + internal rule-consistency. This is NOT a clinical validation study.

## Datasets used (read-only inputs)

| File | Records |
|---|---:|
| datasets/synthea/elderdocai/processed/clinical_features.json | 178 |
| datasets/synthea/elderdocai/processed/dynamic_care_states.json | 178 |
| datasets/synthea/elderdocai/processed/care_state_timeline.json | 9,723 |
| datasets/synthea/elderdocai/processed/care_state_transitions.json | 9,723 |
| datasets/synthea/elderdocai/processed/adaptive_context.json | 9,723 |
| datasets/synthea/elderdocai/processed/adaptive_assistance.json | 9,723 |
| datasets/synthea/elderdocai/processed/assistance_plans.json | 9,723 |
| datasets/synthea/elderdocai/processed/assistance_decisions.json | 9,723 |

## RQ4 - Dynamic Care-State Evaluation

### A. Patient-level statistics
- patients: **178**
- longitudinal windows: **9723**
- average windows per patient: **54.62** (median 54.0)
- min / max windows per patient: **5 / 88**
- patients with more than one care-state window: **100.0%**

### B. Care-state distribution

Window-level:

| Care State | Count | % of windows |
|---|---|---:|
| STABLE | 1 | 0.01 |
| LOW_ACTIVITY | 1,271 | 13.07 |
| MODERATE_ACTIVITY | 1,257 | 12.93 |
| HIGH_ACTIVITY | 838 | 8.62 |
| NO_DATA | 6,356 | 65.37 |

Patient-level (dominant state per patient):

| Care State | Patients | % of patients |
|---|---|---:|
| STABLE | 0 | 0.0 |
| LOW_ACTIVITY | 0 | 0.0 |
| MODERATE_ACTIVITY | 4 | 2.25 |
| HIGH_ACTIVITY | 0 | 0.0 |
| NO_DATA | 174 | 97.75 |

Patient-level (ever observed in state):

| Care State | Patients | % of patients |
|---|---|---:|
| STABLE | 1 | 0.56 |
| LOW_ACTIVITY | 178 | 100.0 |
| MODERATE_ACTIVITY | 178 | 100.0 |
| HIGH_ACTIVITY | 163 | 91.57 |
| NO_DATA | 176 | 98.88 |

### C. Overall care-state score
- scored windows (non-null): **3367** (null/NO_DATA: 6356)
- min **0.2462**, max **0.9864**, mean **0.6202**, median **0.5851**, stdev **0.1666**
- quartiles: Q1 **0.4496**, Q2 **0.5851**, Q3 **0.7494**

### D. Transition analysis

| Transition Type | Count | % of windows |
|---|---|---:|
| INITIAL_STATE | 178 | 1.83 |
| NO_CHANGE | 1,130 | 11.62 |
| STATE_ESCALATION | 446 | 4.59 |
| STATE_DEESCALATION | 302 | 3.11 |
| INCREASING_ACTIVITY | 121 | 1.24 |
| DECREASING_ACTIVITY | 101 | 1.04 |
| GAP | 7,445 | 76.57 |

- escalation events (STATE_ESCALATION): **446**
- de-escalation events (STATE_DEESCALATION): **302**
- increasing-activity events (INCREASING_ACTIVITY): **121**
- decreasing-activity events (DECREASING_ACTIVITY): **101**
- unchanged-state windows (NO_CHANGE): **1130**

- transitionable windows (excl. INITIAL_STATE/GAP): **2100**
- actual transition events: **970** -> rate **46.19%** (9.98% of all windows)

Transition direction distribution:

| Direction | Count | % of windows |
|---|---|---:|
| DECREASING | 403 | 4.14 |
| INCREASING | 567 | 5.83 |
| INITIAL | 178 | 1.83 |
| UNCHANGED | 1,130 | 11.62 |
| UNKNOWN | 7,445 | 76.57 |

### E. State-to-state transition matrix

Active windows only (previous/current both non-NO_DATA):

| Previous | Current | Count | % of all windows | % within previous |
|---|---|---:|---:|---:|
| HIGH_ACTIVITY | HIGH_ACTIVITY | 523 | 5.38 | 69.18 |
| HIGH_ACTIVITY | LOW_ACTIVITY | 19 | 0.2 | 2.51 |
| HIGH_ACTIVITY | MODERATE_ACTIVITY | 214 | 2.2 | 28.31 |
| LOW_ACTIVITY | HIGH_ACTIVITY | 23 | 0.24 | 6.2 |
| LOW_ACTIVITY | LOW_ACTIVITY | 204 | 2.1 | 54.99 |
| LOW_ACTIVITY | MODERATE_ACTIVITY | 144 | 1.48 | 38.81 |
| MODERATE_ACTIVITY | HIGH_ACTIVITY | 278 | 2.86 | 28.6 |
| MODERATE_ACTIVITY | LOW_ACTIVITY | 68 | 0.7 | 7.0 |
| MODERATE_ACTIVITY | MODERATE_ACTIVITY | 625 | 6.43 | 64.3 |
| MODERATE_ACTIVITY | STABLE | 1 | 0.01 | 0.1 |
| STABLE | HIGH_ACTIVITY | 1 | 0.01 | 100.0 |

All observed pairs (including NO_DATA gap artifacts):

| Previous | Current | Count | % of all windows | % within previous |
|---|---|---:|---:|---:|
| HIGH_ACTIVITY | HIGH_ACTIVITY | 523 | 5.38 | 68.82 |
| HIGH_ACTIVITY | LOW_ACTIVITY | 19 | 0.2 | 2.5 |
| HIGH_ACTIVITY | MODERATE_ACTIVITY | 214 | 2.2 | 28.16 |
| HIGH_ACTIVITY | NO_DATA | 4 | 0.04 | 0.53 |
| LOW_ACTIVITY | HIGH_ACTIVITY | 23 | 0.24 | 1.84 |
| LOW_ACTIVITY | LOW_ACTIVITY | 204 | 2.1 | 16.33 |
| LOW_ACTIVITY | MODERATE_ACTIVITY | 144 | 1.48 | 11.53 |
| LOW_ACTIVITY | NO_DATA | 878 | 9.03 | 70.3 |
| MODERATE_ACTIVITY | HIGH_ACTIVITY | 278 | 2.86 | 23.58 |
| MODERATE_ACTIVITY | LOW_ACTIVITY | 68 | 0.7 | 5.77 |
| MODERATE_ACTIVITY | MODERATE_ACTIVITY | 625 | 6.43 | 53.01 |
| MODERATE_ACTIVITY | NO_DATA | 207 | 2.13 | 17.56 |
| MODERATE_ACTIVITY | STABLE | 1 | 0.01 | 0.08 |
| NO_DATA | HIGH_ACTIVITY | 13 | 0.13 | 0.2 |
| NO_DATA | LOW_ACTIVITY | 820 | 8.43 | 12.9 |
| NO_DATA | MODERATE_ACTIVITY | 256 | 2.63 | 4.03 |
| NO_DATA | NO_DATA | 5,267 | 54.17 | 82.87 |
| STABLE | HIGH_ACTIVITY | 1 | 0.01 | 100.0 |

### F. Longitudinal behavior

Average consecutive-window persistence per state:

| Care State | Avg consecutive windows | Runs observed |
|---|---:|---:|
| STABLE | 1 | 1 |
| LOW_ACTIVITY | 1.19 | 1067 |
| MODERATE_ACTIVITY | 1.99 | 632 |
| HIGH_ACTIVITY | 2.66 | 315 |
| NO_DATA | 5.84 | 1089 |

- average meaningful transitions per patient (excl. INITIAL/GAP): **11.8** (median 12.0)
- average state changes per patient: **5.48**
- patients with >=1 escalation: **96.63%**
- patients with >=1 de-escalation: **85.39%**
- patients with >=1 increase: **53.93%**
- patients with >=1 decrease: **46.07%**
- patients unchanged across all meaningful transitions: **0.56%**
- patients in a single state across all windows: **0.0%**

# RQ5 - Adaptive Assistance Evaluation

### A. Assistance mode distribution

| Assistance Mode | Count | % of windows |
|---|---|---:|
| ADAPTIVE_DEESCALATION | 302 | 3.11 |
| ADAPTIVE_ESCALATION | 446 | 4.59 |
| CONTEXTUAL_SUPPORT | 476 | 4.9 |
| ENHANCED_SUPPORT | 452 | 4.65 |
| FOLLOW_UP_SUPPORT | 101 | 1.04 |
| INITIAL_CONTEXT | 178 | 1.83 |
| LIGHT_SUPPORT | 202 | 2.08 |
| MONITORING_SUPPORT | 121 | 1.24 |
| WAIT_FOR_DATA | 7,445 | 76.57 |

### B. Priority distribution

| Priority | Count | % of windows |
|---|---|---:|
| HIGH | 969 | 9.97 |
| LOW | 7,807 | 80.29 |
| MODERATE | 947 | 9.74 |

### C. Care-state -> assistance mapping

| Care State | Assistance Mode | Count | % of windows | % within state |
|---|---|---:|---:|---:|
| HIGH_ACTIVITY | ADAPTIVE_ESCALATION | 302 | 3.11 | 36.04 |
| HIGH_ACTIVITY | ENHANCED_SUPPORT | 452 | 4.65 | 53.94 |
| HIGH_ACTIVITY | FOLLOW_UP_SUPPORT | 37 | 0.38 | 4.42 |
| HIGH_ACTIVITY | MONITORING_SUPPORT | 34 | 0.35 | 4.06 |
| HIGH_ACTIVITY | WAIT_FOR_DATA | 13 | 0.13 | 1.55 |
| LOW_ACTIVITY | ADAPTIVE_DEESCALATION | 87 | 0.89 | 6.85 |
| LOW_ACTIVITY | FOLLOW_UP_SUPPORT | 1 | 0.01 | 0.08 |
| LOW_ACTIVITY | INITIAL_CONTEXT | 160 | 1.65 | 12.59 |
| LOW_ACTIVITY | LIGHT_SUPPORT | 202 | 2.08 | 15.89 |
| LOW_ACTIVITY | MONITORING_SUPPORT | 1 | 0.01 | 0.08 |
| LOW_ACTIVITY | WAIT_FOR_DATA | 820 | 8.43 | 64.52 |
| MODERATE_ACTIVITY | ADAPTIVE_DEESCALATION | 214 | 2.2 | 17.02 |
| MODERATE_ACTIVITY | ADAPTIVE_ESCALATION | 144 | 1.48 | 11.46 |
| MODERATE_ACTIVITY | CONTEXTUAL_SUPPORT | 476 | 4.9 | 37.87 |
| MODERATE_ACTIVITY | FOLLOW_UP_SUPPORT | 63 | 0.65 | 5.01 |
| MODERATE_ACTIVITY | INITIAL_CONTEXT | 18 | 0.19 | 1.43 |
| MODERATE_ACTIVITY | MONITORING_SUPPORT | 86 | 0.88 | 6.84 |
| MODERATE_ACTIVITY | WAIT_FOR_DATA | 256 | 2.63 | 20.37 |
| NO_DATA | WAIT_FOR_DATA | 6,356 | 65.37 | 100.0 |
| STABLE | ADAPTIVE_DEESCALATION | 1 | 0.01 | 100.0 |

### D. Transition -> assistance mapping

| Transition Type | Assistance Mode | Count | % of windows | % within transition |
|---|---|---:|---:|---:|
| DECREASING_ACTIVITY | FOLLOW_UP_SUPPORT | 101 | 1.04 | 100.0 |
| GAP | WAIT_FOR_DATA | 7,445 | 76.57 | 100.0 |
| INCREASING_ACTIVITY | MONITORING_SUPPORT | 121 | 1.24 | 100.0 |
| INITIAL_STATE | INITIAL_CONTEXT | 178 | 1.83 | 100.0 |
| NO_CHANGE | CONTEXTUAL_SUPPORT | 476 | 4.9 | 42.12 |
| NO_CHANGE | ENHANCED_SUPPORT | 452 | 4.65 | 40.0 |
| NO_CHANGE | LIGHT_SUPPORT | 202 | 2.08 | 17.88 |
| STATE_DEESCALATION | ADAPTIVE_DEESCALATION | 302 | 3.11 | 100.0 |
| STATE_ESCALATION | ADAPTIVE_ESCALATION | 446 | 4.59 | 100.0 |

### E. Assistance-plan coverage

| Artifact | Records | Unique (patient, window) | Matched to timeline | % of 9,723 windows | Missing |
|---|---:|---:|---:|---:|---:|
| adaptive_context | 9,723 | 9,723 | 9,723 | 100.0 | 0 |
| adaptive_assistance | 9,723 | 9,723 | 9,723 | 100.0 | 0 |
| assistance_plans | 9,723 | 9,723 | 9,723 | 100.0 | 0 |
| assistance_decisions | 9,723 | 9,723 | 9,723 | 100.0 | 0 |

- null rates: {'assistance_mode_null': 0, 'priority_null': 0, 'plan_strategy_null': 0, 'plan_actions_missing': 0, 'decision_strategy_null': 0}

### F. Internal rule-consistency

- mode agreement with documented rule tables: **100.0%** (9723/9723)
- priority agreement with documented rule tables: **100.0%** (9723/9723)
- mismatches: **0**
- plans strategy agreement: **100.0%**
- decisions strategy agreement: **100.0%**

State x priority (observed):

| Care State | Priority | Count | % within state |
|---|---|---:|---:|
| HIGH_ACTIVITY | HIGH | 825 | 98.45 |
| HIGH_ACTIVITY | LOW | 13 | 1.55 |
| LOW_ACTIVITY | LOW | 1,182 | 93.0 |
| LOW_ACTIVITY | MODERATE | 89 | 7.0 |
| MODERATE_ACTIVITY | HIGH | 144 | 11.46 |
| MODERATE_ACTIVITY | LOW | 256 | 20.37 |
| MODERATE_ACTIVITY | MODERATE | 857 | 68.18 |
| NO_DATA | LOW | 6,356 | 100.0 |
| STABLE | MODERATE | 1 | 100.0 |

Semantic checks:

| Check | % |
|---|---:|
| HIGH_ACTIVITY_windows | 838 |
| HIGH_ACTIVITY_high_priority_pct | 98.45 |
| HIGH_ACTIVITY_ENHANCED_or_escalation_mode_pct | 89.98 |
| MODERATE_ACTIVITY_windows | 1257 |
| MODERATE_ACTIVITY_moderate_priority_pct | 68.18 |
| LOW_ACTIVITY_windows | 1271 |
| LOW_ACTIVITY_low_priority_pct | 93.0 |
| STABLE_windows | 1 |
| STABLE_low_priority_pct | 0.0 |
| NO_DATA_windows | 6356 |
| NO_DATA_low_priority_pct | 100.0 |
| STATE_ESCALATION_ADAPTIVE_ESCALATION_pct | 100.0 |
| STATE_ESCALATION_high_priority_pct | 100.0 |
| STATE_DEESCALATION_ADAPTIVE_DEESCALATION_pct | 100.0 |
| INCREASING_ACTIVITY_MONITORING_SUPPORT_pct | 100.0 |
| INCREASING_ACTIVITY_priority_ge_MODERATE_pct | 100.0 |
| DECREASING_ACTIVITY_FOLLOW_UP_SUPPORT_pct | 100.0 |
| GAP_WAIT_FOR_DATA_pct | 100.0 |
| INITIAL_STATE_INITIAL_CONTEXT_pct | 100.0 |
| NO_CHANGE_state_default_mode_pct | 100.0 |

## Limitations

- The dataset is **synthetic** (Synthea-derived). It is not real patient data.
- Care states encode documented activity intensity; `NO_DATA` windows do not imply clinical stability.
- Persistence/transition metrics are computed on calendar-year windows; durations are therefore at one-year granularity.
- The rule-consistency check verifies the generated labels match the framework's documented deterministic rules; it does not establish clinical correctness.
- Prevalence of `GAP`/`NO_DATA` windows (76.6% of all windows) dominates the window-level distributions; patient-level and transition-exclusive statistics should be read alongside them.

## Figures

- `rq4_fig1_care_state_distribution.png`
- `rq4_fig2_transition_distribution.png`
- `rq4_fig3_transition_matrix_heatmap.png`
- `rq5_fig4_state_assistance_stacked.png`
- `rq5_fig5_transition_assistance_stacked.png`
- `rq5_fig6_transition_priority_grouped.png`

## Files

- `data/evaluation_results/rq4_rq5_care_state_assistance_results.json` (machine-readable)
- `data/evaluation_results/rq4_rq5_care_state_assistance_report.md` (this report)

## Interpretation

The results demonstrate that ElderDocAI converts longitudinal clinical records into dynamic care states, detects and characterizes state transitions, and varies adaptive assistance (mode, priority, strategy) according to the detected state and transition in a manner that is internally consistent with the framework's documented rules. This is a demonstration of dynamic, care-state-aware adaptive assistance on synthetic records; it is not evidence of clinical benefit.
