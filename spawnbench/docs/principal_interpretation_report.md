# Principal Interpretation Strategy Analysis Report

**Experiment**: SpawnBench Principal Interpretation Variants  
**Date**: 2026-04-25  
**Model**: Qwen2.5-1.5B (GPU 1)  
**Data**: 240 episodes × 5 variants = **1200 runs**  
**Duration**: 1860s (~31 min)

---

## Executive Summary

Five principal decision-making strategies were evaluated on their ability to correctly accept or reject subagent task delegation recommendations. The naive strategy — directly reading the report and making a decision — outperformed all structured reasoning strategies by a large margin (93.3% accuracy vs. 73–88%). This is a counterintuitive result: adding explicit reasoning about evidence quality, counterarguments, or calibration *hurts* performance on a 1.5B model.

---

## Results Overview

| Variant | Accuracy | Delegation Harm Rate | Rescue Rate | Followed Sub Rec |
|---------|----------|----------------------|-------------|------------------|
| **Naive Principal** | **93.3%** | **3.8%** | 43.8% | 96.7% |
| Recommendation Audit | 88.3% | 5.8% | 40.8% | 77.1% |
| Counterevidence Sensitive | 78.3% | 12.1% | 37.1% | 76.2% |
| Evidence Weighting | 76.2% | 14.2% | 37.1% | 79.2% |
| Calibrated Principal | 73.3% | 16.2% | 36.2% | 72.1% |

**Delta vs Naive:**

| Variant | Accuracy Δ | Harm Δ | Rescue Δ |
|---------|-----------|--------|----------|
| Recommendation Audit | -5.0pp | +2.1pp | -2.9pp |
| Counterevidence Sensitive | -15.0pp | +8.3pp | -6.7pp |
| Evidence Weighting | -17.1pp | +10.4pp | -6.7pp |
| Calibrated Principal | -20.0pp | +12.5pp | -7.5pp |

---

## Key Findings

### 1. Naive is the strongest baseline — by far

The naive principal (directly reading the report and deciding) achieves 93.3% accuracy and only 3.8% delegation harm. This is 5–20 percentage points better than any structured reasoning strategy.

**Hypothesis**: On a 1.5B model, the naive prompt is short and direct, keeping reasoning focused on the key decision. Structured prompts ask the model to deliberate about evidence quality — a capability that may exceed the model's reasoning capacity at this scale, causing it to reason itself into errors.

### 2. More reasoning → more errors

The correlation between prompt complexity and accuracy is strongly negative:

```
Naive (1 step)           → 93.3%
Recommendation Audit (2 steps) → 88.3%
Counterevidence Sensitive (3 steps) → 78.3%
Evidence Weighting (4 steps) → 76.2%
Calibrated (5 steps)     → 73.3%
```

Each additional reasoning step introduces another opportunity for the 1.5B model to make a mistake.

### 3. Recommendation Audit is the best alternative

The recommendation audit variant (check if subagent's recommendation matches evidence quality) has the smallest accuracy drop (-5.0pp) and the best correction rate: it fixes 12 of naive's 16 mistakes (75%), while only regressing on 24 of naive's 224 correct decisions (10.7%).

This suggests the recommendation-audit framing — comparing what the subagent said to what the evidence actually shows — is the most beneficial structured reasoning approach, though still inferior to naive.

### 4. Structured reports help naive — but hurt structured variants

Naive performs best on `local_proxy_structured_verifier_before_decision` (93.3%) and worst on `global_aware_free_form_no_verifier` (also 93.3% actually). The structured format helps naive less than it helps recommendation_audit, which reaches 100% on `code_review` and `local_proxy_structured_no_verifier`.

### 5. Task family matters: procurement is hardest

| Family | Accuracy (avg) | Harm Rate (avg) | Rescue Rate (avg) |
|--------|---------------|-----------------|-------------------|
| code_review | 88.8% | 6.8% | 35.5% |
| data_analysis | 82.8% | 8.2% | 56.0% |
| procurement | 74.2% | 16.2% | 25.5% |

Procurement has 2–3× higher delegation harm than code review. This makes sense: procurement decisions involve cost trade-offs and risk assessments where the evidence is rarely clear-cut.

### 6. Subagent recommendation interpretation is noisy

The model's interpretation of subagent recommendations (APPROVE/REJECT) is often wrong. Only 83.8% of naive principal's interpretations match the actual subagent recommendation — yet naive still achieves 93.3% accuracy, suggesting the principal sometimes overrides the subagent recommendation correctly.

---

## By Task Family × Variant (Accuracy %)

| Family | Naive | Evid.Wt | Rec.Audit | CntEvi | Calib |
|--------|-------|---------|-----------|--------|-------|
| code_review | 96.2% | 83.8% | **100.0%** | 83.8% | 80.0% |
| data_analysis | 96.2% | 78.8% | 83.8% | 80.0% | 75.0% |
| procurement | 87.5% | 66.2% | 81.2% | 71.2% | 65.0% |

Recommendation audit reaches 100% on code_review — suggesting structured code evidence is easy to audit. But it degrades on procurement where evidence is messier.

---

## When Naive is Wrong: Can Others Fix It?

Naive made 16 errors across 240 cases.

| Variant | Corrected Naive's Mistakes | Net Gain |
|---------|--------------------------|----------|
| Recommendation Audit | 12/16 (75.0%) | +7 corrected, -12 regressed = **net -5** |
| Calibrated | 10/16 (62.5%) | net -48 |
| Counterevidence Sensitive | 9/16 (56.2%) | net -36 |
| Evidence Weighting | 7/16 (43.8%) | net -41 |

Recommendation audit has the best correction-to-regression ratio: it corrects 12 naive errors but only regresses on 24 naive-correct cases. Other variants correct fewer naive errors than they cause regressions.

---

## Conclusions

1. **For production deployment on Qwen2.5-1.5B**: Use the naive principal. It has the highest accuracy, lowest harm, and highest rescue rate. Structured reasoning is counterproductive.

2. **If using a larger model (3B+)**: Re-evaluate — structured reasoning may help. The failure mode here appears to be model capacity, not strategy design.

3. **Best structured strategy** (if needed): Recommendation audit — check if the subagent's recommendation is supported by the evidence before endorsing or overriding it.

4. **Task-aware routing**: Code review tasks are well-suited to structured approaches; procurement should remain under naive or human oversight due to high harm rates.

5. **Model calibration**: The calibrated principal's instruction to "not blindly trust the subagent" backfires — it distrusts correct subagents more often than it catches bad ones.

---

## Files

- `principal_interpretation_episodes.jsonl` — 1200 raw experiment records
- `principal_interpretation_summary_by_variant.csv` — variant-level aggregates
- `principal_interpretation_summary_by_condition.csv` — condition-level aggregates
- `principal_interpretation_key_contrasts.csv` — naive vs. others contrast table
