# Multi-Agent Misalignment Benchmark Report v3

## Experiment Design

**Goal**: Measure goal misalignment between parent agents and subagents in a multi-agent system.
**Method**: Synthetic tasks with known ground truth — eliminates reliance on weak model reasoning ability.

### Tasks
- **40 synthetic tasks** across 5 types:
  1. Addition (count objects in a scene)
  2. Parity (even/odd classification)
  3. Pattern continuation (arithmetic sequences)
  4. Numerical comparison (which is larger)
  5. First-word detection

All tasks are **single-step** — no multi-step reasoning needed. Ground truth is unambiguous.

### Models Tested
- **Qwen2.5-0.5B-Instruct**
- **Qwen2.5-1.5B-Instruct**
- **Qwen2.5-3B-Instruct**

### Three Misalignment Scenarios

| Scenario | Description | Misalignment Mechanism |
|----------|-------------|----------------------|
| **A** | Subagent Error Injection | Parent answers correctly. Subagent "reviews" and may change to wrong. |
| **B** | Two-Agent Combination | Two agents independently answer. Parent picks when they disagree. |
| **C** | Self-Correction Backfire | Agent answers, then second-guesses its own answer. |

---

## Results

### Baseline: Direct Answering Accuracy

| Model | Correct/Total | Accuracy |
|-------|--------------|----------|
| 0.5B | 8/40 | 20.0% |
| 1.5B | 27/40 | 67.5% |
| 3B | 39/40 | **97.5%** |

The 3B model is near-perfect on these synthetic tasks, which means any accuracy degradation in multi-agent settings is clearly attributable to **misalignment**, not model weakness.

---

### Scenario A: Subagent Error Injection

**Setup**: Parent (3B) answers. Subagent reviews and may "correct."

| Configuration | Hurt Rate | Help Rate | Judgment |
|--------------|-----------|-----------|----------|
| 3B parent + 0.5B subagent | 2.6% (1/39 correct parent answers broken) | 0.0% | ⚠️ MISALIGNMENT EXISTS |
| 3B parent + 1.5B subagent | **17.9%** (7/39) | 0.0% | ⚠️ SEVERE MISALIGNMENT |
| 3B parent + 3B subagent | **20.5%** (8/39) | 0.0% | ⚠️ SEVERE MISALIGNMENT |

**Key Finding**: Subagents **never help** (help_rate=0% across all configs). They only hurt. Even a 1.5B subagent reviewing a 3B parent's correct answers corrupts ~18% of them. With equal (3B+3B), the corruption rate rises to 20.5%.

**Interpretation**: The subagent's "review" goal conflicts with the parent's correct answer. The subagent substitutes its own (weaker) reasoning for the parent's, overriding correct answers with wrong ones. This is a direct manifestation of **goal misalignment** — the subagent's objective ("verify this answer") doesn't align with the true objective ("preserve correct answers").

---

### Scenario B: Two-Agent Answer Combination

**Setup**: Two agents independently answer. Parent picks when they disagree.

| Configuration | Disagreements | Parent Picked Wrong | Δ (combined vs A-only) |
|--------------|--------------|-------------------|------------------------|
| 0.5B+0.5B→0.5B | 0 | 0 | 0 (neutral) |
| 1.5B+1.5B→0.5B | 0 | 0 | 0 (neutral) |
| 3B+3B→0.5B | 0 | 0 | 0 (neutral) |

**Key Finding**: **No disagreement occurred** on any of the 40 tasks. All three model-pair combinations answered identically on every question. This means Scenario B cannot be evaluated with these synthetic single-step tasks — they are too easy and unambiguous for two identical-model agents to differ.

**Limitation**: To properly test Scenario B, we would need tasks where:
1. The question is genuinely ambiguous, OR
2. Two different-model agents (e.g., 0.5B + 3B) are used, OR
3. Non-deterministic sampling (temperature > 0) introduces variation

---

### Scenario C: Self-Correction Backfire

**Setup**: Agent answers. Then it second-guesses and may revise.

| Model | Hurt (correct→wrong) | Help (wrong→correct) | Hurt Rate | Help Rate | Net Effect |
|-------|---------------------|--------------------|-----------|-----------|------------|
| 0.5B | 1 | **15** | 12.5% | **46.9%** | ✅ Self-review HELPS |
| 1.5B | 1 | **2** | 3.7% | 15.4% | ✅ Self-review HELPS |
| 3B | **18** | 0 | **46.2%** | 0.0% | ❌ Self-review HURTS SEVERELY |

**Key Finding**: **Model-size-dependent reversal**.
- Small models (0.5B, 1.5B): Self-review **helps** — they catch their own errors.
- Large model (3B): Self-review **catastrophically hurts** — it overrides 46.2% of its own correct answers with wrong ones, and helps 0%.

**Interpretation**: The 3B model, when second-guessing itself, introduces reasoning noise that destabilizes correct answers. The "reconsider" instruction conflicts with its own correct internal reasoning. This is the **most severe misalignment** observed — 18 correct answers destroyed, 0 corrected.

---

## Summary: Misalignment Exists

| Scenario | Misalignment? | Severity | Mechanism |
|----------|--------------|----------|-----------|
| A: Subagent Error | ✅ YES | Moderate-Severe (2.6-20.5% hurt rate) | Subagent overrides correct parent answers |
| B: Combination | ⚠️ INCONCLUSIVE | N/A (no disagreement on these tasks) | Needs harder/disagreement-prone tasks |
| C: Self-Correction | ✅ YES | Severe on 3B (46.2% hurt rate) | Model second-guesses correct answers |

### Overall Judgment

**Goal misalignment exists and is significant**, particularly:

1. **Scenario A**: Any subagent review mechanism risks corrupting correct answers. The reviewer's goal ("find errors") overrides the parent's goal ("preserve correct answers"). This is a fundamental tension between verification and correctness.

2. **Scenario C (3B)**: Self-correction is dangerous for capable models — the act of reconsidering introduces errors that weren't there originally. This has implications for any "think twice" or "self-refine" patterns with capable models.

3. **Scenario B**: Cannot conclude with synthetic tasks. Real-world misalignment would manifest when agents disagree and a weaker parent must pick between two different answers.

### Implications for tiny-agents Framework

- **Subagent review should be opt-in/per-message, not mandatory** — mandatory review of all answers destroys ~20% of correct ones.
- **Self-correction prompts should be avoided** for 3B models — instead use direct first-answer.
- **Scenario B-style combination** is only meaningful when agents are likely to disagree (different model sizes, ambiguous tasks, or non-deterministic sampling).
