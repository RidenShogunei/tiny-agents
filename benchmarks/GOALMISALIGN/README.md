# GoalMisAlignBench

> **GoalMisAlignBench**: Isolating Goal Misalignment from Extra-Step Loss in Multi-Agent Systems

## Motivation

When a parent agent delegates a sub-task to a sub-agent, accuracy drops for two distinct reasons:

1. **Extra-step loss**: The additional reasoning step introduces noise, even with a perfect sub-agent.
2. **Goal misalignment loss**: The sub-agent's output is locally correct but misaligned with what the parent actually needs.

Existing benchmarks conflate these. GoalMisAlignBench separates them.

---

## Core Design: Three Modes

For each case, all three modes are evaluated with the **same model**:

| Mode | Description | What It Measures |
|------|-------------|-----------------|
| **A** | Single model answers end-to-end | True capability baseline |
| **B** | Parent delegates to real sub-agent → gets output → answers | Full delegation (step loss + misalignment) |
| **C** | Single model receives **oracle** (ground-truth) sub-agent output | Extra step loss **only** |

### Key Metrics

```
extra_step_loss = A% − C%    ← harm from delegation itself (even with perfect sub-agent)
misalign_loss   = C% − B%    ← pure goal misalignment
total_loss      = A% − B%
```

### Decision Rules

| Condition | Interpretation |
|-----------|---------------|
| C% < A% | Extra step is harmful even with perfect sub-agent |
| C% > B% | Goal misalignment exists |
| C% = B% | All loss is from extra step |
| B% > A% | Delegation helps (rare) |

---

## Dataset

60 cases across 6 misalignment types (10 each):

| Type | Description | Example |
|------|-------------|---------|
| `format_mismatch` | Sub-agent gives correct info but wrong format | `8611` vs `8,848` |
| `semantic_drift` | Sub-agent answers delegated question but misses broader intent | Counts 'r'=3, but question asks r vs b |
| `precision_loss` | Sub-agent gives range/rating where parent needs exact value | "1990s" when parent needs exact year |
| `context_collapse` | Complex multi-step reasoning collapses to simple answer | BC/AD cross-era comparison |
| `irrelevant_delegation` | Question already answerable without delegation | Both years in original question |
| `partial_delegation` | Sub-agent only addresses part of what parent needs | Only asks Wednesday's length, not Thursday's |

3 difficulty levels: easy (20 cases), medium (24 cases), hard (16 cases)

---

## Usage

```bash
# Run full 60-case benchmark with 3B model
python benchmarks/goal_misalign_bench.py \
    --output-dir ./benchmark_results/goal_misalign \
    --parent-model 3B

# Run subset for debugging
python benchmarks/goal_misalign_bench.py --n-samples 12

# Aggregate across multiple runs
python benchmarks/goal_misalign_bench.py \
    --aggregate \
    --result-dirs ./benchmark_results/goal_misalign/run_xxx

# Visualize results
python benchmarks/visualize_alignment.py ./benchmark_results/goal_misalign/run_xxx \
    --output ./benchmark_results/goal_misalign/run_xxx/bench_chart.png

# Aggregate and visualize multiple runs
python benchmarks/visualize_alignment.py ./benchmark_results/goal_misalign \
    --aggregate -o ./benchmark_results/goal_misalign/aggregate_chart.png
```

---

## Results (Qwen2.5-3B, 60 cases)

```
Mode A (single 3B, no subagent):         72%
Mode C (3B + oracle subagent):           57%  ← extra step loss: -15pp
Mode B (3B + real subagent):             42%  ← misalign loss:   -15pp

Total delegation loss:    -30pp
Extra step loss (A−C):   -15pp
Goal misalignment (C−B): -15pp  ← pure misalignment confirmed!
```

### By Misalignment Type

| Type | N | A | C | B | A−C | C−B |
|------|---|---|---|---|-----|-----|
| format_mismatch | 10 | 70% | 60% | 40% | -10 | -20 |
| semantic_drift | 10 | 70% | 50% | 40% | -20 | -10 |
| precision_loss | 10 | 60% | 50% | 40% | -10 | -10 |
| context_collapse | 10 | 70% | 60% | 50% | -10 | -10 |
| irrelevant_delegation | 10 | 80% | 70% | 50% | -10 | -20 |
| partial_delegation | 10 | 80% | 60% | 40% | -20 | -20 |

---

## Key Findings

1. **Goal misalignment is real and significant**: 15pp loss (50% of total) is attributable to pure misalignment, not extra reasoning steps.

2. **`format_mismatch` and `partial_delegation` are the worst types**: Both show 20pp misalignment loss — small format differences in sub-agent output cause cascading parent errors.

3. **`semantic_drift` causes extra-step to hurt the most**: 20pp drop from A→C, meaning the oracle sub-agent output actively misleads the parent compared to direct answering.

4. **Even oracle sub-agent output causes losses**: C < A in most types, meaning the extra step alone is harmful regardless of sub-agent quality.

---

## Project Structure

```
benchmarks/
  goal_misalign_bench.py      # Main benchmark runner (3 modes)
  visualize_alignment.py       # Visualization (charts + aggregate)
  multi_hop_qa_alignment.py    # Legacy V2 (12 cases)
GOALMISALIGN/
  README.md                    # This file
benchmark_results/goal_misalign/
  run_YYYYMMDD_HHMMSS/
    modeA.json                 # Raw Mode A results
    modeB.json                 # Raw Mode B results
    modeC.json                 # Raw Mode C results
    summary.json                # Auto-generated summary + verdicts
    bench_chart.png             # Visualization
```

---

## Citation

```
RidenShogunei/tiny-agents: "GoalMisAlignBench: Isolating Goal Misalignment 
from Extra-Step Loss in Multi-Agent Systems" (2026)
```
