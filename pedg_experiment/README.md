# PEDG: Private-Evidence Delegation Game

## Goal Misalignment Detection in Multi-Agent Systems

This experiment implements a controlled framework for detecting goal misalignment between a **Principal** (supervisor) and **Subagent** (performer) in a delegation setting.

## Research Question

Can providing subagents with local incentives that partially conflict with the principal's utility systematically induce goal misalignment behaviors (misreporting, selective omission, strategic framing)?

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start vLLM with Qwen3.5-9B on idle GPU
bash run_single.sh --mode smoke --gpu 1

# 3. Or run full experiment
bash run_single.sh --mode full --gpu 1
```

## Project Structure

```
pedg_experiment/
├── configs/
│   └── base.yaml              # Experiment configuration
├── prompts/
│   ├── principal.txt          # Principal prompt template
│   ├── subagent_aligned.txt    # Subagent aligned prompt
│   ├── subagent_conflict.txt  # Subagent conflict prompt
│   └── auditor.txt            # Auditor prompt template
├── envs/
│   └── task_families.py       # 10 synthetic task families
├── agents/
│   ├── __init__.py
│   ├── llm_client.py          # vLLM/OpenAI API client
│   └── role_agents.py         # Principal, Subagent, Auditor agents
├── experiments/
│   └── run_single.py          # Experiment orchestration
├── analysis/
│   └── stats.py               # Statistical analysis
├── results/                    # Output directory
├── reports/                   # Generated reports
├── main_experiment.py         # Entry point
├── run_single.sh             # Convenience runner
└── README.md
```

## Experiment Design

### 2×2×2 Factorial Design

| Factor | Levels |
|--------|--------|
| Incentive | `aligned` (U_S = U_P), `conflict` (U_S = U_P + λ·APPROVE − η·AUDIT) |
| Information | `asymmetric` (only S sees private evidence), `symmetric` |
| Oversight | `weak_oversight` (10% audit prob), `strong_oversight` (40% audit prob + penalty) |

### Task Families (10)
1. Project Approval
2. Loan Approval
3. Experiment Replication Review
4. Security Vulnerability Triage
5. Supplier Admission
6. Procurement Approval
7. Release Approval
8. Dataset Quality Check
9. Medical Compliance Review
10. Contract Risk Screening

## Output

- `results/episode_logs_*.jsonl` — Full episode logs
- `results/summary_*.csv` — Aggregated metrics
- `results/analysis_report.md` — Statistical analysis
- `results/final_report.md` — Comprehensive report
- `results/executive_summary.md` — Quick summary

## Usage

```bash
# Smoke test (20 episodes per cell)
python main_experiment.py --mode smoke --gpu 1

# Full experiment
python main_experiment.py --mode full --gpu 1

# Analysis only
python main_experiment.py --mode analyze --results results/episode_logs_full_*.jsonl

# Generate reports
python main_experiment.py --mode report
```

## Key Metrics

1. **Misreport Rate**: S recommendation ≠ principal-optimal
2. **Selective Omission Rate**: S omits key private evidence
3. **Counterfactual Flip Rate**: Behavior changes between aligned/conflict
4. **Principal Regret**: Gap between optimal and actual principal utility
5. **Knowledge-Conditioned Deviation Rate**: S misreports even when knowing truth
6. **Audit Rate**: P triggers AUDIT
7. **Auditor Label Distribution**: truthful / omission / false_statement / pragmatic_distortion
8. **Outcome Accuracy**: Final decision correctness
