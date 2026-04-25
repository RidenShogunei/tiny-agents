# SpawnBench

**Context-Compression Delegation Game**

Evaluates how well a principal agent delegates decisions to a subagent under information asymmetry — the principal sees only a summary report, not the detailed evidence.

> **Core question**: When a principal must decide based on a compressed subagent report rather than raw evidence, what breaks?

---

## Quick Start

```bash
# Run the full experiment (240 episodes)
python spawnbench/runner.py

# Analyze existing results
python spawnbench/analyze.py

# Re-analyze with custom output
python spawnbench/analyze.py spawnbench/data/episodes_refined.jsonl -o /tmp/report.jsonl
```

---

## Architecture

```
┌─────────────────────────┐
│  Principal (pre-decision) │  ← only public context
│  decides: APPROVE/REJECT  │
└────────────┬────────────┘
             │ delegation
             ▼
┌─────────────────────────┐
│  Subagent                │  ← sees full private context
│  produces: report        │
└────────────┬────────────┘
             │ optional
             ▼
┌─────────────────────────┐
│  Verifier               │  ← audits subagent report
│  produces: audit memo    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Principal (final-decision) │  ← public + report (+verifier)
│  decides: APPROVE/REJECT  │
└─────────────────────────┘
```

## 3D Experimental Design

| Dimension | Options | Effect |
|-----------|---------|--------|
| **Objective Scope** | `global_aware` / `local_proxy` | Subagent knows global goal vs. a narrow local objective |
| **Report Format** | `free_form` / `structured` | Free text vs. mandatory field template |
| **Verifier** | `no_verifier` / `verifier_before_decision` | With or without a verifier audit layer |

8 cells × 10 tasks × 3 families = **240 episodes**

## Task Families

- **Code Review** (10 tasks) — decide whether to merge a code patch
- **Data Analysis** (10 tasks) — select the best model/vendor from evidence
- **Procurement** (10 tasks) — choose a supplier considering cost, risk, features

## Key Results (Qwen2.5-1.5B)

| Cell | Fin% | Harm | Rescue |
|------|------|------|--------|
| global_aware / free_form / no_verifier | **96.7%** | 0 | 15 |
| global_aware / free_form / verifier | **96.7%** | 1 | 14 |
| local_proxy / structured / verifier | **70.0%** | **4** | 7 |

- **Delegation is net positive**: 86 rescue vs 9 harm (+77 net)
- **Bottleneck is report interpretation**: subagent oracle match = 94.2%, principal final = 85.4%
- **Structured format severely hurts local_proxy**: -35pt vs free_form
- **Verifier paradox**: adding verifier increased harm 7x (1→8 episodes)

See `docs/spawnbench_smoke_test_report.md` for full analysis.

## Project Structure

```
spawnbench/
├── runner.py          # Main experiment runner (vLLM-based)
├── analyze.py         # Statistical report generator
├── data/
│   ├── tasks.jsonl              # 30 task definitions
│   ├── episodes.jsonl           # Raw experiment output
│   └── episodes_refined.jsonl   # With refined omission flags
├── docs/
│   └── spawnbench_smoke_test_report.md   # Full analysis document
└── README.md
```

## Requirements

```bash
pip install vllm transformers
```

Models (auto-downloaded to `~/.cache/tiny-agents/models/Qwen/`):
- `Qwen2.5-1.5B-Instruct` (default principal + subagent)
- `Qwen2.5-3B-Instruct`
- `Qwen3.5-9B`

## GPU Configuration

Default uses GPU 1 (principal + verifier) and GPU 2 (subagent):

```bash
python spawnbench/runner.py \
  --principal-model /path/to/model \
  --subagent-model /path/to/model \
  --principal-gpu 1 \
  --subagent-gpu 2
```
