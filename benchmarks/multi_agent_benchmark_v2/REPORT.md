# Multi-Agent Benchmark v2 — Report

**Date:** 2026-04-24 22:56
**Tasks:** 40 GSM8K problems

## Scenario 1: Cross-Model Information Gathering

*Weak parent (0.5B/1.5B) coordinates a strong subagent (3B) to solve problems.*

| System | Accuracy | vs Parent Alone |
|--------|----------|----------------|

## Scenario 2: Self-Refinement

*Single model answers, then critiques/refines its own answer.*

| Model | Initial | After Refinement | Δ |
|-------|---------|------------------|---|

## Scenario 3: Two-Agent Debate

*Two agents reason differently, one acts as arbiter.*

| Debate Pair | Accuracy | vs Single Agent A |
|-------------|----------|-------------------|

## Key Findings

### Scenario 1: Cross-Model Delegation

### Scenario 2: Self-Refinement

### Scenario 3: Debate

## Reproducing

```bash
cd /home/jinxu/tiny-agents
python benchmarks/MULTI_AGENT_BENCHMARK_V2.py --n-samples 40 --scenarios baseline s1 s2 s3
```
