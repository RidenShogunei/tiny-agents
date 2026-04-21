# Tiny Agents

A multi-agent framework powered by small language models (0.5B - 3B parameters).

## Core Idea

Instead of relying on a single large model, Tiny Agents orchestrates multiple small specialized models that collaborate through shared memory, KV-cache pooling, and dynamic LoRA switching.

## Architecture

```
Router Agent (1.5B) → Task decomposition & dispatch
├── VLM Agent (3B) → Visual perception & understanding
├── Coder Agent (3B) → Code generation & execution
├── Reason Agent (3B) → Logic reasoning & analysis
└── Critic Agent (0.5B) → Output validation & feedback
```

## Key Features

- **Small Model Only**: All agents use Qwen2.5 (0.5B - 3B)
- **VLM + LLM Collaboration**: Perception-reasoning-execution pipeline
- **Shared Infrastructure**: Base model weight sharing + LoRA hot-swapping
- **KV Cache Pool**: Cross-agent prefix caching for reduced latency

## Benchmarks

- HumanEval (code generation)
- MATH (mathematical reasoning)
- MathVista (multimodal reasoning)

## Hardware Requirements

Minimum: 1x GPU with 16GB VRAM (Qwen2.5-3B + LoRAs)
Recommended: 1x A100 40GB (all agents concurrent)
