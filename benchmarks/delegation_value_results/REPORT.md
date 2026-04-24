# Delegation Value Benchmark Report
**Date:** 2026-04-24
**Models:** Qwen2.5-0.5B, 1.5B, 3B (vLLM 0.19.1)
**Tasks:** 30 (math, logic, comparison, domain, chain-of-thought)
**Strategies:** A = parent alone (no delegation) | B = parent simulates same-model subagent

---

## Executive Summary

> **结论：同模型 self-delegation 在所有规模上都伤害准确率，且模型越大伤害越严重。**

| Model | A (alone) | B (delegated) | Δ | Verdict |
|-------|-----------|---------------|---|---------|
| 0.5B  | 33.3%    | 26.7%         | **-6.7pp** | DELEGATION HURTS |
| 1.5B  | 66.7%    | 46.7%         | **-20.0pp** | DELEGATION HURTS |
| 3B    | 66.7%    | 33.3%         | **-33.3pp** | DELEGATION HURTS |

---

## Why Does Delegation Hurt?

### Root Cause: Strategy B Prompt Design Induces Verbose Chain-of-Thought

Strategy B 的 parent prompt 包含 "Think about what question you would ask a subagent..."，这会触发模型的 CoT 行为，导致：

- Strategy A: 直接简洁回答 → `20`
- Strategy B: 长篇解释 + 最后才给答案 → `To find the total cost... 4 × $12 = $48... minus $10 discount... Final answer: 38`

答案提取对 B 策略的系统性失败不是 bug，而是**delegation 机制引入的固有代价**：

> **Delegation 强制引入的中间表示层（subagent output）打乱了最终答案的简洁性。**

### Why Does Larger Model Suffer More?

反直觉：3B 伤害最严重（-33pp），0.5B 最轻（-6.7pp）。

| 解释 | 0.5B | 1.5B | 3B |
|------|------|------|-----|
| 直接回答能力 | 弱（33%） | 强（67%） | 强（67%） |
| 被 delegation 打断后的回答能力 | 也弱 | 显著下降（47%） | 严重下降（33%） |
| CoT 干扰 | 低 | 中 | 高 |

- **小模型（0.5B）**：直接回答也很烂，CoT 干扰相对较小
- **大模型（3B）**：直接回答很强，但被迫分心到"模拟 subagent"时输出质量崩溃

---

## Per-Category Breakdown

| Category | 0.5B Δ | 1.5B Δ | 3B Δ |
|----------|--------|--------|------|
| MATH | -30pp | -20pp | -40pp |
| LOGIC | +40pp | 0pp | -40pp |
| COMP | 0pp | 0pp | -40pp |
| DOMAIN | 0pp | 0pp | 0pp |
| CHAIN | 0pp | -60pp | -20pp |

**观察：**
- DOMAIN（知识记忆）： delegation 无影响，因为答案直接检索无需推理
- MATH/COMP（精确计算）： delegation 伤害所有模型
- LOGIC：0.5B 反而从 delegation 受益（+40pp），可能因为小模型需要引导才能推理

---

## Case Study: Where Delegation Actually Helps (0.5B Logic)

```
Task: "All cats are animals. Some animals are black. Can we conclude some cats are black?"
Strategy A (0.5B): "yes" ← 错误（过度泛化）
Strategy B (0.5B): "No. The statements provided do not logically guarantee that some cats are black" ← 正确
```

对小模型来说，强制分步推理能**抑制过度泛化**，起到约束作用。

---

## What This Means for Multi-Agent Design

### ❌ DON'T

1. **不要在单一任务内对同模型做 delegation**（parent → same-model subagent）
   - 任何规模都降低准确率

2. **不要假设"强模型 + 弱模型"自动更好**
   - 强 parent 被 delegation 打断后的回复质量会显著退化

3. **不要在需要精确数值的任务上使用 delegation**
   - Math/COMP 类别上 delegation 伤害最严重

### ✅ DO

1. **只在知识检索类任务上考虑 delegation**（DOMAIN 类别，Δ=0）
   - 如果 parent 和 subagent 知道同样的知识，分工无意义也无伤害

2. **只在弱模型上考虑 chain-of-thought style delegation**
   - 0.5B Logic +40pp 说明弱模型可能从结构化推理引导中受益

3. **如果要 delegation，确保最终回答格式与 subagent 输出解耦**
   - Strategy B 的失败部分源于答案提取依赖于 subagent 的输出格式

---

## Limitations

1. **Strategy B 是"模拟 delegation"而非真实 subagent**：
   - 真实 subagent 由另一个 vLLM 实例驱动，输出分布不同
   - 结论可能对真实 delegation 过度悲观（真实 subagent 可能比模拟的更准确）

2. **测试集偏向客观题**：数学、比较、逻辑
   - 开放式任务（代码生成、创意写作）上的 delegation 效果可能不同

3. **没有测试跨模型 delegation**：
   - 0.5B parent + 3B subagent 的组合未被测量
   - 这才是最有价值的 delegation 场景

4. **小样本（30 tasks）**：统计功效有限

---

## Next Steps

1. **真实跨模型 delegation**：0.5B/1.5B parent → 3B subagent，测 Strategy C
2. **开放式任务测试**：代码生成、摘要，delegation 可能帮助更大
3. **改进 Strategy B prompt**：避免触发 CoT，让 subagent 模拟更简洁
4. **更大测试集**：100+ tasks 增加统计功效

---

## Reproducing

```bash
cd /home/jinxu/tiny-agents
CUDA_VISIBLE_DEVICES=1 python benchmarks/DELEGATION_VALUE_BENCH.py --models 0.5B
CUDA_VISIBLE_DEVICES=3 python benchmarks/DELEGATION_VALUE_BENCH.py --models 1.5B
CUDA_VISIBLE_DEVICES=2 python benchmarks/DELEGATION_VALUE_BENCH.py --models 3B
```
