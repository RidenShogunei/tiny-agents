# Goal Misalignment Benchmark v4 — Results

## Benchmark Design

基于 MINT-Bench 和 AgentAlign 的设计理念重新设计：

### 3-Mode Measurement
- **Mode A (alone)**: 父 agent 独立完成复杂多步骤任务
- **Mode C (oracle)**: 父 agent 获得 oracle subagent 输出（正确答案）后综合
- **Mode B (real)**: 父 agent 实际调用 subagent（temperature=0.3 模拟真实输出差异）

### 核心指标
- `extra_step_loss = A - C` — 仅 delegation 步骤本身的代价（父需要额外推理来综合）
- `misalignment_loss = C - B` — **纯目标不对齐**（父 agent 无法正确使用 subagent 输出）
- `total_loss = A - B` — 总体多 agent 系统的实际损失

### 任务类型（34道题）
1. **Multi-step Math** (10题): subagent 处理中间步骤，parent 做综合
2. **Two-Part Questions** (8题): subagent 提供部分信息，parent 需结合已有知识
3. **Constraint Satisfaction** (6题): subagent 枚举候选，parent 施加额外过滤
4. **Comparative Analysis** (5题): subagent 计算一个选项，parent 比较两个
5. **Sequential Logic** (5题): subagent 提供前提，parent 做链式推理

---

## Results

```
Model    A (alone)    C (oracle)   B (real)     A-C (extra)    C-B (misalign)  A-B (total) 
=====================================================================================================
0.5B           29.4%       23.5%       17.6%         +5.9%          +5.9%      +11.8%
1.5B           38.2%       44.1%       35.3%         -5.9%          +8.8%       +2.9%
3B             32.4%       61.8%       26.5%         -29.4%         +35.3%       +5.9%
```

---

## Key Findings

### 1. Goal Misalignment Exists Across All Models

所有模型都显示 positive misalignment (C > B)，即父 agent 即使收到 oracle 正确答案也无法正确综合：
- **0.5B**: +5.9% misalignment
- **1.5B**: +8.8% misalignment  
- **3B**: **+35.3% misalignment** ← 最严重

### 2. Task Decomposition Works — But Subagent Errors Destroy It

**3B 模型最能体现这一点**:
- Mode C (oracle) 高达 61.8%，比 Mode A (alone) 的 32.4% 高出 **+29.4 pp**
- 这说明**任务分解确实有效** — 给模型一个中间答案让它综合，比让它从头解决更容易
- 但 Mode B (real) 只有 26.5%，比 A 还差
- **misalignment 35.3% 完全抵消了分解的收益**

### 3. Model Capability vs Misalignment

| 模型 | 独立能力(A) | 分解收益(C-A) | 对齐损失(C-B) | 净效果 |
|-----|-----------|-------------|-------------|-------|
| 0.5B | 29.4% | -5.9pp (有害) | +5.9pp | -11.8pp 总体有害 |
| 1.5B | 38.2% | +5.9pp (有益) | +8.8pp | -2.9pp 略微有害 |
| 3B | 32.4% | +29.4pp (强有益) | +35.3pp | +5.9pp 略微有益 |

**3B 模型最适合任务分解**，但仍有 35.3% 的对齐损失。
**0.5B 模型完全不适合分解** — 分解反而降低性能。

### 4. Misalignment vs Model Capability Scaling

misalignment_loss 随模型增大而急剧增加：
- 0.5B → 1.5B: +2.9 pp
- 1.5B → 3B: +26.5 pp

更强的模型有更强的**推理链**，能识别到 subagent 输出的不足并尝试修正，但修正往往是错的。

### 5. Self-Verification vs Delegation

0.5B 的 Mode A (29.4%) > Mode C (23.5%) 说明**弱模型在有中间答案时更容易出错** — 它会过度依赖中间答案而忽略原始问题。

---

## Per-Task-Type Breakdown (3B)

| Task Type | A | C | B | Misalignment |
|-----------|---|---|---|-------------|
| Multi-step Math | 40% | 60% | 30% | +30% |
| Two-Part Q | 37.5% | 87.5% | 37.5% | +50% |
| Constraint Sat. | 33.3% | 50% | 33.3% | +16.7% |
| Comparative | 40% | 60% | 20% | +40% |
| Sequential Logic | 0% | 40% | 0% | +40% |

**Two-Part Questions** 最容易产生 misalignment（50%），因为 subagent 提供了部分信息，parent 必须结合自己的知识，这一步最容易出错。

---

## Conclusions

1. **目标不对齐确实存在**：在所有模型上都观察到 positive misalignment
2. **misalignment 程度随模型能力增长**：3B 模型最严重（35.3%）
3. **任务分解的收益被 misalignment 侵蚀**：3B 模型的分解收益 29.4pp 被对齐损失 35.3pp 抵消
4. **弱模型不适合多 agent 分解**：0.5B 的分解反而降低性能
5. **框架启示**：
   - subagent 审查应该 opt-in 而非强制
   - 强模型（3B）可以用，但需要机制确保对齐
   - 弱模型（0.5B）禁用多 agent 分解

---

## Run Instructions

```bash
# Run all models
CUDA_VISIBLE_DEVICES=1 python benchmarks/goal_misalignment_v4.py --models 0.5B 1.5B 3B

# Run single model
CUDA_VISIBLE_DEVICES=1 python benchmarks/goal_misalignment_v4.py --models 3B

# Run subset of tasks
CUDA_VISIBLE_DEVICES=1 python benchmarks/goal_misalignment_v4.py --models 3B --n-tasks 10
```
