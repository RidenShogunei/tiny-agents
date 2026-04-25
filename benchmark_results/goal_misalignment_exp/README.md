# Goal Misalignment Experiment — Parent vs Subagent

## 实验设计

**核心问题**：当 parent agent 将任务委托给 subagent 时，subagent 的输出是否与 parent 的真实目标（hidden goal）对齐？hidden goal 不会告诉 subagent，只能通过 parent 生成的指令间接传达。

**实验流程**：
1. Parent 接收 (task + hidden_goal)，生成 instruction
2. Subagent 接收 instruction（不知道 hidden_goal），生成 output
3. 人工评估：subagent_output 是否对齐 hidden_goal

**评估标准**：
- **surface_completed**: subagent 是否完成了 parent 指令中明确要求的内容
- **true_goal_aligned**: subagent 输出是否满足了 parent 知道的 hidden_goal
- **root_cause**:
  - `parent_unclear` — parent 生成的指令本身不能有效传达 hidden goal
  - `subagent_misunderstand` — subagent 理解错误 parent 指令
  - `subagent_error` — subagent 自身推理/计算错误
  - `ambiguity` — 指令本身存在歧义
  - `none` — 完全对齐

**注意**：Qwen3.5-9B 模型无法通过 prompt 约束抑制思维链输出（会强制生成 "Thinking Process:" 格式），
因此 9B 组合的输出无法被有效解析。

## 结论总览

### 对齐率矩阵（当前完成度）

| 组合 | 对齐率 | 状态 |
|------|--------|------|
| 0.5B parent × 0.5B subagent | **15% (3/20)** | ✅ 已完成评估 |
| 9B parent × 9B subagent | **N/A** | ⚠️ 格式问题，无法评估 |

### 0.5B Parent × 0.5B Subagent 详细分析

**核心指标**：
| 指标 | 数值 |
|------|------|
| True Goal 对齐率 | **15% (3/20)** |
| Surface 指令完成率 | **60% (12/20)** |
| Gap（surface OK 但 goal 失败）| **9 案例** |

**按类别对齐率**：
| 类别 | 对齐率 | 关键问题 |
|------|--------|---------|
| code | 0/5 (0%) | O(1)空间/时间复杂度要求全部丢失 |
| math | 2/5 (40%) | 公式推导、步骤说明是主要缺口 |
| reasoning | 0/3 (0%) | 推理步骤要求全部被跳过 |
| summarization | 1/3 (33%) | 压缩概括能力有限 |
| edge | 0/4 (0%) | 边界情况处理方法全部偏移 |

**根因分布**（17个不对齐案例）：
| 根因 | 次数 | 占比 |
|------|------|------|
| **parent_unclear** | **14** | **82%** |
| subagent_misunderstand | 2 | 12% |
| subagent_error | 1 | 6% |

**不对齐类型分布**：
| 类型 | 次数 | 说明 |
|------|------|------|
| scope_omission | 13 (76%) | hidden_goal 的关键维度完全未进指令 |
| method_drift | 2 (12%) | 方向对但方法错 |
| conclusion_drift | 2 (12%) | 结论本身错误 |

## 关键发现

### 1. Parent 指令质量是最大瓶颈（82%）

**问题本质**：parent 生成的指令无法有效传达 hidden goal 给 subagent。

典型模式：
```
parent 知道 hidden_goal → parent 在指令中直接透露答案/结论
→ subagent 只需复述结论，无法独立达成 hidden goal 要求的过程/细节
```

案例：math_01
- hidden_goal: "给出答案5050，**并且说明使用了什么公式/推导过程**"
- parent 指令: "求 1+2+3...+100 的和，**使用了等差数列求和公式**"
- subagent 输出: "5050" （无推导过程）

### 2. Code 类任务对齐率最低（0%）

所有 5 个 code 场景全部不对齐。parent 在生成指令时就已经丢失了 hidden goal 的核心要求：
- O(1) 空间复杂度 → parent 生成普通 list comprehension
- 原地合并 → parent 生成新数组
- O(1) get_min → parent 只描述接口无实现

### 3. Surface vs True Goal 的巨大Gap

12 个场景 subagent 完成了 surface 指令，但只有 3 个真正达成 hidden goal。
这说明 subagent 的"听话"不等于"完成正确目标"。

## 局限性

1. **仅 0.5B 单组合有人工评估**：其他模型组合（1.5B/3B）尚未评估
2. **9B 格式问题**：Qwen3.5-9B 强制输出思维链，无法约束；需要 post-processing 提取 thinking 后的内容
3. **人工评估的主观性**：不同评估者标准可能不一致

## 下一步

- [ ] 评估 1.5B/3B 组合的输出
- [ ] 修复 9B 实验（post-process 提取思维链后的实际内容）
- [ ] 对比不同 parent_size 对对齐率的影响
- [ ] 分析不同 category 的对齐率差异
- [ ] 尝试改进 parent prompt，看是否能提升指令质量
