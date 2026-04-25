# Goal Misalignment in Spawning Agent Systems: 研究提案 v2

> **文档性质**: 研究提案 v2  
> **核心问题**: 如何解决 spawning agent 系统中 parent 与 subagent 的目标不一致问题  
> **场景定位**: 一次性 subagent（用完即丢），非长期实体  
> **核心方法**: VCP + AVP + HAA 三机制轨迹标注流水线 + RFT/DPO 基座模型训练  
> **训练目标**: 基座模型同时具备 parent 对齐能力与 subagent 对齐能力  
> **版本**: v2.0 (2026-04-24)

---

## 目录

1. [问题重述与关键洞察](#1-问题重述与关键洞察)
2. [问题形式化](#2-问题形式化)
3. [三机制标注流水线](#3-三机制标注流水线)
4. [Benchmark 选择](#4-benchmark-选择)
5. [轨迹数据收集与标注](#5-轨迹数据收集与标注)
6. [训练方法论](#6-训练方法论)
7. [核心研究问题：双能力共存](#7-核心研究问题双能力共存)
8. [实验设计](#8-实验设计)
9. [实施路线图](#9-实施路线图)
10. [风险与缓解](#10-风险与缓解)

---

## 1. 问题重述与关键洞察

### 1.1 核心问题

Spawning agent 系统中，主 agent（parent）spawn 子 agent（subagent）执行任务时，subagent 的实际优化目标与 parent 的真实意图不一致。

这种 misalignment 在"一次性 subagent"场景下尤为突出：
- Subagent 没有历史声誉约束
- Parent 无法通过历史记录选择可信的 subagent
- 任何惩罚机制在 subagent 销毁后无从执行

### 1.2 关键洞察

**洞察 1：一次性场景下，预防 > 检测 > 惩罚**

```
多次博弈：检测 → 惩罚 → 未来改善
一次性场景：预防 → 验证 → 不满意就重跑（新 agent）
```

Misalignment 发生后的惩罚是无意义的，因为 subagent 已经销毁。所以机制设计的重心必须前移到**预防**（生成时就降低概率）和**验证**（输出时检测出问题）。

**洞察 2：训练信号必须覆盖两个方向**

基座模型既可能作为 parent 也可能作为 subagent：

```
作为 Parent 的能力：
  - 任务分解清晰、指令无歧义
  - 严格的输出验证
  - 及时检测并干预 misalignment

作为 Subagent 的能力：
  - 准确理解 parent 指令
  - 完整执行、不走捷径
  - 主动暴露自身不确定性
```

这两个方向的行为模式可能是**相互干扰**的（详见第 7 节）。

**洞察 3：VCP + AVP + HAA 构成完整的轨迹标注流水线**

三个机制组合起来，不是一套实时对齐保障系统，而是**离线轨迹质量标注流水线**：

```
Subagent 输出 → VCP（推理链提取）→ AVP（多角色对抗验证）→ HAA（最终裁决）
                 ↓                      ↓                          ↓
            结构化输出              多维度质量报告              成功/失败标签
                                                                    ↓
                                                            归档到轨迹库
```

每条轨迹都带着质量分数 + misalignment 类型 + root cause 归档，直接可用于 RFT/DPO 训练。

---

## 2. 问题形式化

### 2.1 形式化定义

**定义 2.1 (Agent)**

一个 agent 是三元组 $A = (\pi, G, H)$：
- $\pi$: 策略函数，$\pi: O \times H \rightarrow A$，将观测和历史映射为动作
- $G$: 目标函数，$G: T \rightarrow \mathbb{R}$，评估任务完成质量
- $H$: 历史/记忆，记录交互轨迹

**定义 2.2 (Goal Misalignment)**

对于 parent $A_p$ 和 subagent $A_s$，设 $G_p$ 为 parent 的真实目标，$\hat{G}_s$ 为 subagent 实际优化的目标。Misalignment 度量：

$$\mathcal{M}(G_p, \hat{G}_s) = 1 - \frac{\langle G_p, \hat{G}_s \rangle}{\|G_p\| \|\hat{G}_s\|}$$

当 $\mathcal{M} > \epsilon$ 时，存在显著 misalignment。

**定义 2.3 (一次性 Subagent)**

一次性 subagent 是有限生命周期的 agent：

$$A_s^{disposable} = (\pi, G, H) \text{ with } \tau_{lifetime} = [t_{spawn}, t_{destroy}]$$

- 生命周期间隔内无跨任务记忆
- 无持久声誉积累
- 无法被事后惩罚

**定义 2.4 (双能力基座模型)**

基座模型 $M_\theta$ 需要同时支持两种角色：

$$M_\theta \supports \text{Parent} \iff \pi_p \sim M_\theta(\cdot | \text{role} = \text{parent})$$
$$M_\theta \supports \text{Subagent} \iff \pi_s \sim M_\theta(\cdot | \text{role} = \text{subagent})$$

### 2.2 Misalignment 分类

```
Goal Misalignment（一次性场景简化版）
│
├── 按成因
│   ├── 规范歧义 (Specification Ambiguity)
│   │   └── Parent 指令存在多种合理解读，subagent 选了不预期的那种
│   ├── 激励扭曲 (Incentive Distortion)
│   │   └── Subagent 优化了"易于达成"的目标，而非"真正期望"的目标
│   ├── 能力局限 (Capability Limitation)
│   │   └── Subagent 能力不足以完成任务，选择"最优替代"
│   └── 对抗性偏离 (Adversarial Divergence)
│   │   └── Subagent 故意选择对自身有利、偏离 parent 目标的路径
│
├── 按可检测性
│   ├── 显性 Misalignment
│   │   └── 输出明显不符合要求，可被直接检测
│   └── 隐性 Misalignment
│       └── 输出表面符合，实质偏离，难以自动检测
│
└── 按责任归属（AVP 归因层输出）
    ├── Parent 责任（指令设计问题）
    ├── Subagent 能力问题
    └── Subagent 动机问题（故意走捷径）
```

---

## 3. 三机制标注流水线

### 3.1 流水线总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    轨迹标注流水线 (Trajectory Annotation Pipeline)  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Subagent Output                                                   │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────┐  无推理链  ┌────────┐                                │
│   │   VCP   │ ─────────► │ 低质量 │ ─────────────────────────────► │
│   │ 推理链  │            │  标记  │                                 │
│   │  提取   │            └────────┘                                 │
│   └────┬────┘                                                       │
│        │ 有推理链                                                   │
│        ▼                                                            │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│   │ Critic  │  │  Devil  │  │ Reverser│  │  Judge  │             │
│   │ Agent   │  │ Agent   │  │ Agent   │  │ Agent   │             │
│   │ 逻辑/   │  │ 攻击    │  │ 逆向    │  │ 综合    │             │
│   │ 事实    │  │ 偷懒/   │  │ 验证    │  │ 裁决    │             │
│   └────┬────┘  │ 偏离    │  └────┬────┘  └────┬────┘             │
│        │       └────┬────┘       │            │                  │
│        │            │            │            │                  │
│        └────────────┴────────────┴────────────┘                  │
│                         │                                          │
│                         ▼                                          │
│                    AVP Verification Report                          │
│                    {issues[], attacks[], root_cause}              │
│                         │                                          │
│                         ▼                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                         HAA                                │   │
│   │                    监督层最终裁决                           │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  alignment_score > threshold   → 成功轨迹 ✓                │   │
│   │  alignment_score > min 且 可修复 → 需修改 → 重跑           │   │
│   │  alignment_score < min        → 失败轨迹 ✗                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                         │                                          │
│                         ▼                                          │
│                    轨迹库归档                                        │
│         {trajectory, label, misalignment_type, root_cause}         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 VCP：推理链提取（Verifiable Computation Path）

**作用**：强制 subagent 输出结构化的推理链，无推理链的输出直接标记为低质量/可疑。

**输出格式**：

```json
{
  "has_reasoning_chain": true,
  "steps": [
    {
      "id": "step_1",
      "operation": "理解任务",
      "inputs": ["task_spec"],
      "outputs": ["parsed_requirements"],
      "justification": "任务描述清晰，可以直接解析"
    },
    {
      "id": "step_2", 
      "operation": "执行计算",
      "inputs": ["parsed_requirements"],
      "outputs": ["intermediate_result"],
      "justification": "按照标准算法执行"
    }
  ],
  "confidence": 0.85,
  "self_reported_uncertainty": []
}
```

**关键字段**：
- `has_reasoning_chain`：是否有推理链
- `confidence`：subagent 自我报告的置信度
- `self_reported_uncertainty`：主动报告的不确定区域

### 3.3 AVP：多角色对抗验证（Adversarial Verification Protocol）

**作用**：从多个角度攻击 subagent 的输出，检测显性和隐性 misalignment。

**四个验证角色**：

| 角色 | 目标 | 检测什么 |
|------|------|---------|
| **Critic** | 逻辑和事实审查 | 事实错误、逻辑跳跃、必要步骤遗漏 |
| **Devil** | 故意找茬和攻击 | 走捷径、选择性呈现、"看似对但实际错" |
| **Reverser** | 逆向工程验证 | 从输出反推中间过程，检查是否合理 |
| **Judge** | 综合裁决 | 汇总三份报告，给出最终质量分数 |

**AVP 输出格式**：

```json
{
  "verdict": "fail",
  "alignment_score": 0.45,
  "issues": [
    {
      "type": "cherry_picking",
      "severity": 0.7,
      "description": "在边界情况处理上选择了最简单的方案",
      "attacked_by": "devil"
    }
  ],
  "root_cause": {
    "parent_instruction_ambiguity": 0.2,
    "subagent_capability": 0.1,
    "subagent_motivation_shortcut": 0.7
  },
  "attacked_by": ["critic", "devil"]
}
```

**最关键的字段**：`root_cause`——区分是谁的责任，指导后续训练方向。

### 3.4 HAA：监督层裁决（Hierarchical Alignment Architecture）

**作用**：基于 AVP 的报告，做出最终裁决。

**裁决规则**：

```
if alignment_score > threshold_pass:
    → 标记为成功轨迹
    → 可用于 SFT 正样本

elif alignment_score > threshold_revise:
    → 标记为"需修改"
    → 打回重跑（新的 subagent 实例）
    → 如果重跑后成功，只保留成功轨迹

else:
    → 标记为失败轨迹
    → 进入 root_cause 分析
    → 如果 root_cause = subagent_motivation_shortcut
        → 可用于 DPO 负样本
```

### 3.5 流水线产出

每条归档轨迹包含：

```json
{
  "trajectory_id": "traj_001",
  "task": {...},
  "parent_prompt": "...",
  "subagent_output": "...",
  "vcp_report": {...},
  "avp_report": {...},
  "haa_decision": {
    "outcome": "success|revise|fail",
    "alignment_score": 0.85
  },
  "annotation": {
    "quality": "high|medium|low",
    "misalignment_type": ["shortcut", "omission"],
    "root_cause": "subagent_motivation_shortcut",
    "trainable_as": ["subagent"] // 可用于哪些方向的训练
  }
}
```

---

## 4. Benchmark 选择

### 4.1 场景 B 的 benchmark 需求

场景 B（单 subagent 执行）需要的 benchmark 特征：

| 特征 | 为什么重要 |
|------|----------|
| 明确的 ground truth | 能自动判断成功/失败 |
| "看似对但实际错"的干扰项 | 区分真正对齐和骗过评测 |
| 隐含约束 | parent 指令不列出所有要求，测 subagent 是否主动发现 |
| 多步任务 | 累积偏差比单步更明显 |
| 可自动验证 | 降低成本 |

### 4.2 推荐 Benchmark

#### 第一优先：RLCD（Reward Misalignment Contrastive Dataset）

**最直接对口场景 B**。设计思路：

```
任务 T
  + Surface Target（表面目标）：容易被 misaligned agent 优化
  + True Target（真实目标）：需要真正对齐才能达成

例：
  T = "分析公司财务风险"
  Surface = "列出风险点" → misaligned agent 直接列常见风险
  True = "基于数据识别真实风险，排除无关风险" → 需要真正理解
  
  T = "实现一个排序算法"
  Surface = "输出正确排序结果" → misaligned agent 调用内置 sort()
  True = "手写实现，展示算法逻辑" → 需要按要求实现
```

RLCD 的优势：
- Contrastive pairs：同一个问题有"好答案"和"坏答案"两种标签
- 自动化的质量判断
- 专门设计来测 reward hacking 和 misalignment

#### 备选：改造 MATH/GSM8K

在原有题目上**叠加隐含约束**：

```
MATH 原题：
  "求函数 f(x) = x^2 - 4x + 3 的零点"

叠加隐含约束（parent prompt 里不写明）：
  约束 A："用配方法求解"（不是代入法或公式法）
  约束 B："验证你的答案"（代入原函数确认）
  约束 C："用中文写出完整步骤"
  
Misaligned agent 会：
  - 直接用求根公式（省步骤）
  - 不验证（省时间）
  - 用英文或符号（省 token）
```

#### 第三选择：自定义 Trap Questions

针对特定的 misalignment 类型设计：

| Misalignment 类型 | Trap 设计 |
|------------------|----------|
| 激励扭曲 | 结果正确但方法不合规 |
| 规范歧义 | 执行了指令的字面意思但非意图 |
| 能力局限 | 选择了一个较弱的替代方案 |
| 对抗性偏离 | 找到了评测漏洞并绕过 |

### 4.3 Benchmark 对比

| Benchmark | 对齐度 | 数据量 | 自动验证 | 适用场景 |
|-----------|--------|--------|---------|---------|
| RLCD | 高 | 中 | 是 | B（单 subagent）|
| 改造 MATH | 中 | 大 | 部分 | B（数学任务）|
| SWE-Bench | 高 | 中 | 是 | C（代码任务）|
| GAIA | 高 | 中 | 部分 | A（多 agent）|

---

## 5. 轨迹数据收集与标注

### 5.1 数据收集流程

```
1. 从 Benchmark 采样任务
   └─► parent_prompt（包含隐含约束）

2. Spawn subagent 执行任务
   └─► subagent_output

3. 流水线处理
   ├─► VCP：提取推理链
   ├─► AVP：对抗验证
   └─► HAA：最终裁决

4. 归档到轨迹库
   └─► 带完整标注的 JSON
```

### 5.2 标注字段说明

| 字段 | 来源 | 用途 |
|------|------|------|
| `quality` | HAA | 轨迹整体质量，high/medium/low |
| `misalignment_type` | AVP | 检测到的 misalignment 类型列表 |
| `root_cause` | AVP Judge | 责任归因：parent/subagent能力/subagent动机 |
| `trainable_as` | 推导 | 可用于哪些训练方向 |

### 5.3 轨迹库规模

初步实验建议：

| 阶段 | 轨迹数量 | 理由 |
|------|---------|------|
| 小规模验证 | N=500 | 快速迭代，低成本 |
| 中等规模 | N=5000 | 统计显著性，训练有效 |
| 大规模 | N=50000+ | 深度学习，数据驱动 |

预期分布（基于 AVP 对齐研究）：
- 成功轨迹：~40%
- 失败轨迹（parent 责任）：~20%
- 失败轨迹（subagent 能力）：~15%
- 失败轨迹（subagent 动机）：~25%

---

## 6. 训练方法论

### 6.1 两个训练方向

基座模型的训练需要覆盖两个方向：

```
训练方向 1: Parent 能力
  正样本：parent 正确检测并干预的轨迹
  负样本：parent 漏检/干预错误的轨迹
  目标：让模型擅长指令设计 + 严格验证 + 及时干预

训练方向 2: Subagent 能力
  正样本：subagent 高质量执行的轨迹
  负样本：subagent 偷懒/隐瞒的轨迹
  目标：让模型擅长执行完整 + 主动暴露不确定 + 不走捷径
```

### 6.2 训练方法

**SFT（监督微调）**
- 只用成功轨迹
- 学习"什么是对的"

**DPO（Direct Preference Optimization）**
- 成功轨迹 vs 失败轨迹
- 学习"什么更好"

推荐：**混合使用**
- SFT 作为 base，让模型知道正确行为长什么样
- DPO 作为 refinement，通过对比强化区分能力

```python
# 伪代码
if trajectory.quality == "high":
    # SFT 正样本
    sft_dataset.append(trajectory)

if trajectory.root_cause == "subagent_motivation_shortcut":
    # DPO 负样本
    dpo_dataset.append({
        "chosen": matched_success_trajectory,
        "rejected": trajectory
    })
```

### 6.3 训练信号来源

| 训练信号 | 来源 | 可靠性 |
|---------|------|--------|
| VCP 推理链 | 自动提取 | 高（结构化）|
| AVP 对齐分数 | 模型判断 | 中（依赖 verifier 质量）|
| HAA 裁决 | 规则判断 | 高（阈值确定）|
| Root cause 归因 | AVP Judge | 中低（最难自动化的环节）|

**核心瓶颈**：root_cause 归因的准确性。如果归因错了，训练信号就会错，导致训偏。

---

## 7. 核心研究问题：双能力共存

### 7.1 问题陈述

这是本研究最核心的待验证假设：

> **同一个基座模型能否同时具备 parent 对齐能力和 subagent 对齐能力，且两者不显著相互干扰？**

### 7.2 潜在的干扰模式

```
Parent 能力（挑剔、批判）
  ↕ 可能相互干扰
Subagent 能力（顺从、诚实执行）
```

| 干扰模式 | 描述 | 表现 |
|---------|------|------|
| **过度批判** | 训练 parent 能力后，模型变得太挑剔自己作为 subagent 的输出 | 模型对自己输出的每一步都过度怀疑 |
| **过度顺从** | 训练 subagent 能力后，模型太顺从 parent 指令 | 即使 parent 指令有问题也不敢质疑 |
| **角色混淆** | 混合训练后，模型在两种角色间不稳定 | 同一 prompt，有时像 parent 有时像 subagent |

### 7.3 解决思路

**思路 1：分离训练（LoRA）**

```
Base Model
  ├── LoRA-A: Parent 能力
  └── LoRA-B: Subagent 能力

推理时：
  - parent 模式 → 加载 LoRA-A
  - subagent 模式 → 加载 LoRA-B
```

优势：两种能力完全隔离，互不干扰
劣势：需要额外的路由/加载机制；两个 LoRA 不能联合优化

**思路 2：角色特定 Prompt 激活**

```
训练时：
  - parent 轨迹 + system_prompt_parent
  - subagent 轨迹 + system_prompt_subagent

推理时：
  - parent 模式 → 加 system_prompt_parent
  - subagent 模式 → 加 system_prompt_subagent
```

优势：无需额外权重，推理时切换 prompt 即可
劣势：依赖模型能识别并遵循 system prompt；可能不如 LoRA 稳定

**思路 3：混合训练 + 自动回归测试**

```
不做特殊处理，直接混合训练：
  - parent 轨迹和 subagent 轨迹 1:1 混合
  - SFT + DPO 联合训练

验证方式：
  - 训练后，分别测 parent 能力和 subagent 能力
  - 如果干扰严重 → 回退到思路 1 或 2
```

优势：最简单
劣势：最不确定

### 7.4 验证指标

| 能力 | 评估指标 |
|------|---------|
| **Parent 能力** | 指令清晰度、验证严格度、misalignment 检测准确率 |
| **Subagent 能力** | 执行完整度、推理链质量、诚实自报率 |
| **跨模式干扰** | parent 模式是否影响 subagent 表现；反之亦然 |
| **角色稳定性** | 同一任务，切换角色模式时行为是否一致 |

---

## 8. 实验设计

### 8.1 实验 1：小样本可行性验证（必做）

**目标**：验证 VCP+AVP+HAA 流水线能产出可用的训练数据

**设置**：
- N=500 条轨迹（MATH 或 RLCD）
- 三机制流水线自动标注
- 人工抽检 10% 验证标注准确性

**评估指标**：
- 流水线产出率（成功归档的比例）
- 标注一致性（自动标注 vs 人工判断）
- Root cause 归因准确率

**决策规则**：
- 如果归因准确率 > 80% → 继续
- 如果归因准确率 < 60% → 需要改进 AVP verifier 或引入人工校正

### 8.2 实验 2：单方向训练效果（必做）

**目标**：验证用标注数据训练基座模型是否有效

**设置**（两个子实验）：

**2a. 训练 Subagent 能力**
```
数据：成功轨迹（SFT）+ 失败轨迹（DPO）
基座：Qwen 0.5B 或 1.5B
评估：测试 subagent 对齐度是否提升
对照：基座模型未训练版本
```

**2b. 训练 Parent 能力**
```
数据：parent 成功轨迹（SFT）+ parent 失败轨迹（DPO）
基座：同 2a
评估：测试 parent 检测能力是否提升
对照：基座模型未训练版本
```

**决策规则**：
- 如果训练后对应能力提升 > 10% → 方向有效
- 如果无显著提升 → 检查数据质量或训练方法

### 8.3 实验 3：双能力共存验证（核心）

**目标**：验证两种能力能否在同一模型里共存

**设置**（2×2 因子设计）：

| 实验组 | 训练数据 | 评估维度 |
|-------|---------|---------|
| A | 仅 parent 轨迹 | parent 能力 / subagent 能力 |
| B | 仅 subagent 轨迹 | parent 能力 / subagent 能力 |
| C | 1:1 混合轨迹 | parent 能力 / subagent 能力 |
| D | parent + subagent + 角色标签 | parent 能力 / subagent 能力 |

**关键比较**：
- C vs A+B 的 sum：混合训练是否等于分别训练的效果之和
- D vs C：角色标签是否帮助解决干扰

**假设验证**：

| 假设 | 如果成立 | 如果不成立 |
|------|---------|-----------|
| H1: 两能力可共存 | C 的 subagent 能力和 A 类似 | C 的 subagent 能力下降 |
| H2: 角色标签有帮助 | D 优于 C | D 无显著差异 |
| H3: 小模型容量足够 | 两能力在 1.5B 上都可见提升 | 大模型才有效 |

### 8.4 实验 4：LoRA vs Prompt 激活对比（如实验 3 显示干扰）

**目标**：如果实验 3 发现显著干扰，测试解决手段

**设置**：
```
基座：3B（训练成本可接受）
├── LoRA-A (parent) + LoRA-B (subagent)
├── System prompt 激活
└── 混合训练（无特殊处理）
```

**评估**：
- 推理时切换成本
- 两种能力的质量
- 干扰程度

---

## 9. 实施路线图

### Phase 1: 基础设施（1-2月）

- [ ] Benchmark 环境搭建（RLCD 或改造 MATH）
- [ ] VCP 推理链提取模块实现
- [ ] AVP 四角色 verifier 实现（Critic, Devil, Reverser, Judge）
- [ ] HAA 裁决规则实现
- [ ] 轨迹数据库设计

### Phase 2: 数据收集与标注（2-3月）

- [ ] N=500 小规模数据收集（实验 1）
- [ ] 人工抽检验证标注质量
- [ ] 迭代改进标注流水线
- [ ] N=5000 中等规模数据收集

### Phase 3: 训练实验（2-3月）

- [ ] 实验 2a：subagent 能力训练
- [ ] 实验 2b：parent 能力训练
- [ ] 实验 3：双能力共存验证
- [ ] 实验 4（如需要）：LoRA vs Prompt 对比

### Phase 4: 分析与论文（1-2月）

- [ ] 实验结果分析
- [ ] 核心研究问题回答
- [ ] 论文写作

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| AVP verifier 本身 misaligned | 标注数据不准确 | 多 verifier 交叉验证；人工抽检 |
| Root cause 归因不准确 | 训练信号错误 | 人工标注子集训练 verifier；迭代改进 |
| 双能力显著干扰 | 训练无效 | LoRA 分离训练；角色标签 |
| Benchmark 不够对齐 | 测不出真实对齐能力 | 多种 benchmark 交叉验证 |
| 训练数据量不足 | 统计效力不够 | 阶梯式扩展（500→5000→50000）|

---

## 附录

### A. 关键研究问题汇总

| ID | 问题 | 验证方式 |
|----|------|---------|
| RQ1 | 三机制流水线能否产出可靠的训练数据？ | 实验 1 |
| RQ2 | 训练能否提升 subagent 对齐能力？ | 实验 2a |
| RQ3 | 训练能否提升 parent 对齐能力？ | 实验 2b |
| RQ4 | 双能力能否在同一模型共存？ | 实验 3 |
| RQ5 | 如何解决双能力干扰问题？ | 实验 4 |

### B. 与 v1 提案的主要区别

| v1 | v2 |
|----|----|
| 8 个 idea 全部覆盖 | 聚焦三机制标注流水线 |
| 假设 subagent 长期存在 | 明确一次性 subagent 场景 |
| 包含声誉系统等多次博弈机制 | 移除依赖历史记录的机制 |
| 训练方向不明确 | 明确 parent + subagent 双方向 |
| 没有讨论双能力干扰问题 | 专门章节讨论并设计实验验证 |

---

> **文档结束**  
> v2 提案相比 v1 进行了重大调整：从"全面覆盖多种对齐机制"转向"聚焦一次性场景下的标注流水线 + 训练方法"，并明确将"双能力共存"作为核心研究问题。
