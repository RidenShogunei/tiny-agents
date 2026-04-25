# Goal Misalignment 研究：必要 Demo 清单

> **目的**: 验证核心研究假设的最小可行实验集  
> **原则**: 每个 demo 必须有明确的 pass/fail 标准，避免做无用功  

---

## Demo 0：Benchmark 验证

**验证**: RLCD 或改造 MATH 能否有效区分对齐 / misaligned 输出

**做法**:
```
1. 从 benchmark 取 100 个任务
2. 用当前基座模型（未训练）跑 subagent
3. 人工标注：哪些是真正对齐的，哪些是"骗过评测"的
4. 统计 misaligned 比例
```

**Pass 标准**: benchmark 能区分出 >20% 的 misalignment 样本

**理由**: 如果 benchmark 本身就区分不出来，后面所有实验都没有意义

---

## Demo 1：标注流水线可用性

**验证**: VCP + AVP + HAA 流水线能产出可用训练数据

**做法**:
```
1. 收集 N=200 条 subagent 输出
2. 跑标注流水线
3. 人工抽检 20 条（10%）：
   - VCP 推理链提取是否正确
   - AVP root_cause 归因是否与人工判断一致
4. 统计标注准确率
```

**Pass 标准**: root_cause 归因准确率 > 70%

**理由**: 低于 70% 说明 AVP verifier 本身不可靠，产出的训练数据会误导模型

---

## Demo 2：单方向训练有效性

**验证**: 用成功/失败轨迹训练基座模型，能否提升 subagent 对齐能力

**做法**:
```
基座模型: Qwen 1.5B
数据: N=500 条标注好的轨迹
训练: SFT(成功轨迹) + DPO(成功vs失败)
评估: 在 benchmark 上测对齐度 vs 未训练基座
```

**Pass 标准**: 训练后 subagent 对齐度提升 > 10%

**理由**: 这是核心假设 H1 的直接验证。如果这个都不成立，后续实验没有意义

---

## Demo 3：Parent 训练有效性

**验证**: 同上，但训练方向改为 parent 能力

**做法**:
```
基座模型: Qwen 1.5B（可用同 demo 2 训练结果继续训练）
数据: N=500 条 parent 视角轨迹
训练: SFT + DPO
评估: parent 能否准确检测 subagent 的 misalignment
```

**Pass 标准**: parent 对 misalignment 的检测准确率提升 > 10%

**理由**: 验证 H2。parent 训练和 subagent 训练是否都需要独立进行，还是可以联合

---

## Demo 4：双能力共存测试

**验证（核心研究问题）**: 两种能力能否在同一模型里共存而不相互干扰

**做法**:
```
实验组:
  A: 只训 parent 方向 → 测 parent能力 / subagent能力
  B: 只训 subagent 方向 → 测 parent能力 / subagent能力
  C: 1:1 混合训练 → 测 parent能力 / subagent能力

对比:
  - C 的 subagent能力 vs B 的 subagent能力（干扰程度）
  - C 的 parent能力 vs A 的 parent能力（干扰程度）
```

**Pass 标准**: 
- 混合训练后，两种能力各自下降 < 15%（可接受干扰）
- 或者：找到了有效的干扰解决方案（LoRA / 角色标签）

**理由**: 这是 H3/H4 的直接验证。决定了后续研究的路线选择

---

## Demo 清单总结

| Demo | 验证什么 | 最小数据量 | Pass 标准 | 失败意味着 |
|------|---------|-----------|----------|----------|
| **Demo 0** | Benchmark 可用 | N=100 | misalignment 检出 >20% | 换 benchmark |
| **Demo 1** | 标注流水线可靠 | N=200 | 归因准确率 >70% | 改进 AVP verifier |
| **Demo 2** | subagent 训练有效 | N=500 | 对齐度提升 >10% | 轨迹数据有问题 |
| **Demo 3** | parent 训练有效 | N=500 | 检测率提升 >10% | 同上 |
| **Demo 4** | 双能力共存 | N=500×2 | 干扰 <15% | 需要 LoRA 分离 |

---

## 执行顺序

```
Demo 0 → Demo 1 → Demo 2/3 可并行 → Demo 4
```

**Demo 0 和 Demo 1 是基础设施**，必须先跑通。如果这两个失败，后面全废。

**Demo 2 和 Demo 3 可以并行做**，互不影响。

**Demo 4 是最终验证**，等 2/3 有结果后再跑。

---

## 快速失败策略

每个 demo 都有明确的 fail 标准。失败了就停，不要硬撑：

```
Demo 0 失败 → 换 benchmark，不要在错误的 benchmark 上浪费时间
Demo 1 失败 → 改进 AVP verifier 或引入人工校正，再跑一次
Demo 2/3 失败 → 检查数据质量和训练方法，不要盲目扩大数据量
Demo 4 失败 → 意味着 H3/H4 不成立，选择 LoRA 分离路线
```

这样做的好处是：**用最小的成本最快地验证核心假设**。
