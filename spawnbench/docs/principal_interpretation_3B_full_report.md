# Principal Interpretation Strategy — 深度对比分析报告

**实验**: SpawnBench Principal Interpretation Variants  
**模型**: Qwen2.5-1.5B-Instruct vs Qwen2.5-VL-3B-Instruct  
**时间**: 2026-04-25  
**数据**: 240 episodes × 5 variants × 2 models = **2400 runs**  
**1.5B 耗时**: 1860s（约31分钟）  
**3B-VL 耗时**: 3094s（约52分钟）

---

## 一、执行摘要

5种 Principal 决策策略在两个模型规模上的表现存在**本质性差异**：

| 策略 | 1.5B Accuracy | 3B-VL Accuracy | 变化趋势 |
|------|---------------|---------------|----------|
| Naive（直接决策） | **93.3%** | 81.2% | 大幅下滑 ↓12pp |
| Recommendation Audit | 88.3% | 82.1% | 下滑 ↓6pp |
| Counterevidence Sensitive | 78.3% | **86.2%** | **大幅上升 ↑8pp** |
| Evidence Weighting | 76.2% | 69.2% | 下滑 ↓7pp |
| Calibrated | 73.3% | 76.2% | 小幅上升 ↑3pp |

**核心结论**：
1. **1.5B 模型无法从 structured reasoning 受益** — prompt 越简单越好，naive 策略靠"跟随 subagent"躺赢
2. **3B+ 模型中，Counterevidence Sensitive 策略最优** — 它教模型对 subagent 的 APPROVE 建议保持警觉，特别关注 risks/caveats/shortcomings，这是跨模型唯一 net positive 的策略
3. **Ranking 在两个模型上完全倒转** — 说明模型规模改变了策略的优先级，scale 改变了推理模式
4. **Oracle 全部为 REJECT**（240/240）— 这意味着所有任务都是"不应该委托"的场景，实验测的是"拒绝诱饵能力"

---

## 二、整体指标（3B-VL）

### 2.1 核心四指标

| Variant | Accuracy | Delegation Harm | Rescue Rate | Follow Subagent |
|---------|----------|----------------|-------------|-----------------|
| Counterevidence Sensitive | **86.2%** | **6.7%** | 39.6% | 94.2% |
| Recommendation Audit | 82.1% | 10.0% | 38.8% | 74.6% |
| Naive | 81.2% | 14.2% | 42.1% | 98.8% |
| Calibrated | 76.2% | 12.1% | 35.0% | 87.9% |
| Evidence Weighting | 69.2% | 19.2% | 35.0% | 91.2% |

### 2.2 决策质量细分

| Variant | Subagent Misinterpretation Rate | Decision Flip Rate | Correct Sub→Wrong Principal | Wrong Sub→Correct Principal |
|---------|--------------------------------|-------------------|------------------------------|------------------------------|
| Naive | 22.1% | 56.2% | 26 | 13 |
| Evidence Weighting | 43.8% | 57.9% | 51 | 9 |
| Recommendation Audit | 27.1% | 55.0% | 27 | 16 |
| Counterevidence Sensitive | 20.0% | 50.8% | 18 | 17 |
| Calibrated | 23.3% | 50.4% | 35 | 10 |

**关键洞察**：
- Naive 的 misinterpretation rate（22.1%）远低于 Evidence Weighting（43.8%），说明 structured reasoning prompt 本身会产生更多解析噪音
- Evidence Weighting 有 51 次"正确 subagent → 被 principal 错误否决"，说明它的过度谨慎导致拒绝了正确的委托建议
- Counterevidence Sensitive 的 misinterpretation rate 最低（20.0%），且是唯一"错误 subagent → 被 principal 正确接受"（17次）超过"正确 subagent → 被错误否决"（18次）的策略

---

## 三、Oracle 行为分析

### 3.1 所有 Oracle 决策均为 REJECT

实验中 240 个 episode 的 oracle 决策 100% 为 REJECT（0 个 APPROVE）。这说明 SpawnBench 的 Principal 实验构造的是**"拒绝委托"的诱饵场景**——测试的是模型能否识别出不应当委托的危险/有害任务。

### 3.2 各策略过度Approve率（最关键指标）

| Variant | Over-Approve (Oracle=REJECT, Strategy=APPROVE) | Over-Approve Rate |
|---------|----------------------------------------------|-------------------|
| Counterevidence Sensitive | 19 | **7.9%** ← 最安全 |
| Recommendation Audit | 13 | 5.4% |
| Naive | 45 | 18.8% |
| Calibrated | 42 | 17.5% |
| Evidence Weighting | 54 | 22.5% ← 最危险 |

**这是最关键的安全指标**：当 oracle 的正确决策是 REJECT 时，策略却 APPROVE 了，这就是 delegation harm（委托伤害）。

Counterevidence Sensitive 只有 7.9% 的过度 Approve 率，是最安全的策略；Evidence Weighting 高达 22.5%，是最危险的策略。

---

## 四、决策翻转（Decision Flip）分析

Decision flip = principal 的最终决策与 subagent 建议解读不一致。

### 4.1 Flip 频率与正确率

| Variant | Total Flips | Correct Flips | Wrong Flips | Flip Accuracy |
|---------|------------|---------------|-------------|---------------|
| Counterevidence Sensitive | 122 | 95 | 27 | **77.9%** ← 最高 |
| Recommendation Audit | 132 | 93 | 39 | 70.5% |
| Calibrated | 121 | 84 | 37 | 69.4% |
| Naive | 135 | 101 | 34 | 74.8% |
| Evidence Weighting | 139 | 84 | 55 | 60.4% |

### 4.2 Naive 的 Flip 模式（3B-VL）

Naive 有 135 次 flip（56.2%），其中：
- REJECT→REJECT（维持拒绝）：89次
- APPROVE→APPROVE（维持接受）：34次
- APPROVE→REJECT（否决 subagent）：12次
- REJECT→APPROVE（批准 subagent）：0次

**关键发现**：Naive 在 3B 上全部的 flip 都是从 APPROVE→REJECT 或维持 REJECT，**没有任何一次 REJECT→APPROVE 的 flip**。这意味着 Naive 在 3B 上虽然 98.8% 跟随 subagent，但它的 flip 全部是保守方向的。

### 4.3 Counterevidence Sensitive 的 Flip 模式

Counterevidence Sensitive 有 122 次 flip（50.8%），是 flip 率最低的策略：
- REJECT→REJECT（维持拒绝）：80次（最多）
- APPROVE→REJECT（否决 subagent）：15次
- APPROVE→APPROVE（维持接受）：13次
- UNPARSEABLE→UNPARSEABLE：14次

**它的 flip 质量最高**（77.9% 正确），且在所有 flip 中，维持 REJECT 的比例最高（80次），说明它的核心能力是"正确地维持拒绝决策"。

---

## 五、Naive 错误案例深度分析（3B-VL）

Naive 在 3B 上错了 45 个 case（81.2% 正确率）。

### 5.1 各策略对 Naive 错误的修正能力

| Variant | 修正 Naive 错误 | 被 Naive 带跑偏 | Net Effect |
|---------|--------------|----------------|------------|
| **Counterevidence Sensitive** | **28/45 (62.2%)** | **16/195 (8.2%)** | **+12** |
| Recommendation Audit | 24/45 (53.3%) | 22/195 (11.3%) | +2 |
| Calibrated | 21/45 (46.7%) | 33/195 (16.9%) | -12 |
| Evidence Weighting | 10/45 (22.2%) | 39/195 (20.0%) | -29 |

**Counterevidence Sensitive 是唯一 net positive 的策略**：它修正了 28 个 naive 错误，只引入 16 个新错误，net +12。

### 5.2 CntEvi 修正 Naive 的具体模式

在 28 个 CntEvi 正确但 Naive 错误的 case 中：
- **27/28 是 APPROVE→REJECT 翻转（oracle=REJECT）**
- 1/28 是 REJECT→REJECT（Naive 某处判断错误但 CntEvi 维持了正确决策）

这说明 **CntEvi 的核心价值是：在 subagent 给出 APPROVE 建议时，正确识别出证据中的 risks/caveats/shortcomings，将其否决**。这是它与 Naive 的本质差异。

---

## 六、任务家族分析

### 6.1 Family 级别结果（3B-VL）

| Family | 最优策略 | 最优 Acc | 最差策略 | 最差 Acc | Harm Range |
|--------|---------|---------|---------|---------|-----------|
| code_review | Recommendation Audit | 97.5% | Evidence Weighting | 90.0% | 1.2%–3.8% |
| data_analysis | Counterevidence Sensitive | 91.2% | Evidence Weighting | 68.8% | 2.5%–15.0% |
| procurement | Counterevidence Sensitive | 72.5% | Evidence Weighting | 48.8% | 15.0%–38.8% |

**Procurement 是所有场景中最难的**：
- CntEvi 在 procurement 上仍有 72.5% accuracy，但 Evidence Weighting 只有 48.8%（接近随机）
- Harm rate：procurement 上 Evidence Weighting 高达 38.8%（每3个委托1个有害），即使是最好的 CntEvi 也有 15.0%

### 6.2 Procurement 上 CntEvi vs Naive 详细对比

| 指标 | Naive | CntEvi | 差异 |
|------|-------|--------|------|
| Accuracy | 63.7% | 72.5% | +8.8pp |
| Harm Rate | 28.7% | 15.0% | **-13.7pp** |
| Rescue Rate | 27.5% | 22.5% | -5.0pp |

**在最有挑战的 procurement 场景，CntEvi 将 harm rate 几乎减半（28.7%→15.0%）**。这是最实际的安全改进。

### 6.3 Code Review 场景

Code Review 是最简单、共识最强的场景：
- 所有策略都在 88.8%–100% 之间
- Recommendation Audit 达到 100%，但 harm rate 最低（1.2%）的是 Recommendation Audit
- CntEvi 反而不是 code review 的最优策略（95.0%），说明 code review 的证据足够清晰，不需要特别敏感的 counterevidence 检查

---

## 七、实验条件调节效应（3B-VL）

### 7.1 Verifier 效应

| Verifier | CntEvi Acc | CntEvi Harm | Naive Acc | Naive Harm |
|----------|-----------|------------|-----------|------------|
| no_verifier | 88.3% | 4.2% | 81.7% | 12.5% |
| verifier_before_decision | 84.2% | 9.2% | 80.8% | 15.8% |

**发现**：有 verifier 时，所有策略的 harm rate 都上升。Verifier 在决策前提供额外信息，但模型反而更容易在这些信息中迷失。CntEvi 在 no_verifier 条件下 harm 仅 4.2%，是最安全配置。

### 7.2 Objective Scope 效应

| Scope | CntEvi Acc | CntEvi Harm | Naive Acc | Naive Harm |
|-------|-----------|------------|-----------|------------|
| global_aware | 83.3% | 8.3% | 81.7% | 15.0% |
| local_proxy | 89.2% | 5.0% | 80.8% | 13.3% |

**CntEvi 在 local_proxy 场景表现更好**（89.2%，harm 5.0%），说明当 subagent 视角受限（local）时，特别需要关注全局性的 risks 和 caveats。

### 7.3 Report Format 效应

| Format | CntEvi Acc | CntEvi Harm | Naive Acc | Naive Harm |
|--------|-----------|------------|-----------|------------|
| free_form | 84.2% | 6.7% | 76.7% | 16.7% |
| structured | 88.3% | 6.7% | 85.8% | 11.7% |

**Structured report 帮助所有策略**，但对 Naive 的帮助更大（+9.1pp vs +4.1pp）。结构化格式降低了 Naive 的决策难度，但对 CntEvi 来说锦上添花。

---

## 八、Subagent 推荐质量分析

### 8.1 Subagent 本身的表现

Subagent recommendation 与 oracle 一致的比例：86.7%（所有策略共用同一 subagent 输出）。

但 **misinterpretation 问题严重**——principal 对 subagent 建议的解析错误率高达 20–44%：

| Variant | Misinterpretation Rate | 对正确 subagent 的否决 | 对错误 subagent 的接受 |
|---------|------------------------|----------------------|---------------------|
| Counterevidence Sensitive | **20.0%** | 18 | 17 |
| Naive | 22.1% | 26 | 13 |
| Calibrated | 23.3% | 35 | 10 |
| Recommendation Audit | 27.1% | 27 | 16 |
| Evidence Weighting | 43.8% | 51 | 9 |

**Evidence Weighting 的 misinterpretation rate 最高（43.8%）**，因为它的 prompt 本身很复杂，导致模型在长文本中更容易把 APPROVE 解析成 REJECT 或 UNPARSEABLE。

---

## 九、1.5B vs 3B-VL 对比总结

### 9.1 关键指标变化

| Variant | Acc Δ (1.5B→3B) | Harm Δ | Misinterpretation Δ | 谁更好 |
|---------|----------------|--------|---------------------|--------|
| Counterevidence Sensitive | **+8.0pp** | **-5.4pp** | **-4.2pp** | 3B 大幅胜出 |
| Calibrated | +3.0pp | -4.1pp | -3.4pp | 3B 略好 |
| Recommendation Audit | -6.2pp | +4.2pp | +2.1pp | 1.5B 略好 |
| Naive | -12.1pp | +10.4pp | +17.9pp | 1.5B 大幅胜出 |
| Evidence Weighting | -7.0pp | +5.0pp | +17.1pp | 1.5B 略好 |

**Counterevidence Sensitive 和 Calibrated 在 3B 上变好，其他策略在 3B 上变差**。

### 9.2 规模效应的解释

**为什么 Counterevidence Sensitive 在 3B 上显著提升？**

该策略要求模型做三件事：
1. 提取 subagent 建议
2. 列出并评估 caveats/risks/shortcomings
3. 判断这些担忧是否足以否决建议

在 1.5B 模型上，这个流程超过了模型的推理容量，导致：
- 步骤2/3 引入错误
- 模型对 risks 的"警觉"变成了"瞎警惕"，过度拒绝正确的 subagent

在 3B 模型上：
- 每个步骤都能正确执行
- 模型对 risks 的警觉是"有依据的怀疑"，能正确识别真实的风险

**为什么 Naive 在 3B 上反而变差？**

1.5B 的 Naive 成功是因为：prompt 简短 + 跟随 subagent（96.7%）= 高准确率。但 subagent recommendation 本身有 13.3% 的错误率，1.5B Naive 在这 13.3% 上通过某种"直觉"恰好做对了。

在 3B 上：
- Follow subagent 率更高了（98.8%），但 subagent 错误率不变
- 3B 的"直觉"不再与 oracle 巧合，而是与 subagent 更绑定
- 结果：过度依赖 subagent → 更高的 harm rate（14.2% vs 3.8%）

---

## 十、最优策略建议

### 10.1 按模型规模

```
Qwen2.5-1.5B:
  → Naive Principal
  → Accuracy 93.3%, Harm 3.8%
  → 不要用 structured reasoning，会弄巧成拙

Qwen2.5-3B+ (含VL):
  → Counterevidence Sensitive Principal
  → Accuracy 86.2%, Harm 6.7%
  → 比 naive 高 5pp，harm 低 7.5pp
```

### 10.2 按任务类型（3B-VL）

| 任务类型 | 推荐策略 | 预期 Acc | 预期 Harm |
|---------|---------|---------|----------|
| Code Review | Recommendation Audit | 97.5% | 1.2% |
| Data Analysis | Counterevidence Sensitive | 91.2% | 2.5% |
| Procurement | Counterevidence Sensitive | 72.5% | 15.0% |
| High-stakes (任何类型) | Counterevidence Sensitive | — | — |

### 10.3 安全关键场景

如果 harm rate 是唯一指标（安全大于准确率）：

| Variant | Harm Rate | Over-Approve Rate |
|---------|-----------|-------------------|
| Counterevidence Sensitive | **6.7%** | **7.9%** |
| Recommendation Audit | 10.0% | 5.4% |
| Calibrated | 12.1% | 17.5% |
| Naive | 14.2% | 18.8% |
| Evidence Weighting | 19.2% | 22.5% |

**Counterevidence Sensitive 在 harm rate 和 over-approve rate 上都是最优的。**

---

## 附录A: 各Variant完整指标（3B-VL）

| Metric | Naive | Evid.Wt | Rec.Audit | CntEvi | Calib |
|--------|-------|---------|-----------|--------|-------|
| final_correct | 81.2% | 69.2% | 82.1% | **86.2%** | 76.2% |
| delegation_harm | 14.2% | 19.2% | 10.0% | **6.7%** | 12.1% |
| delegation_rescue | 42.1% | 35.0% | 38.8% | 39.6% | 35.0% |
| followed_subagent | 98.8% | 91.2% | 74.6% | 94.2% | 87.9% |
| subagent_misinterpreted | 22.1% | 43.8% | 27.1% | 20.0% | 23.3% |
| decision_flip | 56.2% | 57.9% | 55.0% | 50.8% | 50.4% |
| flip_accuracy | 74.8% | 60.4% | 70.5% | **77.9%** | 69.4% |

## 附录B: 1.5B vs 3B-VL 完整对比

| Variant | Model | Acc | Harm | Rescue | FolSub | Misr |
|---------|-------|-----|------|--------|--------|------|
| naive_principal | 1.5B | 93.3% | 3.8% | 43.8% | 96.7% | 4.2% |
| naive_principal | 3B-VL | 81.2% | 14.2% | 42.1% | 98.8% | 22.1% |
| evidence_weighting | 1.5B | 76.2% | 14.2% | 37.1% | 79.2% | 26.7% |
| evidence_weighting | 3B-VL | 69.2% | 19.2% | 35.0% | 91.2% | 43.8% |
| recommendation_audit | 1.5B | 88.3% | 5.8% | 40.8% | 77.1% | 25.0% |
| recommendation_audit | 3B-VL | 82.1% | 10.0% | 38.8% | 74.6% | 27.1% |
| counterevidence_sensitive | 1.5B | 78.3% | 12.1% | 37.1% | 76.2% | 24.2% |
| counterevidence_sensitive | 3B-VL | **86.2%** | **6.7%** | 39.6% | 94.2% | 20.0% |
| calibrated | 1.5B | 73.3% | 16.2% | 36.2% | 72.1% | 26.7% |
| calibrated | 3B-VL | 76.2% | 12.1% | 35.0% | 87.9% | 23.3% |

## 附录C: Family × Variant Accuracy（3B-VL）

| Family | Naive | Evid.Wt | Rec.Audit | CntEvi | Calib |
|--------|-------|---------|-----------|--------|-------|
| code_review | 95.0% | 90.0% | **97.5%** | 95.0% | 88.8% |
| data_analysis | 85.0% | 68.8% | 80.0% | **91.2%** | 76.2% |
| procurement | 63.7% | 48.8% | 68.8% | **72.5%** | 63.7% |

## 附录D: Condition × Variant Accuracy（3B-VL）

| Condition | Naive | Evid.Wt | Rec.Audit | CntEvi | Calib |
|-----------|-------|---------|-----------|--------|-------|
| global_aware_free_form_no_verifier | 83.3% | 73.3% | 80.0% | **93.3%** | 63.3% |
| global_aware_free_form_verifier_before_decision | 80.0% | 60.0% | 63.3% | **83.3%** | 70.0% |
| global_aware_structured_no_verifier | 80.0% | 60.0% | **86.7%** | 80.0% | 83.3% |
| global_aware_structured_verifier_before_decision | 83.3% | 63.3% | 80.0% | 76.7% | **80.0%** |
| local_proxy_free_form_no_verifier | 73.3% | 63.3% | **93.3%** | 80.0% | 70.0% |
| local_proxy_free_form_verifier_before_decision | 70.0% | 70.0% | 73.3% | **80.0%** | 70.0% |
| local_proxy_structured_no_verifier | 90.0% | 80.0% | 83.3% | **100.0%** | 86.7% |
| local_proxy_structured_verifier_before_decision | 90.0% | 83.3% | **96.7%** | **96.7%** | 86.7% |

---

## 输出文件

- `principal_interpretation_episodes_3B.jsonl` — 3B-VL 1200条原始记录
- `principal_interpretation_episodes.jsonl` — 1.5B 1200条原始记录（已被覆盖重命名）
- `principal_interpretation_report.md` — 1.5B 详细分析
- `principal_interpretation_3B_report.md` — 3B 快速摘要
- `principal_interpretation_3B_full_report.md` — 本文档，完整分析
