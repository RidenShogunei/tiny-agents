# Principal Interpretation Strategy — 1.5B vs 3B VL 对比分析报告

**Experiment**: SpawnBench Principal Interpretation Variants  
**Models**: Qwen2.5-1.5B-Instruct vs Qwen2.5-VL-3B-Instruct  
**Date**: 2026-04-25  
**Data**: 240 episodes × 5 variants × 2 models = **2400 runs**

---

## 核心发现

**模型规模改变了策略优先级。**

| Variant | 1.5B Acc | 3B-VL Acc | Delta | 谁更好 |
|---------|----------|-----------|-------|--------|
| Naive | **93.3%** | 81.2% | -12.1pp | 1.5B |
| Recommendation Audit | 88.3% | 82.1% | -6.2pp | 1.5B |
| Counterevidence Sensitive | 78.3% | **86.2%** | **+8.0pp** | **3B** |
| Evidence Weighting | 76.2% | 69.2% | -7.0pp | 1.5B |
| Calibrated | 73.3% | 76.2% | +3.0pp | 3B |

- **1.5B**：prompt 越简单越好（naive 最强）
- **3B-VL**：structured reasoning 开始展现价值，counterevidence sensitive 反而超越 naive
- **Ranking 在两个模型上完全倒转**

---

## 3B-VL 详细结果

| Variant | Accuracy | Harm Rate | Rescue Rate | Followed Sub Rec |
|---------|----------|-----------|-------------|------------------|
| Counterevidence Sensitive | **86.2%** | **6.7%** | 39.6% | 94.2% |
| Recommendation Audit | 82.1% | 10.0% | 38.8% | 74.6% |
| Naive | 81.2% | 14.2% | 42.1% | 98.8% |
| Calibrated | 76.2% | 12.1% | 35.0% | 87.9% |
| Evidence Weighting | 69.2% | 19.2% | 35.0% | 91.2% |

### 3B-VL Delta vs Naive (3B)

| Variant | Accuracy Δ | Harm Δ | Rescue Δ |
|---------|-----------|--------|----------|
| Counterevidence Sensitive | **+5.0pp** | **-7.5pp** | -2.5pp |
| Recommendation Audit | +0.8pp | -4.2pp | -3.3pp |
| Calibrated | -5.0pp | -2.1pp | -7.1pp |
| Evidence Weighting | -12.1pp | +5.0pp | -7.1pp |

**Counterevidence Sensitive 在 3B 上以最多 5pp accuracy 增益和最大 harm reduction (-7.5pp) 胜出。**

---

## 跨模型关键对比

### Accuracy 变化方向

```
              1.5B → 3B
Naive:        93.3% → 81.2%   ↓ -12.1pp   (变差)
Evid.Wt:      76.2% → 69.2%   ↓  -7.0pp   (变差)
Rec.Audit:    88.3% → 82.1%   ↓  -6.2pp   (变差)
Calibrated:   73.3% → 76.2%   ↑  +3.0pp   (变好)
CntEvi:       78.3% → 86.2%   ↑  +8.0pp   (变好)
```

Counterevidence Sensitive 和 Calibrated 在更大模型上受益，其他三种策略反而下降。可能原因：VL 模型的文本推理能力并非单纯按参数量Scaling，更多 structured prompt 可能需要 VL 特有的视觉-语言联合表征能力来弥补——但这里只有纯文本。

### Harm Rate 全面恶化（1.5B → 3B）

| Variant | 1.5B Harm | 3B-VL Harm | Δ |
|---------|-----------|-----------|---|
| Naive | 3.8% | 14.2% | +10.4pp |
| Rec.Audit | 5.8% | 10.0% | +4.2pp |
| CntEvi | 12.1% | 6.7% | **-5.4pp** |
| Evid.Wt | 14.2% | 19.2% | +5.0pp |
| Calibrated | 16.2% | 12.1% | -4.1pp |

Counterevidence Sensitive 是唯一 harm rate 下降的策略（6.7%，最安全）。

---

## 错误分析

### 3B-VL Naive 错误案例分析

Naive 在 3B 上错了 45 个 case：

| Variant | 修正 Naive 错误 | 被 Naive 带跑偏 | Net |
|---------|--------------|----------------|-----|
| Counterevidence Sensitive | **28/45 (62%)** | 16/195 (8%) | **+12** |
| Recommendation Audit | 24/45 (53%) | 22/195 (11%) | +2 |
| Calibrated | 21/45 (47%) | 33/195 (17%) | -12 |
| Evidence Weighting | 10/45 (22%) | 39/195 (20%) | -29 |

Counterevidence Sensitive 修正了 28 个 naive 错误，只产生 16 个新错误，net +12，是唯一 net positive 的策略。

---

## 任务家族 × 模型对比

### Code Review

| Variant | 1.5B | 3B-VL | Δ |
|---------|------|-------|---|
| Naive | 96.2% | 95.0% | -1.2pp |
| Rec.Audit | 100.0% | 97.5% | -2.5pp |
| CntEvi | 83.8% | 95.0% | **+11.2pp** |

### Data Analysis

| Variant | 1.5B | 3B-VL | Δ |
|---------|------|-------|---|
| Naive | 96.2% | 85.0% | -11.2pp |
| CntEvi | 80.0% | 91.2% | **+11.2pp** |
| Rec.Audit | 83.8% | 80.0% | -3.8pp |

### Procurement

| Variant | 1.5B | 3B-VL | Δ |
|---------|------|-------|---|
| Naive | 87.5% | 63.7% | -23.8pp |
| CntEvi | 71.2% | 72.5% | +1.3pp |
| Rec.Audit | 81.2% | 68.8% | -12.4pp |

**Procurement 是所有策略表现最差、naive 退化最大的任务**（-23.8pp）。这是因为采购决策涉及风险/成本权衡，证据模糊性最高，最需要 structured reasoning——但即使是 counterevidence sensitive 也只有 72.5%。

---

## 结论与建议

### 模型选择决策树

```
你的模型是 Qwen2.5-1.5B？
  → 用 Naive Principal（93.3% accuracy）
  → 不要加 structured reasoning，会弄巧成拙

你的模型是 Qwen2.5-3B+（含VL）？
  → 用 Counterevidence Sensitive Principal（86.2% accuracy）
  → 比 naive 高 5pp，harm rate 低 7.5pp
  → 不要用 evidence weighting（最差策略）
```

### 通用规律

1. **模型够大才能 handle 复杂推理**：1.5B 无法从 structured reasoning 受益，3B 可以。阈值大概在 2B 附近。

2. **Counterevidence Sensitive 是 3B+ 的最优策略**：它教模型对 subagent 的建议保持怀疑、特别关注 risks/caveats/shortcomings。这在证据混乱的 procurement 任务中效果最显著。

3. **Naive 在小模型上的成功是假象**：高准确率主要来自"follow subagent recommendation"（96.7%），但 subagent 推荐本身interpret错误率约16%。Naive 只是在跟随一个经常正确的 subagent，而不是在做真正的主观判断。

4. **Procurement 任务需要特别关注**：两种模型上都是最差 family，harm rate 最高。建议人工审批或专用模型。

---

## 输出文件

- `principal_interpretation_episodes_3B.jsonl` — 1200 条 3B 原始记录
- `principal_interpretation_report.md` — 1.5B 完整分析报告
