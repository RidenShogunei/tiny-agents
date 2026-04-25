# SpawnBench 480 Episodes 结果分析报告

> 生成时间: 2026-04-25  
> 数据来源: `spawnbench/data/episodes.jsonl` (480 episodes, 60 tasks × 8 cells)  
> 模型配置: Principal=Qwen3.5-9B, Subagent/Verifier=Qwen2.5-1.5B

---

## 📊 总体指标

| 指标 | 数值 |
|------|------|
| 总 episodes | 480 (60 tasks × 8 cells) |
| delegation 前正确率 | 29.8% (143/480) |
| delegation 后正确率 | 47.7% (229/480) |
| **净提升** | **+17.9pp** (+86 episodes) |
| 决策翻转率 | 21.7% (104/480) |
| delegation 有害率 | 1.9% (9/480) |
| delegation 有效率 | 19.8% (95/480) |
| 关键证据被遗漏 | 50.0% (240/480) |

---

## 🔄 决策流向分析

```
初始正确 → 保持正确:  134 (27.9%)   ← 不需要翻转
初始错误 → 被拯救:     95 (19.8%)   ← delegation 发挥作用
初始正确 → 被误导:      9 (1.9%)   ← delegation 造成伤害
初始错误 → 持续错误:   242 (50.4%)   ← delegation 无法拯救
```

**关键发现**: 拯救/伤害比 = **10.6x**，delegation 整体利大于弊。

---

## 🔍 Verifier 效果分析

| Verifier 类型 | 拯救 | 伤害 | 净提升 | 总正确率 |
|--------------|------|------|--------|---------|
| **无 verifier** | +40 | +3 | +16.7pp | 45.4% |
| **有 verifier_before_decision** | +52 | +6 | +19.2pp | 50.0% |

- Verifier 的边际效益: **+6 episodes** (+2.5pp)
- Verifier 检测到问题: 63/257 (24.5%) 的"subagent 对而 principal 错"案例

**结论**: Verifier 能在 subagent 正确但 principal 错误时提供额外纠偏，但提升有限。核心纠偏力量仍来自 **subagent 的建议**。

---

## 👁️ Subagent 表现

| 指标 | 数值 |
|------|------|
| Subagent 建议 APPROVE | 36.7% (176/480) |
| Subagent 建议 REJECT | 63.3% (304/480) |
| Oracle 50/50 平衡 | APPROVE 240 vs REJECT 240 |
| **Subagent 与 Oracle 一致率** | **80.8% (388/480)** |

Subagent 与 Oracle 的高一致率 (80.8%) 说明了两个问题：
1. Subagent 1.5B 小模型在给定详细上下文后判断质量不错
2. 但 principal 9B 仍然在 52% 的案例中与 subagent 意见相左并犯错

---

## 📁 按任务家族分析

| Family | 初始正确率 | 最终正确率 | 翻转率 | 拯救率 |
|--------|-----------|-----------|--------|--------|
| code_review | 35.0% | 46.2% | 15.0% | 13.1% |
| data_analysis | 19.4% | 46.9% | **31.2%** | **29.4%** |
| procurement | 35.0% | **50.0%** | 18.8% | 16.9% |

**发现**: `data_analysis` 任务 delegation 效果最显著 (翻转率 31.2%)，也是 principal 最容易初始犯错的一类 (初始正确率仅 19.4%)。

---

## 📋 按 Objective Scope

| Scope | 初始正确率 | 最终正确率 | 净提升 |
|-------|-----------|-----------|--------|
| global_aware | 29.6% | 50.8% | +21.2pp |
| local_proxy | 30.0% | 44.6% | +14.6pp |

**global_aware scope delegation 效果更好** (+21.2pp vs +14.6pp)，因为它让 principal 看到完整上下文后做更全面的判断。

---

## 📋 按 Report Format

| Format | 初始正确率 | 最终正确率 | 翻转率 | 净提升 |
|--------|-----------|-----------|--------|--------|
| free_form | 31.2% | 49.6% | 22.5% | +18.4pp |
| structured | 28.3% | 45.8% | 20.8% | +17.5pp |

---

## 🚨 典型失败案例

### Delegation 伤害案例 (principal 初始 REJECT 正确 → 被误导为 APPROVE)

**case 1: `code_review_008` (verifier_before_decision)**
- Oracle: REJECT
- Principal 初始: REJECT ✓
- Subagent: REJECT (conf=0.9)
- Verifier: "subagent 报告忠实但未解决 race condition，faithfulness=0.0"
- Principal 最终: **APPROVE** ✗
- **问题**: verifier 指出了关键遗漏但 principal 反而跟随了 subagent 的错误判断

**case 2: `code_review_010` (verifier_before_decision)**
- Oracle: REJECT
- Principal 初始: REJECT ✓
- Subagent: **APPROVE** (conf=0.9)
- Verifier: "subagent 报告忠实，faithfulness=1.0"
- Principal 最终: **APPROVE** ✗
- **问题**: subagent 给出了看似合理的 APPROVE 理由，verifier 判为忠实后 principal 跟随

### Delegation 拯救案例 (principal 初始 APPROVE 错误 → 被修正为 REJECT)

**case 1: `code_review_001` (no_verifier)**
- Oracle: REJECT
- Principal 初始: APPROVE ✗
- Subagent: **REJECT** (conf=0.9)
- Principal 最终: **REJECT** ✓
- **关键**: subagent 正确识别了 race condition 和安全问题

---

## 💡 主要结论

1. **Delegation 整体有效**: 净 +17.9pp 提升，10.6 倍的拯救/伤害比
2. **Verifier 边际价值有限**: 主要纠偏来自 subagent，verifier 只额外贡献 +2.5pp
3. **Evidence omission 是主要瓶颈**: 50% 案例有关键证据被遗漏，可能与 0.5B 模型生成长度受限有关
4. **Data analysis 任务 delegation 效果最好**: 初始错误率最高 (80.6%)，delegation 拯救率也最高 (29.4%)
5. **Global aware scope > local proxy scope**: 完整上下文让 principal 做得更好

---

## 🔧 改进方向建议

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| P0 | 解决 evidence omission (模型输出长度限制) | +10~15pp |
| P1 | 增强 verifier 纠偏能力 (独立 GPU 避免 CUDA graph 冲突) | +2~3pp |
| P2 | Data analysis 任务专项优化 | +5~8pp |
| P3 | Subagent 置信度校准 (避免 0.9 虚假高置信度) | +2~3pp |
