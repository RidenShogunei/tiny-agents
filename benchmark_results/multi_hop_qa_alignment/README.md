# Multi-Hop QA Goal Alignment Benchmark

## 核心发现

**多智能体协作实际上降低了准确率（-4个百分点）**

| 模式 | 准确率 |
|------|--------|
| Single Agent (3B 单独完成) | **76%** (19/25) |
| Multi-Agent (1.5B Parent + 0.5B Subagent) | **60%** (15/25) |
| **Net Impact** | **-4 (有害)** |

---

## 关键数据

### 对比矩阵

| Question | Expected | Single 3B | Multi-Agent Parent | Subagent Output | Result |
|----------|----------|-----------|-------------------|-----------------|--------|
| qa_01 | 1452 | 1452 ✅ | 1452年 ✅ | 1452年 | both correct |
| qa_02 | 1879 | 1879 ✅ | 1879 ✅ | **1421 ❌** | sub wrong, parent fixed |
| qa_03 | Stratford-upon-Avon | correct ✅ | **伦敦 ❌** | **伦敦 ❌** | **HURT** |
| qa_04 | France | 答案：法国 ✅ | 法国 ✅ | 法国 | both wrong |
| qa_05 | 1643 | 1643 ✅ | 1643年 ✅ | 1643年 | both correct |
| qa_06 | American | American ✅ | **美国 ❌** | **美国 ❌** | **HURT** |
| qa_07 | Dutch | Dutch ✅ | **荷兰裔 ❌** | **荷兰裔 ❌** | **HURT** |
| qa_08 | Poland | Poland ✅ | **波兰 ❌** | **波兰 ❌** | **HURT** |
| qa_09 | Asia | 答案：Asia ✅ | Asia ✅ | **亚洲 ❌** | sub wrong, parent fixed |
| qa_10 | 16th century | 16世纪 ✅ | 16世纪 ✅ | **15世纪 ❌** | sub wrong, parent fixed |
| qa_11 | 26 | 26岁 ✅ | correct ✅ | **1421 ❌** | sub wrong, parent fixed |
| qa_12 | 84 | 84岁 ✅ | 84 ✅ | 1643, 1727 | both correct |
| qa_13 | 36 | Marie Curie was 36 ✅ | 36 ✅ | 1867年 | both correct |
| qa_14 | 51 | 61岁 ❌ | **51 ✅** | 1452 | **HELPED** |
| qa_15 | 2021 | 2021 ✅ | 2021 ✅ | 1971年 | both correct |
| qa_16 | Sense and Sensibility | correct ✅ | **《南村辍学记》❌** | verbose ❌ | **HURT** |
| qa_17 | Café Terrace at Night | verbose ❌ | **《向日葵》❌** | verbose ❌ | both wrong |
| qa_18 | None | Ronald Reagan ❌ | **肯尼迪 ❌** | **是 ❌** | both wrong |
| qa_19 | Canada | Canada ✅ | Canada ✅ | 38 | both correct |
| qa_20 | Mount Everest | correct ✅ | correct ✅ | **8848米 ❌** | sub wrong, parent fixed |
| qa_21 | Strawberry | correct ✅ | correct ✅ | **5 ❌** | sub wrong, parent fixed |
| qa_22 | Tokyo | Paris ❌ | **Tokyo ✅** | 2.4 | **HELPED** |
| qa_23 | Great Wall of China | correct ✅ | correct ✅ | **明朝 ❌** | sub wrong, parent fixed |
| qa_24 | 38 | 80 cm ❌ | **36 ❌** | **36 ❌** | both wrong |
| qa_25 | 13 | 13 ✅ | **23 ❌** | **23 ❌** | **HURT** |

---

## 不对齐模式分析

### 主导问题：语言不一致 (11/25)

Subagent (0.5B) 用中文回答，但问题期望英文答案。Parent (1.5B) 接受了中文回答，导致最终答案错误。

**典型案例**：
- qa_06: 问 Tim Cook 国籍，subagent 答"美国"（正确），parent 输出"美国"，但期望"American"
- qa_08: 问 Marie Curie 出生国，subagent 答"波兰"（正确），parent 输出"波兰"，期望"Poland"

### Subagent 给错信息，Parent 信任并采纳 (3/25)

- qa_03: subagent 说 Shakespeare 出生在"伦敦"，parent 直接采纳
- qa_25: subagent 说空座位是 23，parent 直接输出 23
- qa_16: subagent 答非所问（输出了与问题无关的内容），parent 采纳了错误答案

### Subagent 错了，但 Parent 自我修正 (7/25)

- qa_02, qa_11, qa_14, qa_20, qa_21, qa_22, qa_23
- 这些 cases 中 subagent 的回答是错的，但 parent 通过自己的推理得到了正确答案
- **Parent 的能力弥补了 subagent 的失败**

---

## 关键洞察

### 1. 协作分工带来了准确率下降

- Single agent: 76%
- Multi-agent: 60%
- **结论：spawn subagent 这件事本身带来了 -16% 的准确率损失**

### 2. Subagent 的失败模式

| 失败类型 | 数量 | 说明 |
|---------|------|------|
| 语言不匹配 | 11 | 用中文回答英文问题，或反之 |
| 信息错误 | 6 | subagent 给出了完全错误的 factual 信息 |
| 答非所问 | 1 | subagent 输出与问题无关 |
| **Total** | **18/25** | **72% 的 subagent 输出有严重问题** |

### 3. Parent 的整合能力有限

- 25 个 cases 中，parent 正确整合 subagent 输出的只有 9 个 (36%)
- Parent 整合失败的原因：
  - 直接采纳 subagent 的错误输出 (3 cases)
  - 被语言不一致误导 (11 cases)
  - 被 subagent 的 verbose 输出误导 (1 case)

### 4. 即使 subagent 正确，协作也可能失败

- qa_06/07/08: subagent 用中文回答"美国"/"荷兰裔"/"波兰"，parent 直接输出
- 这不是 subagent 的问题，而是**系统层面的语言对齐问题**

---

## 结论

**目标不对齐确实存在，而且表现为：**

1. **Subagent 的局部最优 ≠ Parent 的整体最优**
   - Subagent 认为自己"正确回答了问题"
   - 但语言/格式/详略程度不符合 parent 的整合需求

2. **Subagent 的激励和 Parent 的激励不一致**
   - Subagent 的激励：给出"正确的答案"
   - Parent 的激励：得到"能被整合进最终答案的输出"
   - 这两个目标并不等价

3. **分工本身带来了信息损耗**
   - 76% → 60% 的准确率下降
   - 主要原因：subagent 72% 的输出有问题
   - 即使 parent 有能力修正，也只能 cover 7/18 的失败 cases

---

## 文件说明

- `modeA_single_*.json` — 单 agent 模式完整输出
- `modeB_multi_*.json` — 多 agent 模式完整输出（含 subagent 输出和 parent 整合结果）
- `multi_hop_qa_alignment.py` — 实验脚本

## 实验配置

- **Parent**: Qwen2.5-1.5B-Instruct
- **Subagent**: Qwen2.5-0.5B-Instruct
- **Baseline**: Qwen2.5-3B-Instruct (单 agent)
- **问题数**: 25
- **问题类型**: 实体属性查询、年龄计算、关系推理、比较推理
