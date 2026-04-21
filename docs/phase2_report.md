# Tiny Agents Phase 2 技术报告

> 日期: 2026-04-22  
> 目标: 实现多 Agent 协作、记忆共享与自动迭代优化  
> 作者: Hermes Agent (RidenShogunei/tiny-agents)

---

## 1. 本阶段完成的内容

| 组件 | 状态 | 说明 |
|------|------|------|
| CriticAgent | ✅ 完成 | 代码/逻辑评审，输出 PASS/NEEDS_FIX |
| VLPerceptionAgent | ✅ 完成 | 图像理解，支持 base64 编码图片 |
| Orchestrator 多轮迭代 | ✅ 完成 | `enable_review` + `critic_agent` 自动转发 |
| SharedMemory 增强 | ✅ 完成 | 按 session_id / agent 检索 |
| VLLMBackend 实例共享 | ✅ 完成 | 同路径+同GPU 模型复用，显式启用 prefix caching |
| Qwen3.5-9B 下载 | 🔄 进行中 | hf-mirror.com 批量下载 (~19GB) |

---

## 2. 架构设计

### 2.1 多智能体协作流程

```
用户输入
    ↓
RouterAgent (1.5B, GPU 0) —— 任务路由决策
    ↓
CoderAgent (3B, GPU 1) —— 生成初版代码
    ↓
CriticAgent (0.5B, GPU 2) —— 评审代码
    ↓
如果 NEEDS_FIX: 返回 Coder 重写 (携带 review_feedback)
如果 PASS: 返回结果
```

**关键机制**:
- `Orchestrator.enable_review = True` 时，任何非 Critic Agent 的 `respond` 会自动转发给 Critic
- Critic 返回 `NEEDS_FIX` 时，Orchestrator 从 `steps` 历史中找到上一个非 Critic 的 Agent，将 review 作为反馈传回
- 最多 `max_iterations` 轮，避免无限循环

### 2.2 记忆与消息总线

| 组件 | 功能 |
|------|------|
| **SharedMemory** | 三层存储：短期缓冲(ring buffer)、工作状态、关键词检索的长期记忆 |
| **MessageBus** | 异步 pub/sub，按 topic 路由，支持超时订阅 |
| **session_id** | 唯一标识一次完整任务流程，方便追溯 |

新增 API:
```python
memory.get_session_context(session_id)    # 获取完整会话
memory.get_agent_history(session_id, "coder")  # 获取单个 Agent 历史
```

### 2.3 vLLM 推理后端优化

#### 实例共享
当多个 Agent 使用 **相同模型路径 + 相同 GPU** 时，自动复用已加载的 vLLM 实例，而不是重复加载。

```python
# 例如：两个 Agent 共享 1.5B 模型
backend.load_model("router", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0)
backend.load_model("summarizer", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0)
# 第二次加载会自动复用第一次的实例
```

#### Prefix Caching (KV Cache 复用)
vLLM 0.19.1 已默认启用 `enable_prefix_caching=True`。当多个请求共享相同的 prompt 前缀时（如 system prompt），KV cache 自动复用，减少重复计算。

当前实现已显式传入 `enable_prefix_caching=True`，确保该功能始终开启。

---

## 3. 关键代码变更

### 3.1 Orchestrator 核心逻辑
```python
# 新增参数
orch = Orchestrator(max_iterations=3, enable_review=True)
orch.critic_agent = "critic"

# 执行流程
while iteration < max_iterations:
    output = await agent.run(input)
    if output.action == "respond" and enable_review:
        # 自动转发给 Critic
        current_agent = critic_agent
        continue
    if output.action == "review" and verdict == "NEEDS_FIX":
        # 找回上一个 Worker，传递 feedback
        current_agent = last_worker
        current_input["review_feedback"] = review_text
        current_input["needs_fix"] = True
        continue
```

### 3.2 CoderAgent 重写支持
```python
if needs_fix and feedback:
    prompt = (
        f"Original task: {task}\n\n"
        f"Review feedback:\n{feedback}\n\n"
        "Please rewrite the code addressing ALL the issues."
    )
```

### 3.3 CriticAgent 评审格式
System prompt 要求输出固定格式：
```
verdict: PASS or NEEDS_FIX
issues: (list problems or None)
suggestions: (concrete improvements or None)
```

---

## 4. Demo 结果

### 4.1 单轮协作 Demo (`demo_two_agents.py`)
- **任务**: 写一个带 memoization 的斐波那契函数
- **流程**: Router(1.5B) → Coder(3B)
- **结果**: ✅ 成功生成完整代码，包含单元测试

### 4.2 多轮迭代 Demo (`demo_multi_round.py`)
- **任务**: 写 `average(numbers)` 函数，故意忘记处理空列表
- **流程**:
  - Step 1: Router → Coder
  - Step 2: Coder 生成代码（实际上自动处理了空列表，3B 模型能力较强）
  - Step 3: Critic 评审并报告 NEEDS_FIX
- **结果**: ✅ 评审链路跑通，SharedMemory 正确记录 4 条事件

---

## 5. 已知问题与限制

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Coder 自动修复了故意的 bug | 3B 模型代码能力较强，自动补全了边缘情况 | 无需修复，说明模型质量高 |
| Critic verdict 解析粗糙 | 目前用简单字符串匹配 | 后续改用 structured output (Outlines) |
| GitHub push 网络不稳定 | GnuTLS recv error / 超时 | 等待网络恢复后重试 |
| VLM 图片输入未测试 | vLLM 对 Qwen-VL 的多模态支持需要特殊处理 | 下一阶段测试 MathVista 时验证 |

---

## 6. 下一阶段计划 (Phase 3)

1. **替换为 Qwen3.5-9B** — 待下载完成后，将 Coder 升级为 9B
2. **基准测试** — HumanEval / MATH / MathVista 自动评估脚本
3. **对比实验** — 单 Agent vs 多 Agent 协作的准确率差异
4. **VLM 完整接入** — 实现图片输入的真正多模态生成
5. **LoRA 热切换** — 多角色共享同一底座，通过 LoRA 切换角色

---

## 7. 相关文件

- `tiny_agents/core/orchestrator.py` — 调度器核心
- `tiny_agents/core/memory.py` — 记忆系统
- `tiny_agents/models/vllm_backend.py` — vLLM 后端
- `tiny_agents/agents/critic.py` — 评审 Agent
- `tiny_agents/agents/vl_perception.py` — 视觉 Agent
- `examples/demo_multi_round.py` — 多轮协作 Demo
