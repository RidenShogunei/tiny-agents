# OpenCode + Qwen2.5-3B-Instruct 配置与 tiny-agents Baseline 设计记录

> 生成时间：2026-04-28
> 关键词：OpenCode、vLLM、Qwen2.5-3B、子 Agent 架构、BudgetOrchestrator、Pipeline

---

## 一、OpenCode + Qwen2.5-3B-Instruct 配置

### 1.1 背景

OpenCode 是一个本地 Code Agent 框架，支持通过 `--provider opencode` 指定后端模型。项目旨在使用小模型（0.5B-3B）在本地完成编程任务，减少对云端 API 的依赖。

### 1.2 最终配置

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
  --model /home/jinxu/.cache/tiny-agents/models/Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen2.5-3B-Instruct \
  --port 18000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 48000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --load-format safetensors \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**启动验证：**
```bash
curl http://localhost:18000/v1/models
# 返回: {"object":"list","data":[{"id":"Qwen2.5-3B-Instruct",...,"max_model_len":48000}]}
```

**OpenCode 调用测试：**
```bash
~/.opencode/bin/opencode run "What is 2+2? Answer in one word."
# 输出: "four"
```

### 1.3 关键配置项解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `port` | 18000 | vLLM API 端口，OpenCode 默认连接本地 18000 |
| `max_model_len` | 48000 | 最大上下文长度 |
| `gpu-memory-utilization` | 0.90 | GPU 显存占用比例 |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | env | 允许超出模型原始位置编码限制 |
| `enable-auto-tool-choice` | flag | 启用自动工具选择 |
| `tool-call-parser` | qwen3_coder | Qwen 工具调用解析器 |

### 1.4 重要发现与问题

#### 问题 1：32k context 不够用

**现象：** 早期配置使用 `max_model_len=32000`，OpenCode 请求 `max_tokens=32000`，加上系统 prompt 和工具定义后，总 context 超过 32768 限制。

**原因：** OpenCode 请求中自己指定了 `max_tokens`，而非依赖 vLLM 的 `override-generation-config`。因此即使设置生成配置覆盖也无效。

**解决：** 将 `max_model_len` 提高到 48000，以容纳 `32000 output + 工具 overhead`。

#### 问题 2：RoPE 位置编码警告

**警告信息：**
```
User-specified max_model_len (48000) is greater than the derived max_model_len
(max_position_embeddings=32768.0 or model_max_length=None in model's config.json).
VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme caution.
If the model uses relative position encoding (RoPE), positions exceeding
derived_max_model_len lead to nan.
```

**影响：** 当生成长度超过 32768 时，attention score 可能出现 NaN，输出质量不可预测。短输出（如 "2+2"）不受影响，但复杂代码生成存在风险。

**建议：** 如需稳定生成，考虑将 `max_model_len` 设置为 32768 或略高（36000 左右），同时在 OpenCode 调用时限制 `max_tokens`。

#### 问题 3：代理连接重置

**现象：**
```
ConnectionResetError: [Errno 104] Connection reset by peer
```

**原因：** vLLM server 在处理某些请求时主动断开了连接，可能与生成长度超限或 RoPE 异常有关。

---

## 二、tiny-agents Baseline 架构

### 2.1 项目概述

tiny-agents 是一个基于小模型（Qwen2.5 0.5B-3B）的多 Agent 框架，使用 vLLM 作为推理后端。项目包含三套并行的 Agent 编排架构，适用于不同场景。

**目录结构（核心）：**
```
tiny_agents/
├── agents/          # Agent 类型定义
│   ├── coder.py
│   ├── critic.py
│   ├── router.py
│   ├── tool_reasoner.py
│   ├── verifier.py
│   └── vl_perception.py
├── budget/          # 预算感知编排
│   ├── controller.py
│   ├── orchestrator.py
│   ├── candidate_manager.py
│   ├── credit_tracer.py
│   └── state_builder.py
├── core/            # 核心抽象
│   ├── agent.py     # BaseAgent 基类
│   ├── orchestrator.py  # 串行链式编排
│   ├── pipeline.py  # DAG 流水线
│   ├── session.py
│   └── message_bus.py
├── models/          # 后端
│   ├── vllm_backend.py
│   └── vlm_backend.py
└── tools/           # 工具实现
```

### 2.2 三套编排架构对比

| 架构 | 类型 | 子 Agent 调用方式 | 并行支持 | 预算控制 | 适用场景 |
|------|------|------------------|----------|----------|----------|
| `Orchestrator` | 串行链式 | `delegate` action | ❌ | ❌ | 简单单轮对话 |
| `BudgetOrchestrator` | 固定调度 | `_execute_atomic_step()` | ❌ | ✅ | 多轮讨论、Math/GSM8K |
| `Pipeline` | DAG 流水线 | `PipelineStep` | ✅ | ❌ | 数据处理、多分支任务 |

---

## 三、Orchestrator — 串行链式委托

### 3.1 设计原理

```python
# 核心循环
while iteration < max_iterations:
    output = await agent.run(current_input, ctx)

    if output.action == "delegate":
        current_agent = target  # 切换到目标 Agent
        continue

    elif output.action == "respond":
        return output.payload

    elif output.action == "tool_call":
        result = tool.execute(args)
        ctx.add_message(agent, "user", result_to_message(result))
        continue  # 同一 Agent 继续
```

### 3.2 AgentOutput 动作类型

```python
class AgentOutput(BaseModel):
    action: str  # respond | delegate | tool_call | review | parallel
    target_agent: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    finished: bool = True
```

| Action | 含义 |
|--------|------|
| `respond` | Agent 产生最终输出，流程结束或进入 Critic 审查 |
| `delegate` | 将控制权交给另一个 Agent（target in payload） |
| `tool_call` | 请求执行工具，Orchestrator 内联执行后注入结果 |
| `review` | Critic 审查完成，等待决策 |
| `parallel` | （代码中存在但未完整实现） |

### 3.3 局限性

- **串行执行**：每次只有一个 Agent 在运行
- **无 KV Cache 共享**：每个 Agent 实例独立，即使同模型也无法复用
- **上下文重复**：每个 Agent 接收完整 system prompt + 历史
- **delegate 是同步调用**：本质是 `agent.run()`，不是真正的子进程/并行

---

## 四、BudgetOrchestrator — 预算感知多 Agent 协作

### 4.1 设计原理

```
BudgetController.decide(s_t, c_t) ← BEFORE action!
        │
    action
        │
    ├─► STOP:      return best_answer
    │
    ├─► CALL_VERIFIER:
    │      verifier_output = verifier.run()
    │      update τ_t, c_t, b_t
    │      continue
    │
    └─► CONTINUE_DISCUSS:
           Orchestrator.step() → single atomic action
           update τ_t, c_t, b_t
           continue
```

### 4.2 Phase 1 固定调度（无隐藏智能）

```python
PHASE1_SCHEDULE = [
    ("reasoner", "REASONER_STEP"),   # Step 1: initial reasoning
    ("critic", "CRITIC_STEP"),       # Step 2: critique
    ("reasoner", "REASONER_STEP"),   # Step 3: respond to critique
    ("critic", "CRITIC_STEP"),       # Step 4: re-evaluate
    ("reasoner", "REASONER_STEP"),   # Step 5: final reasoning
]
```

- 奇数步 = reasoner，偶数步 = critic
- 调度规则**完全显式**，无任何隐藏决策逻辑

### 4.3 核心状态变量

| 变量 | 说明 |
|------|------|
| `budget_state` | 当前剩余预算、已消耗 |
| `candidate_mgr` | 候选答案管理，`best` 存储当前最佳 |
| `credit_tracer` | 追踪每个 step 的 token 消耗和质量 |
| `last_verifier_output` | 上次验证结果，影响 reasoner 后续响应 |
| `_has_seen_verifier_output` | 标记是否看过 verifier 输出 |

### 4.4 子 Agent 调用机制

```python
async def _execute_atomic_step(self, problem, budget_state):
    # 根据 fixed schedule 或 verifier feedback 选择 Agent
    if self._has_seen_verifier_output:
        agent_name = self.reasoner_name
        instruction = f"Based on verifier's feedback, revise..."
    else:
        agent_name, action_type = self._get_next_in_schedule()

    # 构建 prompt
    prompt = self._build_step_prompt(problem, agent_name, action_type)

    # 调用 Agent
    backend = self.agent_backends.get(agent_name, self.llm_backend)
    response = backend.chat([{"role": "user", "content": prompt}])

    return CollaborationStep(content=response, ...)
```

**问题：** 仍然是串行调用，没有真正并行。

---

## 五、Pipeline — DAG 流水线（唯一支持并行的架构）

### 5.1 设计原理

Pipeline 是一个有向无环图（DAG）处理流水线，每个节点是 `PipelineStep`，边是数据依赖。支持 `parallel=True` 的节点在依赖满足时并发执行。

```python
steps = [
    PipelineStep(id="plan", agent=planner, inputs=["topic"], outputs=["outline"]),
    PipelineStep(id="write", agent=writer, inputs=["outline"], outputs=["content"], parallel=True),
    PipelineStep(id="review", agent=reviewer, inputs=["content"], outputs=["review"]),
    PipelineStep(id="format", agent=formatter, inputs=["review"], outputs=["final"]),
]
pipe = Pipeline(steps)
result = await pipe.run({"topic": "LoRA in Vision Models"})
```

### 5.2 依赖解析与并行

```python
def _resolve_order(self) -> List[List[str]]:
    # Kahn算法拓扑排序
    # 返回 [[step_a, step_b], [step_c], [step_d, step_e]]
    # 同组内并行执行，组间串行
```

### 5.3 子 Agent 调用

```python
async def run_one(step_id):
    dep_data = {dep_id: outputs[dep_id] for dep_id in step.depends_on}
    step_input = {**context, **dep_data}

    if step._is_agent():
        session = SessionContext(session_id=f"pipe_{step.id}", config=step.config)
        output = await step.runner.run(step_input, session)
        result = output.payload
    elif step._is_tool():
        result = step.runner.execute(step_input)

    # 存储输出到 context
    for key in step.output_keys:
        context[key] = result[key]

    return step_id, result, None

# 并发执行同组步骤
results = await asyncio.gather(*[run_one(sid) for sid in group])
```

---

## 六、子 Agent 优化的现状与问题

### 6.1 当前优化情况

| 优化点 | 状态 | 说明 |
|--------|------|------|
| 并行子 Agent | ⚠️ 部分 | 仅 Pipeline 支持，需手动声明 `parallel=True` |
| KV Cache 共享 | ❌ 无 | 每个 Agent 实例独立，无跨实例共享 |
| 上下文复用 | ⚠️ 部分 | SessionContext 共享，但每个 Agent 仍需完整 prompt |
| 预算感知调度 | ✅ | BudgetOrchestrator 有完整 BudgetController |
| 固定调度（可解释） | ✅ | BudgetOrchestrator Phase 1 完全显式 |
| 工具调用内联 | ✅ | Orchestrator 内联执行，不走子进程 |

### 6.2 核心未解决问题

#### 6.2.1 延迟瀑布（Latency Cascade）

```
用户请求 → Router → [Tool A] → [Tool B] → Worker → [Tool C] → Critic → 响应
              ↑______________每个箭头是一次完整的 LLM 往返延迟______________↑
```

**影响：** 每多一层 Agent 调用，增加一次完整推理延迟。对于 3B 模型，单次推理约 0.5-2s，多层累积可达 10s+。

#### 6.2.2 上下文膨胀（Context Explosion）

每个 Agent 的 `system prompt` 包含：
- Role 定义
- 工具 schema
- 行为约束
- 示例

**问题：** 10 个子 Agent 意味着相同的工具定义被重复编码 N 次。

#### 6.2.3 KV Cache 无法跨实例共享

即使：
- 所有 Agent 使用同一模型（Qwen2.5-3B）
- 运行在同一 GPU 上
- 使用同一个 vLLM server

每个 `backend.chat()` 调用仍然需要重新编码 system prompt，无法复用之前的 KV cache。

#### 6.2.4 小模型调度 overhead 占比高

对于 0.5B-3B 模型：
- LLM 推理时间：0.5-2s
- Agent 调度 overhead（序列化、IPC、结果聚合）：0.1-0.5s

**overhead 可能占 10-50%**，远高于大模型场景。

### 6.3 改进方向

1. **浅层委托**：限制 Agent 嵌套深度（建议 ≤ 2 层）
2. **共享 LLM 实例**：同 GPU + 同模型用同一个 vLLM backend，复用连接
3. **Pipeline 并行**：对可分解任务，优先用 Pipeline 而非 Orchestrator
4. **上下文缓存**：在调用前将固定 system prompt 的 KV cache 预热
5. **路由前置**：用轻量模型做调度判断，减少不必要的 Agent 调用

---

## 七、相关文件索引

| 文件 | 作用 |
|------|------|
| `tiny_agents/core/orchestrator.py` | 串行链式 Orchestrator |
| `tiny_agents/core/agent.py` | BaseAgent 基类定义 |
| `tiny_agents/core/pipeline.py` | DAG Pipeline 编排器 |
| `tiny_agents/budget/orchestrator.py` | BudgetOrchestrator 实现 |
| `tiny_agents/budget/controller.py` | BudgetController 决策逻辑 |
| `tiny_agents/budget/candidate_manager.py` | 候选答案管理 |
| `tiny_agents/budget/state_builder.py` | 状态构建器 |
| `tiny_agents/models/vllm_backend.py` | vLLM 后端封装 |
| `tiny_agents/tools/python_executor.py` | Python 代码执行工具 |

---

*本文档由 Hermes Agent 自动生成，记录了 OpenCode + Qwen 配置过程与 tiny-agents Baseline 架构分析。*
