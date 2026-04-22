"""Orchestrator: manages multi-agent task decomposition and execution.

Core contract change from v1:
- Orchestrator creates ONE SessionContext per execute() call.
- Every agent.run(input_data, context) receives the same context object.
- No agent holds message_history internally — context.messages is the sole record.
- No manual reset() needed; each execute() call gets a fresh context.
- tool_call action: orchestrator executes tool and re-calls agent with result.

Workflow (serial chain):
    Entry agent -> Worker -> [tool_call loop] -> [optional Critic] -> respond

Supported action types:
    respond   — agent produced final output; return to caller
    delegate  — agent hands off to another agent (target in payload["target"])
    review    — critic has reviewed; orchestrator handles loop-back logic
    tool_call — agent requested a tool; orchestrator executes, injects result, re-calls
"""

import uuid
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.core.message_bus import MessageBus
from tiny_agents.tools.base import ToolRegistry


class Orchestrator:
    """Routes tasks and coordinates multi-agent workflows."""

    def __init__(
        self,
        max_iterations: int = 3,
        enable_review: bool = False,
        tools: Optional[ToolRegistry] = None,
    ):
        self.agents: Dict[str, BaseAgent] = {}
        self.bus = MessageBus()
        self.max_iterations = max_iterations
        self.enable_review = enable_review
        self.critic_agent: Optional[str] = None
        self.tools = tools or ToolRegistry()

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent. Inject the shared message bus."""
        self.agents[agent.name] = agent
        agent.bus = self.bus

    def set_critic(self, agent_name: str) -> None:
        """Set the critic agent for review loops."""
        if agent_name not in self.agents:
            raise ValueError(f"Critic agent '{agent_name}' not registered")
        self.critic_agent = agent_name

    def register_tool(self, tool) -> None:
        """Register a tool with the orchestrator's tool registry."""
        self.tools.register(tool)

    async def execute(
        self,
        task_input: Dict[str, Any],
        entry_agent: str = "router",
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task with a fresh SessionContext."""
        session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        config = config or {}

        ctx = SessionContext(
            session_id=session_id,
            config=config,
        )
        # Expose tools to agents via context
        ctx._tools = self.tools

        # Register system prompts for all agents
        for name, agent in self.agents.items():
            ctx.set_system_prompt(name, agent.role_prompt)

        ctx.add_step({
            "event": "task_start",
            "session_id": session_id,
            "input": task_input,
        })

        current_agent = entry_agent
        current_input = task_input.copy()
        iteration = 0
        tool_call_depth = 0   # prevent infinite tool loops

        while iteration < self.max_iterations:
            iteration += 1

            if current_agent not in self.agents:
                return self._failure(
                    session_id, f"Agent '{current_agent}' not registered", []
                )

            agent = self.agents[current_agent]

            ctx.add_step({
                "step": iteration,
                "agent": current_agent,
                "input": dict(current_input),
            })

            output: AgentOutput = await agent.run(current_input, ctx)

            ctx.short_term[-1]["output"] = output.model_dump()

            await self.bus.publish(
                f"session.{session_id}",
                {"agent": current_agent, "output": output.model_dump()},
            )

            # ── tool_call: execute tool and re-call same agent with result ──
            if output.action == "tool_call":
                tool_call_depth += 1
                if tool_call_depth > 10:
                    return self._failure(
                        session_id,
                        "Tool call depth limit (10) exceeded — possible infinite loop",
                        ctx.short_term,
                    )

                tool_name = output.payload.get("tool") or output.payload.get("tool_name", "")
                tool_args = output.payload.get("args", {})

                tool = self.tools.get(tool_name)
                if tool is None:
                    ctx.add_message(
                        current_agent, "user",
                        f"Tool '{tool_name}' not found. Available tools: {self.tools.list_names()}"
                    )
                    continue   # re-call same agent with error message

                result = tool.execute(tool_args)
                ctx.add_message(current_agent, "user", result_to_message(result))
                # Same agent continues in next iteration with tool result in context
                continue

            tool_call_depth = 0   # reset on any non-tool action

            # ── delegate ───────────────────────────────────────────
            if output.action == "delegate":
                target = output.target_agent or output.payload.get("target")
                if not target or target not in self.agents:
                    return self._failure(
                        session_id,
                        f"Delegate target '{target}' not found",
                        ctx.short_term,
                    )
                current_agent = target
                current_input = output.payload.copy()
                continue

            # ── respond ─────────────────────────────────────────────
            elif output.action == "respond":
                if (
                    self.enable_review
                    and self.critic_agent
                    and current_agent != self.critic_agent
                ):
                    current_agent = self.critic_agent
                    current_input = output.payload.copy()
                    continue
                return self._success(session_id, output.payload, ctx.short_term)

            # ── review ─────────────────────────────────────────────
            elif output.action == "review":
                verdict = output.payload.get("verdict", "PASS")
                if verdict == "NEEDS_FIX" and iteration < self.max_iterations:
                    worker_steps = [s for s in ctx.short_term
                                    if s.get("agent") != self.critic_agent]
                    if worker_steps:
                        last_worker = worker_steps[-1]["agent"]
                        feedback = output.payload.get("review", "")
                        current_agent = last_worker
                        current_input = {
                            **worker_steps[-1]["input"],
                            "review_feedback": feedback,
                            "needs_fix": True,
                        }
                        continue
                return self._success(session_id, output.payload, ctx.short_term)

            else:
                return self._failure(
                    session_id,
                    f"Unknown action: {output.action}",
                    ctx.short_term,
                )

        return self._failure(
            session_id,
            f"Max iterations ({self.max_iterations}) reached",
            ctx.short_term,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _success(
        self,
        session_id: str,
        result: Dict[str, Any],
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "result": result,
            "steps": steps,
            "session_id": session_id,
        }

    def _failure(
        self,
        session_id: str,
        reason: str,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "reason": reason,
            "steps": steps,
            "session_id": session_id,
        }


# ── tool result serializer ───────────────────────────────────────────────────

def result_to_message(result) -> str:
    """Convert a ToolResult to a user message injected into the agent context."""
    if result.success:
        return (
            f"Tool '{result.tool_name}' executed successfully.\n"
            f"Output: {result.output!r}\n"
        )
    else:
        return (
            f"Tool '{result.tool_name}' failed.\n"
            f"Error: {result.error}\n"
            f"Please fix your arguments or try a different approach."
        )
