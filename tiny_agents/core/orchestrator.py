"""Orchestrator: manages multi-agent task decomposition and execution.

Core contract change from v1:
- Orchestrator creates ONE SessionContext per execute() call.
- Every agent.run(input_data, context) receives the same context object.
- No agent holds message_history internally — context.messages is the sole record.
- No manual reset() needed; each execute() call gets a fresh context.

Workflow (serial chain):
    Entry agent -> Worker -> [optional Critic -> loop back] -> respond

Supported action types:
    respond  — agent produced final output; return to caller
    delegate — agent hands off to another agent (target in payload["target"])
    review   — critic has reviewed; orchestrator handles loop-back logic
    tool_call — agent requested tool execution; handled inline (see tools/)
"""

import uuid
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.core.message_bus import MessageBus


class Orchestrator:
    """Routes tasks and coordinates multi-agent workflows."""

    def __init__(
        self,
        max_iterations: int = 3,
        enable_review: bool = False,
    ):
        self.agents: Dict[str, BaseAgent] = {}
        self.bus = MessageBus()
        self.max_iterations = max_iterations
        self.enable_review = enable_review
        self.critic_agent: Optional[str] = None

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent. Inject the shared message bus."""
        self.agents[agent.name] = agent
        agent.bus = self.bus

    def set_critic(self, agent_name: str) -> None:
        """Set the critic agent for review loops."""
        if agent_name not in self.agents:
            raise ValueError(f"Critic agent '{agent_name}' not registered")
        self.critic_agent = agent_name

    async def execute(
        self,
        task_input: Dict[str, Any],
        entry_agent: str = "router",
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task with a fresh SessionContext.

        Args:
            task_input:  the initial payload for the entry agent
            entry_agent: name of the agent to start with
            session_id:  optional stable ID (auto-generated if omitted)
            config:      per-session overrides (temperature, max_tokens, etc.)
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        config = config or {}

        # Fresh context per call — no cross-call leakage
        ctx = SessionContext(
            session_id=session_id,
            config=config,
        )

        # Register system prompts for all agents
        for name, agent in self.agents.items():
            ctx.set_system_prompt(name, agent.role_prompt)

        # Store task start
        ctx.add_step({
            "event": "task_start",
            "session_id": session_id,
            "input": task_input,
        })

        current_agent = entry_agent
        current_input = task_input.copy()
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if current_agent not in self.agents:
                return self._failure(
                    session_id, f"Agent '{current_agent}' not registered", []
                )

            agent = self.agents[current_agent]

            # Record step start
            ctx.add_step({
                "step": iteration,
                "agent": current_agent,
                "input": dict(current_input),   # copy to avoid mutation
            })

            # Call agent with BOTH input_data and context
            output: AgentOutput = await agent.run(current_input, ctx)

            # Record step end (attach output)
            ctx.short_term[-1]["output"] = output.model_dump()

            # Publish to bus
            await self.bus.publish(
                f"session.{session_id}",
                {"agent": current_agent, "output": output.model_dump()},
            )

            # ── action routing ──────────────────────────────────
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

            elif output.action == "respond":
                # Final output — optionally route to critic for review
                if (
                    self.enable_review
                    and self.critic_agent
                    and current_agent != self.critic_agent
                ):
                    current_agent = self.critic_agent
                    current_input = output.payload.copy()
                    continue

                return self._success(session_id, output.payload, ctx.short_term)

            elif output.action == "review":
                verdict = output.payload.get("verdict", "PASS")
                if verdict == "NEEDS_FIX" and iteration < self.max_iterations:
                    # Loop back to last worker with feedback
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

                # PASS or max iterations -> done
                return self._success(session_id, output.payload, ctx.short_term)

            else:
                return self._failure(
                    session_id,
                    f"Unknown action: {output.action}",
                    ctx.short_term,
                )

        # Max iterations reached
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
