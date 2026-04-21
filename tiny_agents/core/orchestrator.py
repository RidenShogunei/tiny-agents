"""Orchestrator: manages multi-agent task decomposition and execution."""

import asyncio
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.memory import SharedMemory
from tiny_agents.core.message_bus import MessageBus


class Orchestrator:
    """Routes tasks and coordinates multi-agent workflows with iteration support."""

    def __init__(self, max_iterations: int = 3, enable_review: bool = False):
        self.agents: Dict[str, BaseAgent] = {}
        self.memory = SharedMemory()
        self.bus = MessageBus()
        self.max_iterations = max_iterations
        self.enable_review = enable_review
        self.critic_agent: Optional[str] = None

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        # Inject shared memory and bus into agent
        agent.memory = self.memory
        agent.bus = self.bus

    def _get_agent(self, name: str) -> BaseAgent:
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not registered")
        return self.agents[name]

    async def execute(
        self,
        task_input: Dict[str, Any],
        entry_agent: str = "router",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a task with full traceability and optional multi-round iteration.

        Workflow:
            1. Entry agent processes input
            2. If action == delegate -> forward to target_agent
            3. If action == review   -> forward to critic for review
            4. If critic says NEEDS_FIX -> loop back to original worker
            5. Repeat up to max_iterations
        """
        session_id = session_id or f"sess_{id(task_input)}"
        steps: List[Dict[str, Any]] = []
        current_agent = entry_agent
        current_input = task_input.copy()
        current_input["session_id"] = session_id

        # Store original task in memory
        self.memory.add_short_term({
            "session_id": session_id,
            "event": "task_start",
            "input": task_input,
        })

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            agent = self._get_agent(current_agent)

            # Run agent
            output: AgentOutput = await agent.run(current_input)

            step_record = {
                "step": iteration,
                "agent": current_agent,
                "input": current_input,
                "output": output.model_dump(),
            }
            steps.append(step_record)
            self.memory.add_short_term({
                "session_id": session_id,
                "event": "agent_step",
                **step_record,
            })

            # Publish to bus for real-time observers
            await self.bus.publish(
                f"session.{session_id}",
                {"agent": current_agent, "output": output.model_dump()},
            )

            if output.action == "delegate":
                # Hand off to another worker agent
                target = output.target_agent or "coder"
                if target not in self.agents:
                    return {
                        "success": False,
                        "reason": f"Target agent '{target}' not found",
                        "steps": steps,
                        "session_id": session_id,
                    }
                current_agent = target
                current_input = output.payload or {}
                current_input["session_id"] = session_id
                continue

            elif output.action == "respond":
                # Worker produced final output
                # If review chain is enabled and critic exists, forward to critic
                if self.enable_review and self.critic_agent and self.critic_agent in self.agents:
                    # Only forward if this agent is not the critic itself
                    if current_agent != self.critic_agent:
                        current_agent = self.critic_agent
                        current_input = output.payload or {}
                        current_input["session_id"] = session_id
                        continue

                return {
                    "success": True,
                    "result": output.payload,
                    "steps": steps,
                    "session_id": session_id,
                }

            elif output.action == "review":
                # Critic has reviewed; check verdict
                verdict = output.payload.get("verdict", "PASS")
                if verdict == "NEEDS_FIX" and iteration < self.max_iterations:
                    # Loop back to the previous worker with review feedback
                    prev_steps = [s for s in steps if s["agent"] != self.critic_agent]
                    if prev_steps:
                        last_worker_step = prev_steps[-1]
                        worker_name = last_worker_step["agent"]
                        worker_input = last_worker_step["input"].copy()
                        worker_input["review_feedback"] = output.payload.get("review", "")
                        worker_input["needs_fix"] = True
                        current_agent = worker_name
                        current_input = worker_input
                        continue
                # Either PASS or max iterations reached -> finish
                return {
                    "success": True,
                    "result": output.payload,
                    "steps": steps,
                    "session_id": session_id,
                }

            else:
                return {
                    "success": False,
                    "reason": f"Unknown action: {output.action}",
                    "steps": steps,
                    "session_id": session_id,
                }

        # Max iterations reached
        return {
            "success": False,
            "reason": f"Max iterations ({self.max_iterations}) reached",
            "steps": steps,
            "session_id": session_id,
        }
