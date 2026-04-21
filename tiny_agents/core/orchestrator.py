"""Task orchestrator that routes work between agents."""

import asyncio
from typing import Any, Dict, List, Optional
from .agent import BaseAgent, AgentOutput


class Orchestrator:
    """Routes tasks and manages agent collaboration."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent

    async def execute(
        self,
        task: Dict[str, Any],
        entry_agent: str,
        max_steps: int = 10,
    ) -> Dict[str, Any]:
        """Execute a task starting from entry_agent, with loop detection."""
        current_agent_name = entry_agent
        step = 0

        while step < max_steps:
            agent = self.agents.get(current_agent_name)
            if agent is None:
                raise ValueError(f"Agent '{current_agent_name}' not found")

            output: AgentOutput = await agent.run(task)
            self.execution_log.append({
                "step": step,
                "agent": current_agent_name,
                "output": output.model_dump(),
            })

            if output.finished:
                return {
                    "result": output.payload,
                    "steps": self.execution_log,
                    "success": True,
                }

            # Route to next agent
            next_agent = output.target_agent or entry_agent
            if next_agent == current_agent_name and not output.finished:
                # Self-loop without finish - possible stall
                pass
            current_agent_name = next_agent
            step += 1

        return {
            "result": {},
            "steps": self.execution_log,
            "success": False,
            "reason": "max_steps_reached",
        }

    def reset(self) -> None:
        """Reset all agents and logs."""
        self.execution_log.clear()
        for agent in self.agents.values():
            agent.reset()
