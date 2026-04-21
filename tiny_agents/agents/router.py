"""Router agent: decomposes tasks and dispatches to worker agents."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


ROUTER_PROMPT = """You are a task router. Analyze the user's request and decide which agent should handle it.

Available agents:
- coder: for programming, code generation, debugging, data analysis
- reasoner: for logic puzzles, math word problems, step-by-step reasoning
- vl_perception: for tasks involving images, screenshots, charts, documents

Respond with exactly one word: the agent name. No explanation."""


class RouterAgent(BaseAgent):
    """Routes incoming tasks to appropriate worker agents."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", **kwargs):
        super().__init__(
            name="router",
            model_name=model_name,
            role_prompt=ROUTER_PROMPT,
            **kwargs,
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Route task to the best worker agent using LLM."""
        task = input_data.get("task", "")
        has_image = input_data.get("image") is not None

        # If image is present, route to VLM directly
        if has_image:
            return AgentOutput(
                thought="Task contains image, routing to vl_perception",
                action="delegate",
                target_agent="vl_perception",
                payload={"task": task, **input_data},
            )

        # Use LLM for routing decision
        if self.backend is not None:
            response = self._call_llm(
                f"Task: {task}\nWhich agent should handle this? (coder/reasoner/vl_perception)",
                temperature=0.1,
                max_tokens=10,
            )
            target = response.strip().lower()
            # Normalize
            if "coder" in target:
                target = "coder"
            elif "reason" in target:
                target = "reasoner"
            elif "vl" in target or "vision" in target or "image" in target:
                target = "vl_perception"
            else:
                target = "coder"  # default fallback
        else:
            # Fallback heuristic when no backend
            target = "coder" if any(kw in task.lower() for kw in ["code", "function", "program", "debug"]) else "reasoner"

        return AgentOutput(
            thought=f"LLM routed task to {target}",
            action="delegate",
            target_agent=target,
            payload={"task": task, **input_data},
        )
