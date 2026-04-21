"""Router agent: decomposes tasks and dispatches to worker agents."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


ROUTER_PROMPT = """You are a task router. Analyze the user's request and decide which agent should handle it.

Available agents:
- vl_perception: for tasks involving images, screenshots, charts, documents
- coder: for programming, code generation, debugging, data analysis
- reasoner: for logic puzzles, math word problems, step-by-step reasoning
- critic: for reviewing, validating, or checking outputs

Respond in JSON format:
{
  "thought": "your analysis",
  "action": "delegate",
  "target_agent": "one of the above",
  "payload": {"task": "summarized task description"}
}"""


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
        """Route task to the best worker agent."""
        # TODO: integrate with LLM inference
        task = input_data.get("task", "")
        has_image = input_data.get("image") is not None

        target = "vl_perception" if has_image else "coder"

        return AgentOutput(
            thought=f"Task has_image={has_image}, routing to {target}",
            action="delegate",
            target_agent=target,
            payload={"task": task, **input_data},
        )
