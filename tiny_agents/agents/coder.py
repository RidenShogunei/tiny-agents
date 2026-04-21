"""Coder agent: generates and reviews code."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


CODER_PROMPT = """You are a coding assistant. Write clean, correct, and well-documented code.
When given a task:
1. Analyze requirements
2. Write code
3. Include brief comments
4. If tests are needed, include them

Respond with the code block first, then a brief explanation."""


class CoderAgent(BaseAgent):
    """Generates code and handles programming tasks."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct", **kwargs):
        super().__init__(
            name="coder",
            model_name=model_name,
            role_prompt=CODER_PROMPT,
            **kwargs,
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Generate code for the given task."""
        task = input_data.get("task", "")
        self.add_message("user", task)

        # TODO: call LLM for code generation
        return AgentOutput(
            thought="Generated code for the task",
            action="respond",
            payload={"code": "# TODO: generated code", "task": task},
            finished=True,
        )
