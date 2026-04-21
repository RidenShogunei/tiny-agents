"""Coder agent: generates and reviews code."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


CODER_PROMPT = """You are an expert coding assistant. Write clean, correct, well-documented Python code.

Rules:
1. Output only the code block (inside ```python ... ```)
2. Include brief docstrings
3. Handle edge cases when possible
4. Do not include explanations outside the code block unless asked"""


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

        if self.backend is not None:
            code = self._call_llm(
                f"Write Python code for: {task}",
                temperature=0.3,
                max_tokens=1024,
            )
        else:
            code = "# Backend not available\ndef placeholder():\n    pass"

        self.add_message("assistant", code)

        return AgentOutput(
            thought="Generated code using LLM",
            action="respond",
            payload={"code": code, "task": task},
            finished=True,
        )
