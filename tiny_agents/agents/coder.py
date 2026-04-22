"""Coder agent: generates and reviews code."""

from typing import Any, Dict

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


CODER_PROMPT = """You are an expert coding assistant. Write clean, correct, well-documented Python code.

Rules:
1. Include ALL necessary imports at the top (e.g., from typing import List)
2. Output only the code block (inside ```python ... ```)
3. Include brief docstrings
4. Handle edge cases when possible
5. Do not include explanations outside the code block unless asked"""


class CoderAgent(BaseAgent):
    """Generates code and handles programming tasks (stateless)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", **kwargs):
        super().__init__(
            name="coder",
            model_name=model_name,
            role_prompt=CODER_PROMPT,
            **kwargs,
        )

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Generate code for the given task, optionally with review feedback."""
        task = input_data.get("task", "")
        feedback = input_data.get("review_feedback", "")
        needs_fix = input_data.get("needs_fix", False)

        if needs_fix and feedback:
            prompt = (
                f"Original task: {task}\n\n"
                f"Review feedback:\n{feedback}\n\n"
                "Please rewrite the code addressing ALL the issues mentioned above."
            )
        else:
            prompt = f"Write Python code for: {task}"

        # Build messages using context (no internal history)
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 1024)
            code = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        else:
            code = "# Backend not available\ndef placeholder():\n    pass"

        # Record in context
        context.add_message(self.name, "user", prompt)
        context.add_message(self.name, "assistant", code)

        return AgentOutput(
            thought="Generated code using LLM" + (" (rewrite with feedback)" if needs_fix else ""),
            action="respond",
            payload={"code": code, "task": task, "feedback": feedback},
            finished=True,
        )
