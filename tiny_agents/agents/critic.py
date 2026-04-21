"""Critic agent: reviews and validates outputs from other agents."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


CRITIC_PROMPT = """You are a strict code reviewer and logic validator.

Your job:
1. Check if the code is syntactically correct Python
2. Check if the logic matches the task requirements
3. Identify edge cases that are not handled
4. Check for performance issues or anti-patterns
5. Verify that the docstrings and comments are accurate

Respond in this exact format:
verdict: PASS or NEEDS_FIX
issues: (list specific problems, one per line, or "None" if PASS)
suggestions: (concrete improvement suggestions, or "None" if PASS)

Be concise and actionable."""


class CriticAgent(BaseAgent):
    """Reviews code, logic, and reasoning from worker agents."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", **kwargs):
        super().__init__(
            name="critic",
            model_name=model_name,
            role_prompt=CRITIC_PROMPT,
            **kwargs,
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Review the output from another agent."""
        code = input_data.get("code", "")
        task = input_data.get("task", "")
        reasoning = input_data.get("reasoning", "")

        # Build review prompt
        review_prompt = f"Task: {task}\n\n"
        if code:
            review_prompt += f"Code to review:\n```python\n{code}\n```\n\n"
        if reasoning:
            review_prompt += f"Reasoning to validate:\n{reasoning}\n\n"
        review_prompt += "Please review the above and provide your verdict."

        self.add_message("user", review_prompt)

        if self.backend is not None:
            review_text = self._call_llm(
                review_prompt,
                temperature=0.2,
                max_tokens=512,
            )
        else:
            review_text = "verdict: PASS\nissues: None\nsuggestions: None"

        self.add_message("assistant", review_text)

        # Parse verdict
        verdict = "PASS"
        if "NEEDS_FIX" in review_text.upper() or "FAIL" in review_text.upper():
            verdict = "NEEDS_FIX"

        return AgentOutput(
            thought=f"Review complete: {verdict}",
            action="review",
            payload={
                "verdict": verdict,
                "review": review_text,
                "original_task": task,
                "original_code": code,
            },
            finished=(verdict == "PASS"),
        )
