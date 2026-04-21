"""Critic agent: validates outputs and provides feedback."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


CRITIC_PROMPT = """You are a quality critic. Review the given output carefully.
Check for:
- Correctness (facts, logic, calculations)
- Completeness (did it address all requirements?)
- Code quality (if code is present: syntax, style, edge cases)

Respond with:
- verdict: "pass" or "revise"
- feedback: specific issues and suggestions"""


class CriticAgent(BaseAgent):
    """Reviews outputs from other agents."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", **kwargs):
        super().__init__(
            name="critic",
            model_name=model_name,
            role_prompt=CRITIC_PROMPT,
            **kwargs,
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Review the provided output."""
        output_to_review = input_data.get("output", {})
        self.add_message("user", f"Please review: {output_to_review}")

        # TODO: call LLM for critique
        return AgentOutput(
            thought="Reviewed the output",
            action="respond",
            payload={"verdict": "pass", "feedback": "No issues found."},
            finished=True,
        )
