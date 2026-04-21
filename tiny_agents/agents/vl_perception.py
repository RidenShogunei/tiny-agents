"""Visual perception agent: processes images and extracts structured information."""

from typing import Any, Dict
from tiny_agents.core.agent import BaseAgent, AgentOutput


VL_PROMPT = """You are a visual perception assistant. Analyze images and extract structured information.
Describe what you see clearly and concisely. If the image contains charts, tables, or diagrams,
extract the data in structured format."""


class VLPerceptionAgent(BaseAgent):
    """Processes visual input using a VLM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        super().__init__(
            name="vl_perception",
            model_name=model_name,
            role_prompt=VL_PROMPT,
            **kwargs,
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process image and return structured description."""
        image = input_data.get("image")
        task = input_data.get("task", "")

        # TODO: call VLM for image understanding
        return AgentOutput(
            thought="Analyzed the image",
            action="delegate",
            target_agent="coder",
            payload={
                "image_description": "# TODO: VLM output",
                "task": task,
            },
        )
