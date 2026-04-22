"""VL Perception agent: understands images using Qwen2.5-VL."""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


VL_PROMPT = """You are a visual perception assistant. Analyze the provided image carefully.

Your capabilities:
1. Describe what you see in the image
2. Read and interpret text, charts, diagrams in the image
3. Answer specific questions about the image content
4. Extract structured data from tables or forms

Be precise and thorough. If the image contains math problems, solve them step by step."""


def _encode_image(image_path: str) -> str:
    """Encode an image file to base64 data URL."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class VLPerceptionAgent(BaseAgent):
    """Processes images and visual content using Qwen2.5-VL-3B (stateless)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        super().__init__(
            name="vl_perception",
            model_name=model_name,
            role_prompt=VL_PROMPT,
            **kwargs,
        )

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Process an image and answer the task question (stateless)."""
        task = input_data.get("task", "")
        image = input_data.get("image")
        image_path = input_data.get("image_path")

        # Resolve image URL
        if image_path:
            image_url = _encode_image(image_path)
        elif image and isinstance(image, str) and image.startswith("data:"):
            image_url = image
        else:
            image_url = None

        # Build prompt
        if image_url:
            prompt = f"<|vision_start|>{image_url}<|vision_end|>\n{task}"
        else:
            prompt = task

        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 1024)
            response = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        else:
            response = "[VL Backend not available]"

        context.add_message(self.name, "user", prompt)
        context.add_message(self.name, "assistant", response)

        return AgentOutput(
            thought="Analyzed image content",
            action="respond",
            payload={
                "description": response,
                "task": task,
                "image_url": image_url,
            },
            finished=True,
        )
