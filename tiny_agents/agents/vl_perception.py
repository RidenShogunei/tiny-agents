"""VL Perception agent: understands images using Qwen2.5-VL.

Fully stateless — receives (input_data, context) and returns AgentOutput.
The backend (VLMBackend) must be set before use via register_backend().
"""

import base64
import re
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


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to a base64 data URL."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class VLPerceptionAgent(BaseAgent):
    """Processes images and visual content using Qwen2.5-VL-3B (fully stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen2.5-VL-3B-Instruct",
        **kwargs,
    ):
        super().__init__(
            name="vl_perception",
            model_name=model_name,
            role_prompt=VL_PROMPT,
            **kwargs,
        )
        # Lazy backend reference (set by orchestrator or direct initialization)
        self._vl_backend: Optional[Any] = None

    @property
    def vl_backend(self):
        return self._vl_backend

    @vl_backend.setter
    def vl_backend(self, backend: Any) -> None:
        self._vl_backend = backend

    def set_backend(self, backend: Any) -> None:
        """Set the VLM backend (compatibility alias for vl_backend setter)."""
        self._vl_backend = backend

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Process an image and answer the task question (stateless)."""
        task = input_data.get("task", "Describe this image.")
        image = input_data.get("image")
        image_path = input_data.get("image_path")
        image_url: Optional[str] = None

        # Resolve image to data URL
        if image_path:
            image_url = encode_image_base64(image_path)
        elif image and isinstance(image, str):
            if image.startswith("data:"):
                image_url = image
            elif image.startswith("http"):
                # Remote URL — pass as-is (vLLM will fetch)
                image_url = image
            else:
                image_url = image
        elif image and isinstance(image, dict):
            # Already a structured image dict
            image_url = image.get("url") or image.get("image_url")

        # Build content blocks
        if image_url:
            content_blocks = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": task},
            ]
        else:
            content_blocks = [{"type": "text", "text": task}]

        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": content_blocks})

        # Generate response via VLM backend
        if self._vl_backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 1024)
            response = self._vl_backend.generate(
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
            )
        else:
            response = "[VLM backend not available — set .vl_backend or .set_backend()]"

        context.add_message(self.name, "user", f"[image: {image_url[:50]}... if image_url else 'none'] {task}")
        context.add_message(self.name, "assistant", response)

        return AgentOutput(
            thought=f"Analyzed image: {image_url[:50] if image_url else 'none'}",
            action="respond",
            payload={
                "description": response,
                "task": task,
                "image_url": image_url,
            },
            finished=True,
        )
