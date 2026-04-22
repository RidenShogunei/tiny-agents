"""VLM Backend: vLLM-backed multimodal LLM for Qwen2.5-VL.

Uses vLLM's .chat() API which handles:
- Chat template application (Qwen's special format)
- Vision encoding via Qwen's built-in processor
- Multimodal content list: [{"type": "image_url"}, {"type": "text"}]

GPU assignment: set CUDA_VISIBLE_DEVICES before init (default: GPU 2)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

DEFAULT_VL_MODEL = "/home/jinxu/models/Qwen/Qwen2___5-VL-3B-Instruct"


class VLMBackend:
    """Multimodal backend using Qwen2.5-VL via vLLM.

    Usage:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        backend = VLMBackend()
        await backend.initialize()
        response = backend.generate(messages=[...])
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VL_MODEL,
        default_gpu: int = 2,
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.model_path = model_path
        self.default_gpu = default_gpu
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self._llm: Optional[LLM] = None

    async def initialize(self) -> None:
        """Load the VLM onto GPU. Must be called before generate()."""
        if self._llm is not None:
            return

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.default_gpu)

        print(f"[VLMBackend] Loading {self.model_path} on GPU {self.default_gpu}...")

        self._llm = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 1},
        )

        print(f"[VLMBackend] Loaded successfully on GPU {self.default_gpu}.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response for a multimodal conversation.

        Supports two content formats:
        1. OpenAI-style list (recommended):
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/...base64"}},
                {"type": "text", "text": "What do you see?"},
            ]}]

        2. Legacy string (for backward compatibility):
            {"role": "user", "content": "<|vision_start|>data:image/...<|vision_end|>Question?"}
        """
        if self._llm is None:
            raise RuntimeError("VLMBackend not initialized. Call initialize() first.")

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Normalize content to list format
        normalized = self._normalize_content(messages)

        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )

        outputs = self._llm.chat(normalized, sampling_params=sampling)
        return outputs[0].outputs[0].text

    def _normalize_content(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize all message content to list-of-blocks format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Legacy format: check for vision tokens
                if "<|vision_start|>" in content:
                    result.append({
                        "role": role,
                        "content": self._parse_legacy_vision(content),
                    })
                else:
                    # Plain string — wrap in list
                    result.append({"role": role, "content": [{"type": "text", "text": content}]})

            elif isinstance(content, list):
                # Already list format — validate and pass through
                normalized_items = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            normalized_items.append({"type": "text", "text": str(item["text"])})
                        elif item.get("type") == "image_url":
                            normalized_items.append({
                                "type": "image_url",
                                "image_url": {"url": item["image_url"]["url"] if isinstance(item["image_url"], dict) else item["image_url"]}
                            })
                        else:
                            normalized_items.append(item)
                    else:
                        normalized_items.append({"type": "text", "text": str(item)})
                result.append({"role": role, "content": normalized_items})

            else:
                result.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

        return result

    def _parse_legacy_vision(self, content: str) -> List[Dict[str, Any]]:
        """Parse legacy <|vision_start|>data:image/...<|vision_end|>text<|vision_end|> format."""
        import re

        parts = []
        segments = re.split(r"<\|vision_start\|>|<\|vision_end\|>", content)

        for i, seg in enumerate(segments):
            if i % 2 == 0:
                # Text segment
                if seg.strip():
                    parts.append({"type": "text", "text": seg.strip()})
            else:
                # Image segment
                url = seg.strip()
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})

        return parts if parts else [{"type": "text", "text": content}]

    @property
    def is_initialized(self) -> bool:
        return self._llm is not None
