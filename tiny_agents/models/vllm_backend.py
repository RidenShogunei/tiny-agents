"""vLLM inference backend for Tiny Agents."""

import os
from typing import Any, Dict, List, Optional

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VLLMBackend:
    """Manages vLLM instances for multiple agents.

    Each model is loaded once and reused across requests.
    Agents with the same base model share the vLLM instance.
    """

    def __init__(self, default_gpu: int = 0):
        self.default_gpu = default_gpu
        self.instances: Dict[str, LLM] = {}
        self._model_path_map: Dict[str, str] = {}  # model_key -> resolved path
        self._path_to_key: Dict[str, str] = {}     # resolved path -> first model_key
        self._key_gpu: Dict[str, int] = {}         # model_key -> gpu
        self.cache_dir = os.path.expanduser("~/.cache/tiny-agents/models")

    def _resolve_path(self, model_name: str) -> str:
        """Resolve a HuggingFace model ID to local path."""
        # If it's a local path, use it directly
        if os.path.exists(model_name):
            return model_name

        # Otherwise, look in our cache directory
        local_path = os.path.join(self.cache_dir, model_name)
        if os.path.exists(local_path):
            return local_path

        # Fallback: let vLLM handle it (will try HF hub)
        return model_name

    def load_model(
        self,
        model_key: str,
        model_name: str,
        gpu: Optional[int] = None,
        max_model_len: int = 32768,
        **kwargs,
    ) -> None:
        """Load a model into a vLLM instance. Share instances for same path+gpu."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")

        if model_key in self.instances:
            return

        model_path = self._resolve_path(model_name)
        gpu = gpu if gpu is not None else self.default_gpu

        # Check if same model path is already loaded on the same GPU
        existing_key = self._path_to_key.get(model_path)
        if existing_key and self._key_gpu.get(existing_key) == gpu:
            print(f"[VLLM] Reusing existing instance for {model_name} (shared with '{existing_key}')")
            self.instances[model_key] = self.instances[existing_key]
            self._model_path_map[model_key] = model_path
            self._key_gpu[model_key] = gpu
            return

        # vLLM controls GPU via env var
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"[VLLM] Loading {model_name} (gpu={gpu})...")
        self.instances[model_key] = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=kwargs.pop("gpu_memory_utilization", 0.45),  # 0.45 * 40GB = 18GB
            max_model_len=max_model_len,
            trust_remote_code=True,
            enable_prefix_caching=True,  # explicit for clarity
            **kwargs,
        )
        print(f"[VLLM] {model_name} loaded")
        self._model_path_map[model_key] = model_path
        self._path_to_key[model_path] = model_key
        self._key_gpu[model_key] = gpu

    def generate(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text from a conversation using the model's chat template."""
        resolved_key = model_key
        if resolved_key not in self.instances:
            # Smart GPU suffix resolution: writer_gpu3 → gpu3
            import re
            m = re.match(r"^(.*?)_gpu(\d+)$", resolved_key)
            if m:
                base_key = f"gpu{m.group(2)}"
                if base_key in self.instances:
                    resolved_key = base_key

            # Path-based fallback: resolve model_key as a model name/path
            if resolved_key not in self.instances:
                resolved = self._resolve_path(model_key)
                for k, v in self._model_path_map.items():
                    if v == resolved:
                        resolved_key = k
                        break

            if resolved_key not in self.instances:
                raise ValueError(
                    f"Model '{model_key}' not loaded. "
                    f"Available keys: {list(self.instances.keys())}"
                )

        # Extract chat_template_kwargs before passing to SamplingParams
        chat_template_kwargs = kwargs.pop("chat_template_kwargs", None)

        # Use the model's chat template (via .chat()) for correct formatting
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
            **kwargs,
        )

        outputs = self.instances[resolved_key].chat(
            messages,
            sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        return outputs[0].outputs[0].text.strip()

    def unload(self, model_key: str) -> None:
        """Unload a model to free GPU memory."""
        if model_key in self.instances:
            del self.instances[model_key]

    def unload_all(self) -> None:
        """Unload all models."""
        self.instances.clear()
