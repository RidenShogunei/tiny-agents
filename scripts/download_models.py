"""Download required Qwen models for Tiny Agents.

Usage:
    python scripts/download_models.py
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "llm",
    "Qwen/Qwen2.5-1.5B-Instruct": "llm",
    "Qwen/Qwen2.5-3B-Instruct": "llm",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "llm",
    "Qwen/Qwen2.5-VL-3B-Instruct": "vlm",
}

CACHE_DIR = os.path.expanduser("~/.cache/tiny-agents/models")


def download_model(model_name: str, model_type: str) -> None:
    """Download a model and its tokenizer."""
    print(f"\n[Downloading] {model_name} ({model_type})")
    print("-" * 50)

    # Always download tokenizer
    AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)

    if model_type == "vlm":
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
    else:
        AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

    print(f"[Done] {model_name}")


def main():
    """Download all required models."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Models will be cached to: {CACHE_DIR}")
    print(f"Total models to download: {len(MODELS)}")

    for model_name, model_type in MODELS.items():
        try:
            download_model(model_name, model_type)
        except Exception as e:
            print(f"[Error] Failed to download {model_name}: {e}")

    print("\n" + "=" * 50)
    print("All downloads completed (or attempted).")


if __name__ == "__main__":
    main()
