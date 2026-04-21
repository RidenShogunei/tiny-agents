"""Download required Qwen models from ModelScope using aria2c.

Usage:
    python scripts/download_models.py

Requires aria2c to be installed.
"""

import os
import subprocess
import sys
from pathlib import Path

# ModelScope base URL
MS_BASE = "https://modelscope.cn/models"

# Models to download
MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "generation_config.json",
            "chat_template.json",
        ]
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "generation_config.json",
            "chat_template.json",
        ]
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "generation_config.json",
            "chat_template.json",
        ]
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "generation_config.json",
            "chat_template.json",
        ]
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "generation_config.json",
            "chat_template.json",
            "preprocessor_config.json",
        ]
    },
}

CACHE_DIR = os.path.expanduser("~/.cache/tiny-agents/models")


def download_file(model_id: str, filename: str, out_dir: str) -> bool:
    """Download a single file using aria2c."""
    url = f"{MS_BASE}/{model_id}/resolve/master/{filename}"
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        print(f"  [SKIP] {filename} already exists")
        return True

    print(f"  [DOWNLOAD] {filename}")
    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "--max-connection-per-server=16",
        "--timeout=60",
        "--max-tries=3",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "-d", out_dir,
        "-o", filename,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Check if it was a 404 (file may not exist for this model variant)
        if "errorCode=3" in result.stderr or "404" in result.stderr:
            print(f"  [WARN] {filename} not found (404), skipping")
            return True
        print(f"  [ERROR] Failed to download {filename}: {result.stderr[:200]}")
        return False

    print(f"  [OK] {filename}")
    return True


def download_model(model_id: str, config: dict) -> bool:
    """Download all files for a model."""
    print(f"\n[Model] {model_id}")
    out_dir = os.path.join(CACHE_DIR, model_id)
    os.makedirs(out_dir, exist_ok=True)

    success = True
    for filename in config["files"]:
        if not download_file(model_id, filename, out_dir):
            success = False

    return success


def main():
    """Download all required models."""
    # Check aria2c
    if subprocess.run(["which", "aria2c"], capture_output=True).returncode != 0:
        print("Error: aria2c not found. Please install it first.")
        sys.exit(1)

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Models to download: {len(MODELS)}")

    for model_id, config in MODELS.items():
        download_model(model_id, config)

    print("\n" + "=" * 50)
    print("Download process completed.")
    print(f"Models cached at: {CACHE_DIR}")


if __name__ == "__main__":
    main()
