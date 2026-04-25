"""Configuration utilities."""

import os
import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> str:
    """Return the absolute path to the project root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
