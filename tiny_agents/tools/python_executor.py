"""Sandboxed Python code execution for agent tool use."""

import tempfile
import subprocess
import os
from typing import Any, Dict, Optional


class PythonExecutor:
    """Executes Python code in a temporary sandbox."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def run(self, code: str) -> Dict[str, Any]:
        """Execute code and return stdout, stderr, and status."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s",
                "returncode": -1,
            }
        finally:
            os.unlink(temp_path)
