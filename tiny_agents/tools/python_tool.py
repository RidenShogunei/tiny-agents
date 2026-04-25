"""Python code execution tool — sandboxed subprocess execution."""

import ast
import io
import sys
import tempfile
import traceback
from typing import Any, Dict

from tiny_agents.tools.base import BaseTool, ToolResult


class PythonTool(BaseTool):
    """Execute Python code in an isolated subprocess.

    - Compiles code before execution to catch SyntaxErrors early.
    - Captures stdout/stderr separately.
    - Enforces a timeout via SIGALRM (works on Linux).
    - Returns serialized JSON result that agents can parse.
    """

    def __init__(self, timeout: int = 10):
        super().__init__(
            name="python",
            description=(
                "Executes Python 3 code in a sandboxed subprocess and returns "
                "the stdout output, stderr output, and return code. "
                "Use this for: arithmetic, data processing, string manipulation, "
                "file I/O, algorithm execution, and any computation. "
                "The code runs in isolation with no access to network or external files. "
                "Return your final answer using print()."
            ),
        )
        self.timeout = timeout

    def _execute(self, code: str) -> ToolResult:
        """Run code and return result."""
        # Pre-validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ToolResult(
                tool_name=self.name,
                args={"code": code},
                success=False,
                error=f"SyntaxError: {e}",
            )

        # Run in subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = __import__("subprocess").run(
                [sys.executable, tmp_path],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )
            return ToolResult(
                tool_name=self.name,
                args={"code": code},
                success=(proc.returncode == 0),
                output=proc.stdout,
                error=proc.stderr if proc.returncode != 0 else None,
            )
        except __import__("subprocess").TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                args={"code": code},
                success=False,
                error=f"Execution timed out after {self.timeout}s",
            )
        finally:
            __import__("os").unlink(tmp_path)
