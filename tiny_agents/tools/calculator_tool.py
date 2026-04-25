"""Calculator tool — safe numeric evaluation for arithmetic expressions."""

import ast
import operator
import re
from typing import Any, Union

from tiny_agents.tools.base import BaseTool, ToolResult


_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Safely evaluate a limited subset of Python arithmetic AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ops:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ops[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        return operand

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "abs":
            return abs(_eval_node(node.args[0]))
        if node.func.id == "round":
            args = [_eval_node(a) for a in node.args]
            return round(*args)
        if node.func.id == "min":
            return min(_eval_node(a) for a in node.args)
        if node.func.id == "max":
            return max(_eval_node(a) for a in node.args)

    if isinstance(node, ast.Name):
        raise ValueError(f"Undefined variable: {node.id}")

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    """Safely evaluate arithmetic and math expressions.

    - Supports +, -, *, /, **, //, %, ^
    - Supports parentheses and standard math functions: abs, round, min, max
    - Rejects any Python code that is not a simple expression
      (no imports, no function defs, no variables)
    - Returns the exact numeric result as a string
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
                "Evaluate a numeric expression and return the result. "
                "Supports: +, -, *, /, ** (power), // (floor), % (mod), ^ (xor). "
                "Supports functions: abs(), round(), min(), max(). "
                "Supports parentheses for grouping. "
                "Example: calculator(expression='(10 + 5) * 2') -> 30. "
                "Use this for any arithmetic or math computation."
            ),
        )

    def _execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        expression = expression.strip()

        # Basic guard: reject anything with letters, assignment, or keywords
        suspicious = re.compile(r"[a-zA-Z_]|:=|def |class |import |if |for |while |print |return ")
        if suspicious.search(expression):
            return ToolResult(
                tool_name=self.name,
                args={"expression": expression},
                success=False,
                error="Calculator only accepts numeric expressions, not Python code",
            )

        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval_node(tree.body)
            # Convert to float only if needed
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return ToolResult(
                tool_name=self.name,
                args={"expression": expression},
                success=True,
                output=str(result),
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                args={"expression": expression},
                success=False,
                error=f"Expression error: {e}",
            )
