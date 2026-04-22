"""Tools package — universal tool protocol."""

from tiny_agents.tools.base import (
    BaseTool,
    ToolDefinition,
    ToolParam,
    ToolResult,
    ToolRegistry,
    get_registry,
    register_tool,
    tool_schema,
)

from tiny_agents.tools.python_tool import PythonTool
from tiny_agents.tools.calculator_tool import CalculatorTool

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParam",
    "ToolResult",
    "ToolRegistry",
    "PythonTool",
    "CalculatorTool",
    "get_registry",
    "register_tool",
    "tool_schema",
]
