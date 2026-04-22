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
from tiny_agents.tools.web_search_tool import WebSearchTool
from tiny_agents.tools.arxiv_tool import ArxivTool
from tiny_agents.tools.markdown_writer_tool import MarkdownWriterTool

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParam",
    "ToolResult",
    "ToolRegistry",
    "PythonTool",
    "CalculatorTool",
    "WebSearchTool",
    "ArxivTool",
    "MarkdownWriterTool",
    "get_registry",
    "register_tool",
    "tool_schema",
]
