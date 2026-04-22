"""Tools package — universal tool protocol for the agent framework.

Generic tools (usable by any agent/pipeline) are exported here.
Survey-specific tools (ArxivTool, OpenAlexTool, etc.) live in tiny_agents.survey.tools.
"""

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
from tiny_agents.tools.cluster_tool import ClusterTool

__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolParam",
    "ToolResult",
    "ToolRegistry",
    "PythonTool",
    "CalculatorTool",
    "WebSearchTool",
    "ClusterTool",
    "get_registry",
    "register_tool",
    "tool_schema",
]
