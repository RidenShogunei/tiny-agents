"""Universal Tool protocol — clean separation between agents and tools.

Design:
- Tool = a callable with a JSON Schema that describes its interface.
- ToolRegistry = global registry of available tools.
- Agent returns action="tool_call" with {tool_name, args};
  Orchestrator looks up the tool, executes it, and injects result
  back into context as a user message, then calls agent again to continue.
- This mirrors OpenAI function calling / Anthropic tool use but is
  backend-agnostic and works with any LLM that can output structured JSON.

Example usage:
    tool_registry.register(PythonTool(timeout=10))
    tool_registry.register(CalculatorTool())

    tool_schema = tool_registry.get_schema()  # → list of tool definitions
    # Pass schema to LLM so it knows what tools are available
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


# ── Schema types ────────────────────────────────────────────────────────────

@dataclass
class ToolParam:
    """A single parameter of a tool."""
    type: str = "string"          # string | number | integer | boolean | object | array
    description: str = ""
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_schema(self) -> Dict[str, Any]:
        s = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = self.enum
        if self.default is not None:
            s["default"] = self.default
        return s


@dataclass
class ToolDefinition:
    """JSON Schema representation of one tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema "parameters": {}

    @staticmethod
    def from_callable(func: Callable) -> ToolDefinition:
        """Derive a ToolDefinition from a Python function's type hints and docstring."""
        sig = inspect.signature(func)
        hints = {}
        try:
            hints = sig.parameters
        except Exception:
            pass

        properties = {}
        required = []
        for pname, param in hints.items():
            if pname in ("self", "cls"):
                continue
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "string"
            if annotation is bool:
                ptype = "boolean"
            elif annotation in (int, "int"):
                ptype = "integer"
            elif annotation in (float, "float"):
                ptype = "number"
            elif annotation in (list, "list", List, "array"):
                ptype = "array"
            elif annotation in (dict, "dict", Dict, "object"):
                ptype = "object"
            else:
                ptype = "string"

            properties[pname] = {
                "type": ptype,
                "description": f"Parameter {pname}",
            }
            if param.default is not inspect.Parameter.empty:
                properties[pname]["default"] = param.default
            else:
                required.append(pname)

        return ToolDefinition(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )


# ── Tool result ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    args: Dict[str, Any]
    success: bool
    output: Any = None
    error: Optional[str] = None


# ── Base Tool ────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """Every tool inherits from BaseTool and implements _execute()."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def _execute(self, **kwargs) -> ToolResult:
        """Implement tool logic. Raises are caught by execute()."""
        raise NotImplementedError

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Public API — catches exceptions and returns a ToolResult."""
        try:
            return self._execute(**args)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                args=args,
                success=False,
                error=str(e),
            )

    def to_definition(self) -> ToolDefinition:
        """Return the JSON Schema definition for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={},
        )


# ── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """Central registry for all tools available to agents.

    Usage:
        registry = ToolRegistry()
        registry.register(PythonTool(timeout=10))
        registry.register(CalculatorTool())

        # Get schema for LLM prompt
        schema = registry.get_schema()
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_schema(self) -> List[Dict[str, Any]]:
        """Return the tool list as a JSON-serializable list of definitions.

        Matches OpenAI function-calling schema format.
        """
        definitions = []
        for tool in self._tools.values():
            defn = tool.to_definition()
            definitions.append({
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": defn.parameters or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            })
        return definitions

    def list_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())


# ── Global default registry ─────────────────────────────────────────────────

_default_registry: Optional[ToolRegistry] = None

def get_registry() -> ToolRegistry:
    """Get the global default registry (lazy init)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry

def register_tool(tool: BaseTool) -> None:
    """Convenience wrapper for global registry."""
    get_registry().register(tool)

def tool_schema() -> List[Dict[str, Any]]:
    """Convenience wrapper for global registry schema."""
    return get_registry().get_schema()
