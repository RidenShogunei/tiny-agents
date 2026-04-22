"""Base agent class for Tiny Agents — stateless by design.

Key principle: agents are pure transformation functions.
They receive (input_data, context) and return AgentOutput.
All conversation history lives in SessionContext, not on the agent object.
This eliminates the need for manual reset() calls between problems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from tiny_agents.core.session import SessionContext


class AgentOutput(BaseModel):
    """Standardized agent output format returned by every agent.run()."""
    thought: str = ""
    action: str = "respond"   # respond | delegate | tool_call | review
    target_agent: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    finished: bool = True


class BaseAgent(ABC):
    """Base class for all agents.

    Agents MUST NOT hold conversation state internally.
    The SessionContext is the single source of truth for all history.

    Subclasses implement run(input_data, context) which should be a
    pure function: same inputs -> same outputs (no internal side-effects).
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        role_prompt: str,
        backend: Optional[Any] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
        self.backend = backend

    @abstractmethod
    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Execute one agent step.

        Args:
            input_data: task-specific payload from the orchestrator.
            context: SessionContext — contains messages, working_state, config.
                     The agent MUST NOT store references to context or mutate
                     context.session_id or context.config.

        Returns:
            AgentOutput with action, thought, payload, and optional target_agent.
        """
        raise NotImplementedError

    def _get_override(self, context: SessionContext, key: str, default: Any) -> Any:
        """Get a config override from context, falling back to agent default."""
        return context.config.get(key, default)
