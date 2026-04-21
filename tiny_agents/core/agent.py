"""Base agent class for Tiny Agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AgentOutput(BaseModel):
    """Standardized agent output format."""
    thought: str
    action: str
    target_agent: Optional[str] = None
    payload: Dict[str, Any] = {}
    finished: bool = False


class BaseAgent(ABC):
    """Base class for all agents in the framework."""

    def __init__(
        self,
        name: str,
        model_name: str,
        role_prompt: str,
        memory: Optional[Any] = None,
        backend: Optional[Any] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
        self.memory = memory
        self.backend = backend
        self.bus = None  # injected by orchestrator
        self.message_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Execute one agent step. Must be implemented by subclasses."""
        pass

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the agent's local history."""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.message_history.append(msg)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return conversation history for LLM prompt construction."""
        system_msg = {"role": "system", "content": self.role_prompt}
        return [system_msg] + self.message_history

    def reset(self) -> None:
        """Clear message history."""
        self.message_history.clear()

    def _call_llm(self, user_message: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Call the backend LLM with the current context + new user message."""
        if self.backend is None:
            raise RuntimeError(f"Agent '{self.name}' has no backend configured")
        messages = self.get_messages()
        messages.append({"role": "user", "content": user_message})
        return self.backend.generate(
            model_key=self.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
