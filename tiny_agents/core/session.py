"""Session context: single-source-of-truth for agent state per execution call.

All per-session state lives here instead of on the Agent object.
Orchestrator creates one SessionContext per execute() call and injects it
into every agent's run() call. This eliminates:
- agent.message_history accumulation across calls (no more manual reset())
- cross-problem context leakage
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import defaultdict


@dataclass
class SessionContext:
    """Immutable-ish container for all per-session state.

    Agents receive this object (not input_data alone) so they have
    full access to conversation history, shared memory, and config
    without holding any internal state.

    Design principles:
    - messages are per-agent (keyed by agent name) so agents can't
      accidentally read each other's private context unless the
      orchestrator explicitly shares via working_state.
    - The orchestrator controls routing; agents just produce output
      and optionally write to working_state.
    """

    session_id: str
    # Per-agent message lists. Key = agent name, value = list of {"role", "content"}
    messages: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    # Short-term: list of step records (orchestrator writes, agents can read)
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    # Working state: arbitrary key-value for cross-agent data sharing
    working_state: Dict[str, Any] = field(default_factory=dict)
    # Config overrides for this session (temperature, max_tokens, etc.)
    config: Dict[str, Any] = field(default_factory=dict)

    # Lazily-created tool registry for this session (set by Orchestrator)
    _tools: Optional[Any] = field(default=None)

    @property
    def tools(self) -> "ToolRegistry":
        """Tool registry available to agents. Set by Orchestrator via register_tool()."""
        if self._tools is None:
            from tiny_agents.tools.base import ToolRegistry
            object.__setattr__(self, '_tools', ToolRegistry())
        return self._tools

    # ── message API ──────────────────────────────────────────────

    def add_message(self, agent_name: str, role: str, content: str) -> None:
        """Append a message to an agent's conversation history."""
        self.messages[agent_name].append({"role": role, "content": content})

    def get_messages(self, agent_name: str) -> List[Dict[str, str]]:
        """Return full message list for an agent (includes system prompt)."""
        return list(self.messages[agent_name])

    def set_system_prompt(self, agent_name: str, system_prompt: str) -> None:
        """Set or overwrite the system prompt for an agent."""
        msgs = self.messages[agent_name]
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = system_prompt
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})

    def get_last_message(self, agent_name: str, role: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Return the last message from an agent (optionally filtered by role)."""
        msgs = self.messages[agent_name]
        if role:
            for m in reversed(msgs):
                if m["role"] == role:
                    return m
            return None
        return msgs[-1] if msgs else None

    def clear_messages(self, agent_name: Optional[str] = None) -> None:
        """Clear message history for one agent, or all if agent_name is None."""
        if agent_name:
            self.messages[agent_name].clear()
        else:
            self.messages.clear()

    # ── step tracking API ────────────────────────────────────────

    def add_step(self, step: Dict[str, Any]) -> None:
        """Record an orchestrator step (agent name, input, output)."""
        self.short_term.append(step)

    def get_agent_steps(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all steps executed by a specific agent in this session."""
        return [s for s in self.short_term if s.get("agent") == agent_name]

    # ── working state API ─────────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        """Store cross-agent shared data."""
        self.working_state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve cross-agent shared data."""
        return self.working_state.get(key, default)
