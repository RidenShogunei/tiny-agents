"""Shared memory system for cross-agent knowledge persistence."""

from typing import Any, Dict, List, Optional
from collections import deque


class SharedMemory:
    """Three-tier memory: short-term buffer, working state, long-term store."""

    def __init__(
        self,
        short_term_limit: int = 10,
        working_state: Optional[Dict[str, Any]] = None,
    ):
        self.short_term: deque = deque(maxlen=short_term_limit)
        self.working_state: Dict[str, Any] = working_state or {}
        self.long_term: List[Dict[str, Any]] = []

    def add_short_term(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the short-term memory buffer."""
        self.short_term.append(entry)

    def get_short_term(self) -> List[Dict[str, Any]]:
        """Return short-term memory as a list."""
        return list(self.short_term)

    def get_session_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all short-term entries for a specific session."""
        return [e for e in self.short_term if e.get("session_id") == session_id]

    def get_agent_history(self, session_id: str, agent_name: str) -> List[Dict[str, Any]]:
        """Retrieve steps from a specific agent in a session."""
        return [
            e for e in self.short_term
            if e.get("session_id") == session_id and e.get("agent") == agent_name
        ]

    def update_working_state(self, key: str, value: Any) -> None:
        """Update a key in the working memory state."""
        self.working_state[key] = value

    def get_working_state(self) -> Dict[str, Any]:
        """Return the current working state."""
        return self.working_state.copy()

    def add_long_term(self, entry: Dict[str, Any]) -> None:
        """Add a summarized entry to long-term memory."""
        self.long_term.append(entry)

    def search_long_term(self, keyword: str) -> List[Dict[str, Any]]:
        """Simple keyword search in long-term memory."""
        results = []
        for entry in self.long_term:
            text = str(entry)
            if keyword.lower() in text.lower():
                results.append(entry)
        return results

    def clear(self) -> None:
        """Reset all memory tiers."""
        self.short_term.clear()
        self.working_state.clear()
        self.long_term.clear()
