"""Lightweight async message bus for inter-agent communication."""

import asyncio
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque


class MessageBus:
    """Async message bus with topic-based routing."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a message to a topic."""
        self._history[topic].append(message)
        await self._queues[topic].put(message)

    async def subscribe(
        self,
        topic: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Subscribe and wait for a message on a topic."""
        try:
            return await asyncio.wait_for(
                self._queues[topic].get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    def get_history(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent message history for a topic."""
        hist = self._history.get(topic, deque())
        return list(hist)[-limit:]

    def clear_topic(self, topic: str) -> None:
        """Clear a topic's queue and history."""
        if topic in self._queues:
            while not self._queues[topic].empty():
                try:
                    self._queues[topic].get_nowait()
                except asyncio.QueueEmpty:
                    break
        if topic in self._history:
            self._history[topic].clear()
