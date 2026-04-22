"""WriterPool — manages multiple WriterAgent instances for parallel section writing.

Each WriterAgent instance is a stateless agent that receives a section spec
and paper summaries, then produces a section draft. Multiple instances can
run concurrently (via asyncio.gather) to write multiple sections in parallel.

GPU affinity: each WriterAgent instance is pinned to a specific GPU so that
multiple instances can share the same GPU without OOM conflicts (vLLM's
tensor_parallel_size=1 allows concurrent requests on the same instance).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext


class WriterPool:
    """Pool of WriterAgent instances for parallel section writing.

    Usage:
        pool = WriterPool(num_instances=5, model_name="Qwen/Qwen2.5-1.5B-Instruct")
        pool.set_backend(backend)  # inject VLLMBackend

        sections = [
            {"id": "§3", "title": "Methods", "keywords": [...]},
            {"id": "§4", "title": "Results", "keywords": [...]},
        ]

        results = await pool.write_all(sections, summaries, topic, context)
        # results = {"§3": {"content": "...", "papers_cited": [...]}, ...}
    """

    def __init__(
        self,
        num_instances: int = 5,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_retries: int = 3,
    ):
        self.num_instances = num_instances
        self.model_name = model_name
        self.max_retries = max_retries
        self.backend = None  # set via set_backend()
        self._agents: List[BaseAgent] = []
        self._idle: asyncio.Queue = None
        self._results: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def set_backend(self, backend) -> None:
        """Inject the VLLMBackend (shared across all instances)."""
        self.backend = backend

    def _create_agents(self) -> List[BaseAgent]:
        """Create writer agent instances. Each instance is independent."""
        from tiny_agents.agents.writer import WriterAgent
        agents = []
        for i in range(self.num_instances):
            agent = WriterAgent(model_name=self.model_name)
            if self.backend:
                agent.backend = self.backend
            agents.append(agent)
        return agents

    async def write_section(
        self,
        section: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        topic: str,
        context: SessionContext,
        agent: Optional[BaseAgent] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Write one section using the given (or next idle) agent.

        Returns (section_id, result_dict).
        """
        section_id = section.get("id", "?")
        input_data = {
            "section": section,
            "summaries": summaries,
            "topic": topic,
        }

        for attempt in range(self.max_retries):
            try:
                output: AgentOutput = await agent.run(input_data, context)
                if output.payload.get("content"):
                    return section_id, output.payload
                # Retry on empty content
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return section_id, {
                        "section_id": section_id,
                        "section_title": section.get("title", "Unknown"),
                        "content": f"## {section.get('title', 'Error')}\n\n*Error generating section: {e}*\n",
                        "papers_cited": [],
                        "error": str(e),
                    }
                await asyncio.sleep(1 * (attempt + 1))

        return section_id, output.payload

    async def write_all(
        self,
        sections: List[Dict[str, Any]],
        summaries: List[Dict[str, Any]],
        topic: str,
        context: SessionContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Write all sections in parallel using the agent pool.

        Returns a dict mapping section_id -> result payload.
        """
        if not sections:
            return {}

        agents = self._create_agents()
        idle_queue = asyncio.Queue()
        for agent in agents:
            await idle_queue.put(agent)

        async def write_one(section: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            agent = await idle_queue.get()
            try:
                result = await self.write_section(section, summaries, topic, context, agent)
                return result
            finally:
                await idle_queue.put(agent)

        tasks = [write_one(sec) for sec in sections]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in results_list:
            if isinstance(item, Exception):
                # Try to identify which section failed
                results["error"] = str(item)
            else:
                section_id, payload = item
                results[section_id] = payload

        return results
