"""MultiGPUWriterPool — distribute WriterAgent instances across multiple GPUs.

Solves the KV cache contention problem from WriterPool (all writers on 1 GPU
competing for the same 30GB pool). By splitting writers across GPUs, each
writer gets its own KV cache pool, enabling true parallelism.

GPU layout (default, 5 writers):
  GPU 0: writer_0, writer_1, writer_2  (3 instances, shared KV cache)
  GPU 1: writer_3, writer_4            (2 instances, shared KV cache)

Each GPU loads one Qwen2.5-1.5B model (≈3.3GB), leaving ~6GB for KV cache
per GPU at gpu_memory_utilization=0.85.

Usage:
    pool = MultiGPUWriterPool(
        num_instances=6,
        gpu_assignment=[0, 0, 0, 1, 1, 1],  # 3 per GPU
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    pool.set_backend(backend)
    results = pool.write_sections(specs)  # parallel
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tiny_agents.core.session import SessionContext
from tiny_agents.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class SectionSpec:
    """Specification for one section to be written."""
    section_id: str
    title: str
    keywords: List[str]
    paper_summaries: List[Dict[str, Any]]
    max_length: int = 600


@dataclass
class SectionResult:
    """Result of writing one section."""
    section_id: str
    title: str
    content: str = ""
    success: bool = False
    error: str = ""
    papers_used: List[str] = field(default_factory=list)
    duration: float = 0.0


class MultiGPUWriterPool:
    """Writer pool spread across multiple GPUs.

    Args:
        num_instances: Total number of writer instances to create.
        gpu_assignment: List of GPU indices (same length as num_instances).
                       Defaults to round-robin across available_gpus.
        available_gpus: GPU indices to use for round-robin distribution.
        model_name: Model path key for VLLMBackend.load_model().
        max_retries: Number of retry attempts per section on failure.
        timeout_per_section: Max seconds to wait for one section.
    """

    def __init__(
        self,
        num_instances: int = 5,
        gpu_assignment: Optional[List[int]] = None,
        available_gpus: Optional[List[int]] = None,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_retries: int = 1,
        timeout_per_section: int = 120,
    ):
        self.num_instances = num_instances
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_per_section = timeout_per_section

        # Default GPU assignment: round-robin across available_gpus
        if gpu_assignment is None:
            gpus = available_gpus or [0, 1, 2]
            gpu_assignment = [gpus[i % len(gpus)] for i in range(num_instances)]
        self.gpu_assignment = gpu_assignment

        self.backend = None
        self._agents: List[Any] = []
        self._loaded_keys: Dict[int, str] = {}  # gpu_index -> backend model key
        self._lock = asyncio.Lock()

    def set_backend(self, backend) -> None:
        """Set the VLLMBackend and load writer models on assigned GPUs."""
        self.backend = backend
        self._agents = []
        self._loaded_keys = {}

        # Group instances by GPU to minimize model loading
        gpu_to_instances: Dict[int, List[int]] = {}
        for i, gpu in enumerate(self.gpu_assignment):
            gpu_to_instances.setdefault(gpu, []).append(i)

        for gpu, instance_indices in gpu_to_instances.items():
            # Each GPU gets one model key; all its writers share it
            key = f"writer_gpu{gpu}"
            if key not in self.backend.instances:
                logger.info(f"[MultiGPUWriterPool] Loading {self.model_name} on GPU {gpu} as '{key}'")
                self.backend.load_model(key, self.model_name, gpu=gpu)
            else:
                logger.info(f"[MultiGPUWriterPool] Reusing existing '{key}' on GPU {gpu}")

            self._loaded_keys[gpu] = key

            # Create one WriterAgent per instance, all pointing to same model key
            for idx in instance_indices:
                agent = self._create_agent(key)
                self._agents.append(agent)

        logger.info(f"[MultiGPUWriterPool] {len(self._agents)} writers across {len(gpu_to_instances)} GPUs")

    def _create_agent(self, model_key: str):
        """Create a single writer agent."""
        from tiny_agents.agents.writer import WriterAgent
        agent = WriterAgent(model_name=self.model_name)
        if self.backend:
            agent.backend = self.backend
        return agent

    def _get_agent(self, section_idx: int):
        """Get the agent for a given section index (round-robin)."""
        return self._agents[section_idx % len(self._agents)]

    async def write_all(
        self,
        sections: list,
        summaries: list,
        topic: str,
        context,
    ) -> dict:
        """Write all sections in parallel using the writer pool.

        Mirrors the WriterPool.write_all() API for compatibility with
        ResearchOrchestrator.

        Args:
            sections: list of section dicts with 'id', 'title', 'keywords'
            summaries: list of paper summary dicts
            topic: research topic
            context: SessionContext (not used directly; each writer gets a fresh context)

        Returns:
            dict mapping section_id -> result payload dict
        """
        if not sections:
            return {}

        # Build SectionSpec list
        specs = []
        for sec in sections:
            section_id = sec.get("id", f"section_{len(specs)}")
            # Match papers to this section's keywords
            keywords = sec.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            matched = self._match_papers(summaries, keywords)
            max_len = sec.get("max_words", 800)
            specs.append(SectionSpec(
                section_id=section_id,
                title=sec.get("title", ""),
                keywords=keywords,
                paper_summaries=matched,
                max_length=max_len,
            ))

        # Run in thread pool (writers are synchronous)
        results = self.write_sections(specs)

        # Convert List[SectionResult] -> dict mapping section_id -> payload
        output = {}
        for r in results:
            output[r.section_id] = {
                "content": r.content,
                "papers_used": r.papers_used,
                "success": r.success,
                "error": r.error,
            }
        return output

    def _match_papers(self, summaries: list, keywords: list) -> list:
        """Select papers relevant to given keywords."""
        if not keywords or not summaries:
            return summaries[:5] if summaries else []
        kw_lower = [k.lower() for k in keywords]
        scored = []
        for p in summaries:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            score = sum(1 for kw in kw_lower if kw in text)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:5]]

    def write_sections(self, specs: List[SectionSpec]) -> List[SectionResult]:
        """Write multiple sections in parallel using the writer pool.

        This is the main entry point — call this instead of individual write().

        Args:
            specs: List of section specifications.

        Returns:
            List of SectionResult in the same order as specs.
        """
        if not self._agents:
            return [
                SectionResult(
                    section_id=s.section_id,
                    title=s.title,
                    success=False,
                    error="No backend set — call set_backend() first",
                )
                for s in specs
            ]

        # Dispatch all sections to the pool concurrently
        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            futures = [
                executor.submit(self._write_one_with_retry, spec)
                for spec in specs
            ]
            results = [f.result() for f in futures]

        return results

    def _write_one_with_retry(self, spec: SectionSpec) -> SectionResult:
        """Write a single section with retry on failure."""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                return self._write_one(spec)
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[Writer-{spec.section_id}] attempt {attempt+1} failed: {e}"
                )
        return SectionResult(
            section_id=spec.section_id,
            title=spec.title,
            success=False,
            error=last_error,
        )

    def _write_one(self, spec: SectionSpec) -> SectionResult:
        """Write one section using an assigned writer agent."""
        start = time.time()
        agent = self._get_agent(hash(spec.section_id) % len(self._agents))

        # Build input for the writer agent
        papers_text = "\n\n".join(
            f"[{p.get('paper_id', '?')}] {p.get('title', '')}\n"
            f"  Authors: {p.get('authors', 'Unknown')}\n"
            f"  Summary: {p.get('summary', p.get('abstract', ''))[:500]}"
            for p in spec.paper_summaries
        )

        input_data = {
            "section_title": spec.title,
            "keywords": ", ".join(spec.keywords),
            "paper_summaries": papers_text,
            "max_length": spec.max_length,
        }

        # Create a fresh session context for this write
        session_id = f"write_{spec.section_id}"
        context = SessionContext(
            session_id=session_id,
            metadata={
                "task": "section_write",
                "section_id": spec.section_id,
                "model": self.model_name,
            },
        )

        output = agent.run(input_data, context)

        content = ""
        papers_used = []
        if isinstance(output, dict):
            content = output.get("section_content", output.get("content", ""))
            papers_used = output.get("papers_used", [])
        elif isinstance(output, str):
            content = output

        # Extract citations if found in content
        if not papers_used:
            import re
            citations = re.findall(r'\[([^\]]+)\]', content)
            papers_used = list(set(citations))

        return SectionResult(
            section_id=spec.section_id,
            title=spec.title,
            content=content,
            success=True,
            papers_used=papers_used,
            duration=time.time() - start,
        )
