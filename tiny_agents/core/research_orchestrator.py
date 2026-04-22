"""ResearchOrchestrator — full pipeline for survey/review paper generation.

Pipeline:
  1. PlanningAgent   → generate section outline
  2. SearchAgent      → discover papers from arXiv/web
  3. ReaderAgent     → extract structured summaries from papers
  4. WriterPool       → write all sections in parallel
  5. SynthesizerAgent → merge sections + write intro/conclusion/abstract
  6. CitationAgent    → format references + BibTeX
  7. MarkdownWriterTool → output final .md file

Usage:
    orchestrator = ResearchOrchestrator()
    orchestrator.set_backend(vllm_backend)
    result = await orchestrator.run(
        topic="LoRA in Vision Models",
        num_sections=8,
        max_papers=50,
    )
    # result = {success, filepath, sections_written, papers_found, ...}
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.core.multi_gpu_writer_pool import MultiGPUWriterPool
from tiny_agents.tools import ArxivTool, MarkdownWriterTool


class ResearchOrchestrator:
    """Full pipeline orchestrator for survey paper generation."""

    def __init__(
        self,
        num_writers: int = 5,
        writer_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        planner_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        reader_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        synthesizer_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        citation_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_papers: int = 50,
        output_dir: str = "output",
    ):
        self.num_writers = num_writers
        self.writer_model = writer_model
        self.planner_model = planner_model
        self.reader_model = reader_model
        self.synthesizer_model = synthesizer_model
        self.citation_model = citation_model
        self.max_papers = max_papers
        self.output_dir = output_dir

        self.backend = None  # VLLMBackend, set via set_backend()
        self._planner = None
        self._searcher = None
        self._reader = None
        self._synthesizer = None
        self._citation = None
        self._writer_pool = None
        self._arxiv = ArxivTool()
        self._md_writer = MarkdownWriterTool(default_output_dir=output_dir)

    def set_backend(self, backend) -> None:
        """Inject VLLMBackend and configure all agents."""
        self.backend = backend

        from tiny_agents.agents.planner import PlanningAgent
        from tiny_agents.agents.searcher import SearchAgent
        from tiny_agents.agents.reader import ReaderAgent
        from tiny_agents.agents.synthesizer import SynthesizerAgent
        from tiny_agents.agents.citation_agent import CitationAgent

        self._planner = PlanningAgent(model_name="gpu0")
        self._planner.backend = backend

        self._searcher = SearchAgent(model_name="gpu0")
        self._searcher.backend = backend

        self._reader = ReaderAgent(model_name="gpu0")
        self._reader.backend = backend

        self._synthesizer = SynthesizerAgent(model_name="gpu0")
        self._synthesizer.backend = backend

        self._citation = CitationAgent(model_name="gpu0")
        self._citation.backend = backend

        # GPU assignment: writers distributed across GPUs 0, 1, 2
        # Each GPU gets ~2 writers sharing its KV cache
        num_gpus = 3
        gpu_assignment = [i % num_gpus for i in range(self.num_writers)]
        self._writer_pool = MultiGPUWriterPool(
            num_instances=self.num_writers,
            gpu_assignment=gpu_assignment,
            available_gpus=[0, 1, 2],
            model_name=self.writer_model,
        )
        self._writer_pool.set_backend(backend)

    async def run(
        self,
        topic: str,
        num_sections: int = 8,
        max_papers: int = None,
        context: Optional[SessionContext] = None,
    ) -> Dict[str, Any]:
        """Run the full survey generation pipeline.

        Returns:
            {success, filepath, title, sections_written, papers_found,
             papers_read, execution_time, ...}
        """
        start_time = time.time()
        max_papers = max_papers or self.max_papers
        session_id = f"survey_{int(start_time)}"

        # Use provided context or create a new one
        if context is None:
            context = SessionContext(session_id=session_id)
        else:
            context.session_id = session_id

        steps = []
        errors = []

        # ── Step 1: Planning ───────────────────────────────────────────
        print(f"[{session_id}] Step 1/7: Planning — topic: {topic}")
        step_start = time.time()
        try:
            planner_output: AgentOutput = await self._planner.run(
                {"topic": topic, "num_sections": num_sections}, context
            )
            outline = planner_output.payload.get("outline", {})
            sections = outline.get("sections", [])
            title = outline.get("title", f"Survey on {topic}")
        except Exception as e:
            errors.append(f"Planning failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        steps.append({"step": "planning", "time": time.time() - step_start, "sections": len(sections)})
        print(f"[{session_id}]   → {len(sections)} sections planned")

        if not sections:
            errors.append("No sections generated by planner")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 2: Search ───────────────────────────────────────────────
        print(f"[{session_id}] Step 2/7: Searching papers...")
        step_start = time.time()
        try:
            # Collect all keywords from sections
            all_keywords = []
            for sec in sections:
                kws = sec.get("keywords", [])
                if isinstance(kws, list):
                    all_keywords.extend(kws)

            search_output: AgentOutput = await self._searcher.run(
                {"topic": topic, "section_keywords": [s.get("keywords", []) for s in sections]},
                context,
            )
            papers = search_output.payload.get("papers", [])[:max_papers]
            arxiv_count = search_output.payload.get("arxiv_count", 0)
            web_count = search_output.payload.get("web_count", 0)
        except Exception as e:
            errors.append(f"Search failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        steps.append({"step": "search", "time": time.time() - step_start, "papers": len(papers)})
        print(f"[{session_id}]   → {len(papers)} papers found ({arxiv_count} arXiv, {web_count} web)")

        if not papers:
            errors.append("No papers found")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 3: Read ─────────────────────────────────────────────────
        print(f"[{session_id}] Step 3/7: Reading and extracting summaries...")
        step_start = time.time()
        try:
            reader_output: AgentOutput = await self._reader.run(
                {"papers": papers, "topic": topic}, context
            )
            summaries = reader_output.payload.get("summaries", [])
        except Exception as e:
            errors.append(f"Reading failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        steps.append({"step": "reading", "time": time.time() - step_start, "summaries": len(summaries)})
        print(f"[{session_id}]   → {len(summaries)} paper summaries extracted")

        if not summaries:
            errors.append("No summaries generated")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 4: Write sections in parallel ───────────────────────────
        print(f"[{session_id}] Step 4/7: Writing {len(sections)} sections with {self.num_writers} writers...")
        step_start = time.time()
        try:
            section_results = await self._writer_pool.write_all(
                sections=sections,
                summaries=summaries,
                topic=topic,
                context=context,
            )
        except Exception as e:
            errors.append(f"Section writing failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # Collect drafted sections
        drafted_sections = []
        for sec in sections:
            sid = sec.get("id", "?")
            if sid in section_results:
                drafted_sections.append({
                    "id": sid,
                    "title": sec.get("title", ""),
                    "content": section_results[sid].get("content", ""),
                    "papers_cited": section_results[sid].get("papers_cited", []),
                })

        steps.append({"step": "writing", "time": time.time() - step_start, "sections_written": len(drafted_sections)})
        print(f"[{session_id}]   → {len(drafted_sections)} sections drafted")

        if not drafted_sections:
            errors.append("No sections were drafted")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 5: Synthesize ────────────────────────────────────────────
        print(f"[{session_id}] Step 5/7: Synthesizing full paper...")
        step_start = time.time()
        try:
            synthesize_output: AgentOutput = await self._synthesizer.run(
                {
                    "sections": drafted_sections,
                    "topic": topic,
                    "title": title,
                },
                context,
            )
            full_paper = synthesize_output.payload.get("full_paper", "")
        except Exception as e:
            errors.append(f"Synthesis failed: {e}")
            full_paper = ""  # fallback: use section contents only

        steps.append({"step": "synthesis", "time": time.time() - step_start})
        print(f"[{session_id}]   → Synthesis complete ({len(full_paper)} chars)")

        # ── Step 6: Citation ──────────────────────────────────────────────
        print(f"[{session_id}] Step 6/7: Formatting references...")
        step_start = time.time()
        try:
            # Collect all cited papers
            all_papers = []
            for sec in drafted_sections:
                all_papers.extend(sec.get("papers_cited", []))

            citation_output: AgentOutput = await self._citation.run(
                {"papers": all_papers, "full_paper": full_paper}, context
            )
            references = citation_output.payload.get("references", "")
            bibtex = citation_output.payload.get("bibtex", "")
        except Exception as e:
            errors.append(f"Citation formatting failed: {e}")
            references = ""
            bibtex = ""

        steps.append({"step": "citation", "time": time.time() - step_start})
        print(f"[{session_id}]   → {citation_output.payload.get('count', 0)} references formatted")

        # ── Step 7: Write output file ────────────────────────────────────
        print(f"[{session_id}] Step 7/7: Writing output file...")
        step_start = time.time()

        # Assemble final paper
        date_str = datetime.now().strftime("%Y-%m-%d")
        final_content = f"# {title}\n\n*Generated: {date_str}*\n\n"
        final_content += f"*Topic: {topic}*\n\n"
        final_content += f"*Statistics: {len(sections)} sections, {len(papers)} papers found, "
        final_content += f"{len(summaries)} papers read, {len(drafted_sections)} sections drafted*\n\n"
        final_content += "---\n\n"

        if full_paper:
            final_content += full_paper
        else:
            # Fallback: just concatenate drafted sections
            for sec in drafted_sections:
                final_content += sec.get("content", "") + "\n\n"

        if references:
            final_content += "\n---\n\n" + references + "\n"
        if bibtex:
            final_content += "\n---\n\n## BibTeX\n\n" + bibtex + "\n"

        # Write to file
        filename = f"{self._sanitize_filename(topic)}_{date_str}.md"
        filepath = os.path.join(self.output_dir, filename)

        md_result = self._md_writer.execute({
            "content": final_content,
            "path": self.output_dir,
            "filename": filename,
            "mode": "write",
        })

        if md_result.success:
            filepath = md_result.output["filepath"]
        else:
            errors.append(f"File write failed: {md_result.error}")

        elapsed = time.time() - start_time
        steps.append({"step": "output", "time": time.time() - step_start, "filepath": filepath})

        print(f"[{session_id}] ✓ Complete in {elapsed:.1f}s — {filepath}")

        return {
            "success": len(errors) == 0,
            "session_id": session_id,
            "title": title,
            "topic": topic,
            "filepath": filepath,
            "sections_written": len(drafted_sections),
            "papers_found": len(papers),
            "papers_read": len(summaries),
            "papers_cited": len(set(p.get("title", "") for sec in drafted_sections for p in sec.get("papers_cited", []))),
            "execution_time": elapsed,
            "steps": steps,
            "errors": errors,
        }

    def _sanitize_filename(self, topic: str) -> str:
        """Convert topic to a safe filename."""
        import re
        name = re.sub(r"[^\w\s\-]", "", topic)
        name = re.sub(r"\s+", "_", name)
        return name[:50] or "survey"

    def _fail(self, session_id: str, errors: List[str], steps: List, elapsed: float) -> Dict[str, Any]:
        print(f"[{session_id}] ✗ Failed: {errors}")
        return {
            "success": False,
            "session_id": session_id,
            "errors": errors,
            "steps": steps,
            "execution_time": elapsed,
        }
