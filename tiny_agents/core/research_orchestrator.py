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
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.core.multi_gpu_writer_pool import MultiGPUWriterPool
from tiny_agents.core.writer_pool_v2 import MultiGPUWriterPoolV2
from tiny_agents.core.knowledge_graph import KnowledgeGraph
from tiny_agents.tools import ArxivTool, MarkdownWriterTool
from tiny_agents.tools.openalex_tool import OpenAlexTool


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
        self._openalex = OpenAlexTool(email="tiny-agents@research.ai")

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

        # GPU assignment: writers on GPU 0/1/2 only (1.5B models)
        # GPU 3 is reserved for 9B writer/critic loaded externally in demo
        writer_gpu_assignment = [i % 3 for i in range(self.num_writers)]
        self._writer_pool = MultiGPUWriterPool(
            num_instances=self.num_writers,
            gpu_assignment=writer_gpu_assignment,
            available_gpus=[0, 1, 2],
            model_name=self.writer_model,
        )
        # V2 writer pool with critic + revision loop
        self._writer_pool_v2 = MultiGPUWriterPoolV2(
            num_instances=self.num_writers,
            gpu_assignment=writer_gpu_assignment,
            available_gpus=[0, 1, 2, 3],
            model_name=self.writer_model,
            critic_model_key="gpu3",
            revision_threshold=6.0,
            max_revision_attempts=2,
        )
        self._writer_pool_v2.set_backend(backend)

        # Knowledge graph for cross-reference tracking (Direction B)
        self._knowledge_graph = KnowledgeGraph()
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

        # ── Step 2: Search via arXiv (real papers + real abstracts) ──────────
        print(f"[{session_id}] Step 2/7: Searching papers via arXiv...")
        step_start = time.time()
        try:
            # Collect all keywords from sections
            all_keywords = []
            for sec in sections:
                kws = sec.get("keywords", [])
                if isinstance(kws, list):
                    all_keywords.extend(kws)

            papers = await self._search_papers_arxiv(all_keywords, max_papers)
        except Exception as e:
            errors.append(f"arXiv search failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        steps.append({"step": "search", "time": time.time() - step_start, "papers": len(papers)})
        print(f"[{session_id}]   → {len(papers)} real papers found via arXiv")

        if not papers:
            errors.append("No papers found")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 2.5: Enrich papers with real metadata (OpenAlex citation counts) ─
        print(f"[{session_id}] Step 2.5/7: Enriching papers with real citation counts...")
        step_enrich_start = time.time()
        try:
            papers = await self._enrich_papers_openalex(papers)
        except Exception as e:
            errors.append(f"Paper enrichment failed: {e}")
            # Non-fatal: continue with original papers
        steps.append({"step": "enrich", "time": time.time() - step_enrich_start, "papers": len(papers)})
        print(f"[{session_id}]   → {len(papers)} papers enriched with citation counts")

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

        # ── Step 3.5: Build knowledge graph ────────────────────────────────
        # Populate graph with papers (Direction B: cross-reference tracking)
        print(f"[{session_id}] Step 3.5/7: Building paper knowledge graph...")
        step_kg_start = time.time()
        for paper in summaries:
            self._knowledge_graph.add_paper(paper)
        kg_stats = self._knowledge_graph.get_paper_stats()
        print(f"[{session_id}]   → {kg_stats['total_papers']} papers in graph "
              f"({len(kg_stats.get('categories', {}))} categories, "
              f"{len(kg_stats.get('methods', {}))} methods)")
        steps.append({"step": "knowledge_graph", "time": time.time() - step_kg_start, **kg_stats})

        # ── Step 4: Write sections with critique + revision ──────────────
        print(f"[{session_id}] Step 4/7: Writing {len(sections)} sections "
              f"(critic threshold={self._writer_pool_v2.revision_threshold}, "
              f"max {self._writer_pool_v2.max_revision_attempts} rewrites)...")
        step_start = time.time()
        try:
            # Inject knowledge graph for overlap detection
            self._writer_pool_v2.set_knowledge_graph(self._knowledge_graph)
            section_results = await self._writer_pool_v2.write_all_v2(
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
                result = section_results[sid]
                drafted_sections.append({
                    "id": sid,
                    "title": sec.get("title", ""),
                    "content": result.get("content", ""),
                    "papers_cited": result.get("papers_used", []),  # NOTE: writer pool returns papers_used
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
            # Pass the full paper list collected from search results
            citation_output: AgentOutput = await self._citation.run(
                {"papers": papers, "full_paper": full_paper}, context
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

        date_str = datetime.now().strftime("%Y-%m-%d")
        final_content = f"# {title}\n\n*Generated: {date_str}*\n\n"
        final_content += f"*Topic: {topic}*\n\n"
        final_content += f"*Statistics: {len(sections)} sections planned, {len(papers)} papers found, "
        final_content += f"{len(summaries)} papers read, {len(drafted_sections)} sections drafted*\n\n"
        final_content += "---\n\n"

        # ── Assemble paper: Abstract/Conclusion from synthesizer + drafted sections + skip intro ──
        if full_paper:
            import re as _re

            # Extract Abstract: first section (before any ## header)
            first_section = _re.search(r'^(.*?)^##', full_paper, _re.DOTALL | _re.MULTILINE)
            if first_section:
                abstract_block = first_section.group(1).strip()
                # Take first 300 words only
                abstract_words = abstract_block.split()[:300]
                abstract_text = " ".join(abstract_words)
                final_content += "### Abstract\n\n" + abstract_text + "\n\n---\n\n"

            # Use our drafted sections for body content
            for sec in sorted(drafted_sections, key=lambda s: s.get("id", "")):
                sec_content = sec.get("content", "")
                if sec_content.strip():
                    final_content += sec_content.strip() + "\n\n---\n\n"

            # Extract Conclusion: last section (after last ## before references)
            conclusion_match = _re.search(
                r'(?i)conclusion[:\s]*\n(.*?)$', full_paper, _re.DOTALL | _re.MULTILINE
            )
            if conclusion_match:
                conclusion_text = conclusion_match.group(1).strip()
                conclusion_words = conclusion_text.split()[:300]
                conclusion_text = " ".join(conclusion_words)
                final_content += "### Conclusion\n\n" + conclusion_text + "\n\n"
        else:
            for sec in sorted(drafted_sections, key=lambda s: s.get("id", "")):
                final_content += sec.get("content", "").strip() + "\n\n---\n\n"

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

    async def _search_papers_arxiv(self, keywords: List[str], max_papers: int) -> List[Dict]:
        """Search arXiv for papers matching topic keywords.

        arXiv gives us real titles, abstracts, authors, and DOIs.
        We enrich with OpenAlex citation counts in Step 2.5.
        """
        import arxiv as _arxiv

        seen_ids: Dict[str, bool] = {}
        results: List[Dict] = []

        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(keywords))[:10]

        client = _arxiv.Client()

        for kw in unique_keywords:
            if len(results) >= max_papers:
                break

            try:
                search = _arxiv.Search(
                    query=kw,
                    max_results=20,
                    sort_by=_arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    paper_id = result.entry_id
                    if paper_id in seen_ids:
                        continue
                    seen_ids[paper_id] = True

                    # Extract DOI if available
                    doi = ""
                    if result.doi:
                        doi = result.doi
                    else:
                        # Extract arXiv ID from entry_id
                        # e.g. https://arxiv.org/abs/2312.12345 -> 2312.12345
                        import re as _re
                        m = _re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+)', result.entry_id)
                        if m:
                            doi = f"10.48550/arXiv.{m.group(1)}"

                    results.append({
                        "title": result.title or "Unknown",
                        "authors": [a.name for a in (result.authors or [])],
                        "year": result.published.year if result.published else None,
                        "abstract": result.summary or "",
                        "doi": doi,
                        "paper_url": result.entry_id or "",
                        "openalex_id": None,  # Will be filled in enrichment
                        "citation_count": 0,  # Will be filled in enrichment
                        "venue": "arXiv",
                        "concepts": [],
                    })

                    if len(results) >= max_papers:
                        break
            except Exception:
                continue

        return results

    async def _enrich_papers_openalex(self, papers: List[Dict]) -> List[Dict]:
        """Enrich paper list with real citation counts from OpenAlex.

        Look up each paper by DOI (most reliable). For papers without DOI,
        search by title. Falls back to original paper dict if lookup fails.
        """
        enriched = []
        for paper in papers:
            doi = paper.get("doi", "")
            title = paper.get("title", "")

            if doi and doi.startswith("10."):
                # DOI lookup is most reliable
                result = self._openalex.get_paper(doi)
                if result.success and result.output:
                    enriched_paper = {**paper, **result.output}
                    enriched.append(enriched_paper)
                    continue

            if title:
                # Fallback: search by title keywords
                title_keywords = " ".join(title.split()[:8])
                result = self._openalex.search(query=title_keywords, top_k=5)
                if result.success and result.output:
                    candidates = result.output.get("papers", [])
                    from tiny_agents.core.similarity import title_similarity
                    best = None
                    best_score = 0.0
                    for c in candidates:
                        score = title_similarity(title, c.get("title", ""))
                        if score > best_score and score > 0.5:
                            best = c
                            best_score = score
                    if best:
                        enriched_paper = {**paper, **best}
                        enriched.append(enriched_paper)
                        continue

            # Fallback: keep original
            enriched.append(paper)

        return enriched

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
