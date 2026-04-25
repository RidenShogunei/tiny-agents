"""SurveyPipeline — full pipeline for survey/review paper generation.

This is a domain-specific pipeline built on top of the generic core.
It is NOT part of the core framework — it is one example application.

Pipeline flow:
  1. PlanningAgent   → generate section outline
  2. SearchAgent      → discover papers from arXiv
  3. ReaderAgent     → extract structured summaries
  4. WriterPool       → write all sections in parallel
  5. SynthesizerAgent → merge + write intro/conclusion/abstract
  6. CitationAgent    → format references + BibTeX
  7. MarkdownWriterTool → output final .md file

Usage:
    from tiny_agents.survey import SurveyPipeline
    from tiny_agents.models import VLLMBackend

    backend = VLLMBackend()
    backend.load_model("gpu0", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0)
    backend.load_model("gpu1", "Qwen/Qwen2.5-1.5B-Instruct", gpu=1)

    pipeline = SurveyPipeline(
        num_writers=4,
        planner_model="gpu0",
        writer_model="gpu1",
        output_dir="./output",
    )
    pipeline.set_backend(backend)

    result = await pipeline.run(
        topic="LoRA in Vision Models",
        num_sections=8,
        max_papers=30,
    )
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.survey.writer_pool_v2 import MultiGPUWriterPoolV2
from tiny_agents.survey.paper_kg import KnowledgeGraph as PaperKnowledgeGraph
from tiny_agents.survey.tools.arxiv_tool import ArxivTool
from tiny_agents.survey.tools.markdown_writer_tool import MarkdownWriterTool
from tiny_agents.survey.tools.openalex_tool import OpenAlexTool
from tiny_agents.survey.agents import (
    PlanningAgent,
    SearchAgent,
    ReaderAgent,
    WriterAgent,
    SynthesizerAgent,
    CitationAgent,
)

logger = __import__("logging").getLogger(__name__)


class SurveyPipeline:
    """Survey paper generation pipeline — a concrete application of the core framework.

    This class is intentionally NOT in the core — it is a domain-specific pipeline
    that demonstrates how to compose agents and tools into a multi-step workflow.
    """

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

        self.backend = None
        self._planner: Optional[PlanningAgent] = None
        self._searcher: Optional[SearchAgent] = None
        self._reader: Optional[ReaderAgent] = None
        self._synthesizer: Optional[SynthesizerAgent] = None
        self._citation: Optional[CitationAgent] = None
        self._writer_pool: Optional[MultiGPUWriterPoolV2] = None
        self._arxiv = ArxivTool()
        self._md_writer = MarkdownWriterTool(default_output_dir=output_dir)
        self._openalex = OpenAlexTool(email="tiny-agents@research.ai")
        self._knowledge_graph: Optional[PaperKnowledgeGraph] = None

    def set_backend(self, backend) -> None:
        """Inject VLLMBackend and wire up all agents."""
        self.backend = backend

        self._planner = PlanningAgent(model_name=self.planner_model)
        self._planner.backend = backend

        self._searcher = SearchAgent(model_name=self.planner_model)
        self._searcher.backend = backend

        self._reader = ReaderAgent(model_name=self.reader_model)
        self._reader.backend = backend

        self._synthesizer = SynthesizerAgent(model_name=self.synthesizer_model)
        self._synthesizer.backend = backend

        self._citation = CitationAgent(model_name=self.citation_model)
        self._citation.backend = backend

        # Writers spread across available GPUs (GPU IDs determined by model key)
        gpu_assignment = [i % 4 for i in range(self.num_writers)]
        self._writer_pool = MultiGPUWriterPoolV2(
            num_instances=self.num_writers,
            gpu_assignment=gpu_assignment,
            available_gpus=[0, 1, 2, 3],
            model_name=self.writer_model,
            critic_model_key=None,  # Set to a 9B key to enable critique
            revision_threshold=6.0,
            max_revision_attempts=2,
        )
        self._writer_pool.set_backend(backend)
        self._knowledge_graph = PaperKnowledgeGraph()

    async def run(
        self,
        topic: str,
        num_sections: int = 8,
        max_papers: Optional[int] = None,
        context: Optional[SessionContext] = None,
    ) -> Dict[str, Any]:
        """Execute the full survey generation pipeline.

        Returns:
            {success, filepath, title, sections_written, papers_found,
             papers_read, execution_time, ...}
        """
        start_time = time.time()
        max_papers = max_papers or self.max_papers
        session_id = f"survey_{int(start_time)}"

        if context is None:
            context = SessionContext(session_id=session_id)
        else:
            context.session_id = session_id

        steps = []
        errors: List[str] = []

        # ── Step 1: Planning ───────────────────────────────────────────
        print(f"[{session_id}] Step 1/7: Planning — topic: {topic}")
        step_start = time.time()
        try:
            planner_out: AgentOutput = await self._planner.run(
                {"topic": topic, "num_sections": num_sections}, context
            )
            outline = planner_out.payload.get("outline", {})
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

        # ── Step 2: Search via arXiv ─────────────────────────────────
        print(f"[{session_id}] Step 2/7: Searching papers via arXiv...")
        step_start = time.time()
        try:
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

        # ── Step 2.5: Enrich with OpenAlex citation counts ────────────
        print(f"[{session_id}] Step 2.5/7: Enriching papers with real citation counts...")
        step_enrich = time.time()
        try:
            papers = await self._enrich_papers_openalex(papers)
        except Exception as e:
            errors.append(f"Paper enrichment failed: {e}")
        steps.append({"step": "enrich", "time": time.time() - step_enrich, "papers": len(papers)})
        print(f"[{session_id}]   → {len(papers)} papers enriched")

        # ── Step 3: Read ───────────────────────────────────────────────
        print(f"[{session_id}] Step 3/7: Reading and extracting summaries...")
        step_start = time.time()
        try:
            reader_out: AgentOutput = await self._reader.run(
                {"papers": papers, "topic": topic}, context
            )
            summaries = reader_out.payload.get("summaries", [])
        except Exception as e:
            errors.append(f"Reading failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        steps.append({"step": "reading", "time": time.time() - step_start, "summaries": len(summaries)})
        print(f"[{session_id}]   → {len(summaries)} paper summaries extracted")

        if not summaries:
            errors.append("No summaries generated")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 3.5: Build knowledge graph ───────────────────────────
        print(f"[{session_id}] Step 3.5/7: Building paper knowledge graph...")
        step_kg = time.time()
        for paper in summaries:
            self._knowledge_graph.add_paper(paper)
        kg_stats = self._knowledge_graph.get_paper_stats()
        print(f"[{session_id}]   → {kg_stats['total_papers']} papers in graph "
              f"({len(kg_stats.get('categories', {}))} categories, "
              f"{len(kg_stats.get('methods', {}))} methods)")
        steps.append({"step": "knowledge_graph", "time": time.time() - step_kg, **kg_stats})

        # ── Step 4: Write sections ─────────────────────────────────────
        print(f"[{session_id}] Step 4/7: Writing {len(sections)} sections...")
        step_start = time.time()
        try:
            self._writer_pool.set_knowledge_graph(self._knowledge_graph)
            section_results = await self._writer_pool.write_all_v2(
                sections=sections,
                summaries=summaries,
                topic=topic,
                context=context,
            )
        except Exception as e:
            errors.append(f"Section writing failed: {e}")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        drafted_sections = []
        for sec in sections:
            sid = sec.get("id", "?")
            if sid in section_results:
                result = section_results[sid]
                drafted_sections.append({
                    "id": sid,
                    "title": sec.get("title", ""),
                    "content": result.get("content", ""),
                    "papers_cited": result.get("papers_used", []),
                })

        steps.append({"step": "writing", "time": time.time() - step_start, "sections_written": len(drafted_sections)})
        print(f"[{session_id}]   → {len(drafted_sections)} sections drafted")

        if not drafted_sections:
            errors.append("No sections were drafted")
            return self._fail(session_id, errors, steps, time.time() - start_time)

        # ── Step 5: Synthesize ────────────────────────────────────────
        print(f"[{session_id}] Step 5/7: Synthesizing full paper...")
        step_start = time.time()
        try:
            synth_out: AgentOutput = await self._synthesizer.run(
                {"sections": drafted_sections, "topic": topic, "title": title},
                context,
            )
            full_paper = synth_out.payload.get("full_paper", "")
        except Exception as e:
            errors.append(f"Synthesis failed: {e}")
            full_paper = ""

        steps.append({"step": "synthesis", "time": time.time() - step_start})
        print(f"[{session_id}]   → Synthesis complete ({len(full_paper)} chars)")

        # ── Step 6: Citation ──────────────────────────────────────────
        print(f"[{session_id}] Step 6/7: Formatting references...")
        step_start = time.time()
        try:
            cit_out: AgentOutput = await self._citation.run(
                {"papers": papers, "full_paper": full_paper}, context
            )
            references = cit_out.payload.get("references", "")
            bibtex = cit_out.payload.get("bibtex", "")
        except Exception as e:
            errors.append(f"Citation formatting failed: {e}")
            references = ""
            bibtex = ""

        steps.append({"step": "citation", "time": time.time() - step_start})
        print(f"[{session_id}]   → {cit_out.payload.get('count', 0)} references formatted")

        # ── Step 7: Write output file ─────────────────────────────────
        print(f"[{session_id}] Step 7/7: Writing output file...")
        step_start = time.time()

        date_str = datetime.now().strftime("%Y-%m-%d")
        final_content = self._assemble_paper(title, topic, sections, drafted_sections,
                                              full_paper, references, bibtex, papers, summaries)

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
            "papers_cited": len(set(
                p.get("title", "") for sec in drafted_sections for p in sec.get("papers_cited", [])
            )),
            "execution_time": elapsed,
            "steps": steps,
            "errors": errors,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _search_papers_arxiv(self, keywords: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search arXiv for papers matching topic keywords."""
        import arxiv as _arxiv

        seen_ids: Dict[str, bool] = {}
        results: List[Dict[str, Any]] = []
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

                    doi = ""
                    if result.doi:
                        doi = result.doi
                    else:
                        m = re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+)', result.entry_id)
                        if m:
                            doi = f"10.48550/arXiv.{m.group(1)}"

                    results.append({
                        "title": result.title or "Unknown",
                        "authors": [a.name for a in (result.authors or [])],
                        "year": result.published.year if result.published else None,
                        "abstract": result.summary or "",
                        "doi": doi,
                        "paper_url": result.entry_id or "",
                        "openalex_id": None,
                        "citation_count": 0,
                        "venue": "arXiv",
                        "concepts": [],
                    })

                    if len(results) >= max_papers:
                        break
            except Exception:
                continue

        return results

    async def _enrich_papers_openalex(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich paper list with real citation counts from OpenAlex."""
        from tiny_agents.survey.similarity import title_similarity

        enriched = []
        for paper in papers:
            doi = paper.get("doi", "")
            title = paper.get("title", "")

            if doi and doi.startswith("10."):
                result = self._openalex.get_paper(doi)
                if result.success and result.output:
                    enriched.append({**paper, **result.output})
                    continue

            if title:
                title_keywords = " ".join(title.split()[:8])
                result = self._openalex.search(query=title_keywords, top_k=5)
                if result.success and result.output:
                    candidates = result.output.get("papers", [])
                    best = None
                    best_score = 0.0
                    for c in candidates:
                        score = title_similarity(title, c.get("title", ""))
                        if score > best_score and score > 0.5:
                            best = c
                            best_score = score
                    if best:
                        enriched.append({**paper, **best})
                        continue

            enriched.append(paper)

        return enriched

    def _assemble_paper(
        self,
        title: str,
        topic: str,
        sections: List[Dict],
        drafted_sections: List[Dict],
        full_paper: str,
        references: str,
        bibtex: str,
        papers: List[Dict],
        summaries: List[Dict],
    ) -> str:
        """Assemble the final markdown paper from components."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        content = f"# {title}\n\n*Generated: {date_str}*\n\n"
        content += f"*Topic: {topic}*\n\n"
        content += f"*Statistics: {len(sections)} sections planned, {len(papers)} papers found, "
        content += f"{len(summaries)} papers read, {len(drafted_sections)} sections drafted*\n\n---\n\n"

        if full_paper:
            first_section = re.search(r'^(.*?)^##', full_paper, re.DOTALL | re.MULTILINE)
            if first_section:
                abstract_block = first_section.group(1).strip()
                abstract_words = abstract_block.split()[:300]
                content += "### Abstract\n\n" + " ".join(abstract_words) + "\n\n---\n\n"

        for sec in sorted(drafted_sections, key=lambda s: s.get("id", "")):
            sec_content = sec.get("content", "")
            if sec_content.strip():
                content += sec_content.strip() + "\n\n---\n\n"

        if full_paper:
            conclusion_match = re.search(
                r'(?i)conclusion[:\s]*\n(.*?)$', full_paper, re.DOTALL | re.MULTILINE
            )
            if conclusion_match:
                conclusion_text = " ".join(conclusion_match.group(1).strip().split()[:300])
                content += "### Conclusion\n\n" + conclusion_text + "\n\n"

        if references:
            content += "\n---\n\n" + references + "\n"
        if bibtex:
            content += "\n---\n\n## BibTeX\n\n" + bibtex + "\n"

        return content

    def _sanitize_filename(self, topic: str) -> str:
        name = re.sub(r"[^\w\s\-]", "", topic)
        name = re.sub(r"\s+", "_", name)
        return name[:50] or "survey"

    def _fail(
        self,
        session_id: str,
        errors: List[str],
        steps: List[Dict],
        elapsed: float,
    ) -> Dict[str, Any]:
        print(f"[{session_id}] ✗ Failed: {errors}")
        return {
            "success": False,
            "session_id": session_id,
            "errors": errors,
            "steps": steps,
            "execution_time": elapsed,
        }
