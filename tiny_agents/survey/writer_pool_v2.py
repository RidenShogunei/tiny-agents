"""MultiGPUWriterPoolV2 — Writer pool with Critic + Revision Loop.

Direction A: After each section is drafted, a CriticAgent scores it.
If score < threshold, the section is rewritten (up to max_revision_attempts).
Direction B: PaperKnowledgeGraph tracks which papers are covered in which
sections, and injects cross-section context to avoid duplication.

Pipeline per section:
    Draft → Critic(score) → [score < threshold?] → Rewrite → Critic → ...

Usage:
    pool = MultiGPUWriterPoolV2(
        num_instances=5,
        gpu_assignment=[0, 1, 2, 0, 1],
        critic_model_key="gpu0",   # model key for critic
        revision_threshold=6.0,    # minimum score to accept (out of 10)
        max_revision_attempts=2,   # rewrite up to 2 times
    )
    pool.set_backend(backend)
    pool.set_knowledge_graph(graph)

    results = pool.write_all_v2(sections, summaries, topic, context)
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext
from tiny_agents.survey.paper_kg import KnowledgeGraph, PaperNode

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
    """Result of writing one section (with critique metadata)."""
    section_id: str
    title: str
    content: str = ""
    success: bool = False
    error: str = ""
    papers_used: List[str] = field(default_factory=list)
    duration: float = 0.0
    revision_attempts: int = 0
    critique_score: float = 0.0
    critique_feedback: str = ""


# ── Critique prompt ──────────────────────────────────────────────────────────

CRITIC_REVISION_PROMPT = """You are a strict academic paper reviewer. Evaluate the following section of a survey paper.

SECTION TITLE: {title}
SECTION CONTENT:
---
{content}
---

EVALUATION CRITERIA (score each 1-10, then compute weighted total /10):
1. Relevance (3pts): Does it discuss papers relevant to the section keywords?
2. Structure (2pts): Is it organized by theme, with clear paragraphs?
3. Citation quality (2pts): Are citations [N] used correctly and consistently?
4. Depth (2pts): Does it go beyond surface-level description?
5. Avoiding redundancy (1pt): Does it NOT repeat content covered in other sections?

SCORING FORMAT — respond with ONLY this exact format (no explanation before/after):
SCORE: X/Y
FEEDBACK: one sentence explaining the main issue (or "None" if score ≥ 7)
"""

CRITIC_REVISION_PROMPT_NO_GRAPH = """You are a strict academic paper reviewer. Evaluate the following section of a survey paper.

SECTION TITLE: {title}
SECTION CONTENT:
---
{content}
---

EVALUATION CRITERIA (score each 1-10, then compute weighted total /10):
1. Relevance (3pts): Does it discuss papers relevant to the section keywords?
2. Structure (2pts): Is it organized by theme, with clear paragraphs?
3. Citation quality (2pts): Are citations [N] used correctly and consistently?
4. Depth (2pts): Does it go beyond surface-level description?
5. Avoiding redundancy (1pt): Does NOT repeat content covered in other sections?

SCORING FORMAT — respond with ONLY this exact format:
SCORE: X/Y
FEEDBACK: one sentence explaining the main issue (or "None" if score ≥ 7)
"""


def _parse_critique(response: str) -> tuple[float, str]:
    """Parse score and feedback from critic output."""
    score = 0.0
    feedback = ""
    lines = response.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                parts = line.split("SCORE:")[1].strip()
                if "/" in parts:
                    num = float(parts.split("/")[0].strip())
                else:
                    num = float(parts.strip().split()[0])
                score = num
            except (ValueError, IndexError):
                score = 0.0
        elif line.startswith("FEEDBACK:"):
            feedback = line.split("FEEDBACK:")[1].strip()
    return score, feedback


# ── Writer agent with revision support ──────────────────────────────────────


WRITER_REVISION_PROMPT = """You are an academic paper writer. Your task is to write one section of a survey/review paper.

STYLE RULES (strictly follow these):
1. Write in academic prose — formal, impersonal, precise
2. Use citations like [1], [2], [3] inline — NOT parenthetical author-year style
3. Group papers by THEME, not by individual paper
4. For each theme, discuss 2-4 papers together, explaining WHY they are grouped
5. Do NOT list papers one-by-one (no "The paper [1] did X, then [2] did Y...")
6. Write 3-6 paragraphs per section, each 3-5 sentences
7. Acknowledge limitations and contradictions honestly
8. Do NOT overstate contributions — be objective

OUTPUT FORMAT:
## Section Title

Opening paragraph: motivate why this topic matters, give an overview...

### Theme A (2-4 sentences)
Paragraph discussing 2-4 related papers...

### Theme B
...

### References
[1] Title. Authors. Year. Venue.
"""


class RevisionWriterAgent(BaseAgent):
    """Writer agent that accepts revision feedback and rewrites accordingly (stateless)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        model_key: str = None,
        **kwargs,
    ):
        super().__init__(
            name="revision_writer",
            model_name=model_name,
            role_prompt=WRITER_REVISION_PROMPT,
            **kwargs,
        )
        self._model_key = model_key or model_name

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Draft or rewrite a section, optionally with critique feedback."""
        section_id = input_data.get("section_id", "?")
        section_title = input_data.get("section_title", "Unknown Section")
        keywords = input_data.get("keywords", [])
        relevant_papers = input_data.get("relevant_papers", [])
        topic = input_data.get("topic", "the research topic")
        revision_feedback = input_data.get("revision_feedback", "")
        max_length = input_data.get("max_length", 600)

        prompt = self._build_prompt(
            section_title, keywords, relevant_papers, topic, revision_feedback, max_length
        )
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": prompt})

        if self.backend is not None:
            temp = context.config.get("temperature", 0.3)
            max_tok = context.config.get("max_tokens", 1536)
            content = self.backend.generate(
                model_key=self._model_key,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        else:
            content = f"## {section_title}\n\n[Content generation failed — no backend]"

        # Extract cited paper IDs
        import re
        cited_titles = []
        refs = re.findall(r'\[\d+\]\s+(.+?)(?:\n|$)', content)
        for ref in refs:
            ref = re.sub(r'\s+\d+\s*$', '', ref.strip())
            if ref and len(ref) > 5:
                cited_titles.append(ref)

        cited_papers = [p for p in relevant_papers if p.get("title", "") in cited_titles]

        context.add_message(self.name, "user", prompt[:200] + "...")
        context.add_message(self.name, "assistant", content[:200] + "...")

        return AgentOutput(
            thought=f"Section {section_id} ({section_title}): {len(relevant_papers)} papers, feedback={bool(revision_feedback)}",
            action="respond",
            payload={
                "section_id": section_id,
                "section_title": section_title,
                "content": content,
                "papers_cited": cited_papers,
            },
            finished=True,
        )

    def _build_prompt(
        self,
        section_title: str,
        keywords: List[str],
        papers: List[Dict],
        topic: str,
        revision_feedback: str,
        max_length: int,
    ) -> str:
        lines = []
        lines.append(f"Survey topic: {topic}")
        lines.append(f"Section: {section_title}")
        lines.append(f"Keywords: {', '.join(keywords)}")
        lines.append("")

        if revision_feedback:
            lines.append("=" * 60)
            lines.append("REVISION FEEDBACK (address these issues):")
            lines.append(revision_feedback)
            lines.append("=" * 60)
            lines.append("")

        lines.append("RELEVANT PAPERS:")
        for idx, p in enumerate(papers):
            lines.append(f"\n[Paper {idx + 1}]")
            lines.append(f"Title: {p.get('title', 'Unknown')}")
            authors = p.get("authors", [])
            if authors:
                lines.append(f"Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}")
            lines.append(f"Method: {p.get('method', 'Unknown')}")
            lines.append(f"Contribution: {p.get('contribution', 'Unknown')}")
            lines.append(f"Limitation: {p.get('limitation', 'Not clearly stated')}")
            lines.append(f"Category: {p.get('category', 'general')}")
            if p.get("year"):
                lines.append(f"Year: {p.get('year')}")

        lines.append("")
        lines.append(f"Write section '{section_title}' following the style rules.")
        lines.append(f"Target length: ~{max_length} words.")
        lines.append("Cite papers as [1], [2], [3] inline. Group by theme.")
        if revision_feedback:
            lines.append("IMPORTANT: Address ALL points in the revision feedback above.")

        return "\n".join(lines)


# ── Main writer pool ─────────────────────────────────────────────────────────


class MultiGPUWriterPoolV2:
    """Writer pool with critic + revision loop and knowledge graph cross-ref.

    Direction A: Each section is drafted, critiqued, and potentially rewritten.
    Direction B: PaperKnowledgeGraph tracks coverage to avoid duplication.
    """

    def __init__(
        self,
        num_instances: int = 5,
        gpu_assignment: Optional[List[int]] = None,
        available_gpus: Optional[List[int]] = None,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        critic_model_key: str = "gpu0",
        revision_threshold: float = 6.0,
        max_revision_attempts: int = 2,
        timeout_per_section: int = 120,
    ):
        self.num_instances = num_instances
        self.model_name = model_name
        self.critic_model_key = critic_model_key
        self.revision_threshold = revision_threshold
        self.max_revision_attempts = max_revision_attempts
        self.timeout_per_section = timeout_per_section

        if gpu_assignment is None:
            gpus = available_gpus or [0, 1, 2]
            gpu_assignment = [gpus[i % len(gpus)] for i in range(num_instances)]
        self.gpu_assignment = gpu_assignment

        self.backend = None
        self._writers: List[RevisionWriterAgent] = []
        self._loaded_keys: Dict[int, str] = {}
        self._critic: Optional[BaseAgent] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._written_sections: Dict[str, str] = {}  # section_id -> content (for overlap detection)

    def set_backend(self, backend) -> None:
        self.backend = backend
        self._writers = []
        self._loaded_keys = {}

        # Load writer models per GPU
        gpu_to_instances: Dict[int, List[int]] = {}
        for i, gpu in enumerate(self.gpu_assignment):
            gpu_to_instances.setdefault(gpu, []).append(i)

        for gpu, instance_indices in gpu_to_instances.items():
            key = f"writer_gpu{gpu}"
            if key not in self.backend.instances:
                logger.info(f"[WriterPoolV2] Loading {self.model_name} on GPU {gpu} as '{key}'")
                self.backend.load_model(key, self.model_name, gpu=gpu)
            else:
                logger.info(f"[WriterPoolV2] Reusing existing '{key}' on GPU {gpu}")

            self._loaded_keys[gpu] = key

            for idx in instance_indices:
                agent = RevisionWriterAgent(model_name=self.model_name, model_key=key)
                agent.backend = backend
                self._writers.append(agent)

        # Load critic on its designated GPU
        if self.critic_model_key not in self.backend.instances:
            # Extract gpu from critic_model_key if it contains gpuN pattern
            import re
            m = re.search(r'gpu(\d+)', self.critic_model_key)
            critic_gpu = int(m.group(1)) if m else 0
            logger.info(f"[WriterPoolV2] Loading critic on GPU {critic_gpu}")
            self.backend.load_model(self.critic_model_key, self.model_name, gpu=critic_gpu)

        logger.info(f"[WriterPoolV2] {len(self._writers)} writers + 1 critic ready")

    def set_knowledge_graph(self, graph: KnowledgeGraph) -> None:
        self._knowledge_graph = graph

    def set_written_sections(self, written: Dict[str, str]) -> None:
        """Inject already-written section contents for overlap detection."""
        self._written_sections = written

    async def write_all_v2(
        self,
        sections: List[Dict],
        summaries: List[Dict],
        topic: str,
        context: SessionContext,
    ) -> Dict[str, Dict[str, Any]]:
        """Write all sections with critique + revision + cross-ref.

        Direction A: Draft → Critique → [score < threshold?] → Rewrite → ...
        Direction B: Check knowledge graph for cross-section duplication risk.
        """
        if not sections:
            return {}

        # Match papers to sections (keyword-based)
        specs = []
        for sec in sections:
            section_id = sec.get("id", f"section_{len(specs)}")
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

        # Run all sections through draft → critique → rewrite loop
        results = {}
        for spec in specs:
            result = await self._write_one_with_critique(spec, topic, context)
            results[spec.section_id] = {
                "content": result.content,
                "papers_used": result.papers_used,
                "success": result.success,
                "error": result.error,
                "revision_attempts": result.revision_attempts,
                "critique_score": result.critique_score,
                "critique_feedback": result.critique_feedback,
            }
            # Track this section's content for overlap detection in subsequent sections
            if result.success and result.content:
                self._written_sections[spec.section_id] = result.content

        return results

    async def _write_one_with_critique(
        self,
        spec: SectionSpec,
        topic: str,
        context: SessionContext,
    ) -> SectionResult:
        """Draft → critique → rewrite loop for one section."""
        start = time.time()
        revision_attempts = 0
        current_content = ""
        current_feedback = ""
        papers_used = []

        # Direction B: check overlap with already-written sections
        overlap_warning = ""
        if self._knowledge_graph or self._written_sections:
            overlaps = self._detect_overlap(spec)
            if overlaps:
                overlap_warning = (
                    f" OVERLAP WARNING: These already-written sections share keywords with this one: "
                    + "; ".join(f"{o['section_id']} (overlap={o['overlap_score']:.1f})" for o in overlaps[:2])
                    + ". Ensure your content does NOT repeat what's already covered."
                )

        for attempt in range(self.max_revision_attempts + 1):
            revision_attempts = attempt
            revision_feedback = current_feedback + overlap_warning if attempt > 0 else overlap_warning

            # Draft or rewrite
            draft_content, draft_papers = await self._draft(
                spec, topic, revision_feedback, context, attempt
            )

            if not draft_content:
                return SectionResult(
                    section_id=spec.section_id,
                    title=spec.title,
                    success=False,
                    error="Draft generation failed",
                    revision_attempts=revision_attempts,
                )

            # Critique
            score, feedback = await self._critique(spec.title, draft_content, context)

            if score >= self.revision_threshold or attempt == self.max_revision_attempts:
                # Accept this version
                current_content = draft_content
                papers_used = draft_papers
                break
            else:
                # Not good enough — rewrite
                current_feedback = f"[Attempt {attempt+1}/{self.max_revision_attempts}] Score={score}/10. {feedback}"
                logger.info(f"[Writer-{spec.section_id}] Score={score}/10 — rewriting (attempt {attempt+1})")
                continue

        return SectionResult(
            section_id=spec.section_id,
            title=spec.title,
            content=current_content,
            success=bool(current_content),
            papers_used=papers_used,
            duration=time.time() - start,
            revision_attempts=revision_attempts,
            critique_score=score,
            critique_feedback=feedback,
        )

    async def _draft(
        self,
        spec: SectionSpec,
        topic: str,
        revision_feedback: str,
        context: SessionContext,
        attempt: int = 0,
    ) -> tuple[str, List[str]]:
        """Generate or regenerate a section draft."""
        writer = self._writers[len(spec.section_id) % len(self._writers)]

        session_id = f"write_{spec.section_id}_{attempt:02d}"
        draft_context = SessionContext(
            session_id=session_id,
            config={"temperature": 0.3, "max_tokens": 1536},
        )

        input_data = {
            "section_id": spec.section_id,
            "section_title": spec.title,
            "keywords": spec.keywords,
            "relevant_papers": spec.paper_summaries,
            "topic": topic,
            "revision_feedback": revision_feedback,
            "max_length": spec.max_length,
        }

        output: AgentOutput = await writer.run(input_data, draft_context)

        payload = output.payload
        content = payload.get("content", "")
        cited_papers = payload.get("papers_cited", [])
        paper_ids = [p.get("title", "") for p in cited_papers]

        return content, paper_ids

    async def _critique(
        self,
        section_title: str,
        content: str,
        context: SessionContext,
    ) -> tuple[float, str]:
        """Score a draft and return (score, feedback)."""
        if not content or len(content.strip()) < 50:
            return 0.0, "Content too short to evaluate."

        # Truncate content for the prompt (avoid exceeding context)
        content_snippet = content[:3000]

        prompt = CRITIC_REVISION_PROMPT.format(
            title=section_title,
            content=content_snippet,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.backend.generate(
                model_key=self.critic_model_key,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as e:
            logger.warning(f"[Critic] Failed: {e}")
            return 5.0, f"Critic error: {e}"  # accept on error

        score, feedback = _parse_critique(response)
        return score, feedback

    def _detect_overlap(self, spec: SectionSpec) -> List[Dict[str, Any]]:
        """Check if section keywords overlap with already-written sections."""
        if not self._written_sections:
            return []
        new_kw = set(k.lower() for k in spec.keywords)
        overlaps = []
        for sec_id, sec_content in self._written_sections.items():
            if not sec_content:
                continue
            shared = [kw for kw in spec.keywords if kw.lower() in sec_content.lower()]
            if shared:
                score = len(shared) / max(len(new_kw), 1)
                overlaps.append({
                    "section_id": sec_id,
                    "overlap_score": score,
                    "shared_keywords": shared,
                })
        return sorted(overlaps, key=lambda x: x["overlap_score"], reverse=True)

    def _match_papers(self, summaries: List[Dict], keywords: List[str]) -> List[Dict]:
        """Select top papers for a section based on keyword relevance."""
        if not keywords or not summaries:
            return summaries[:5] if summaries else []
        kw_lower = [k.lower() for k in keywords]
        scored = []
        for p in summaries:
            text = (p.get("title", "") + " " + p.get("method", "") + " " + p.get("category", "")).lower()
            score = sum(1 for kw in kw_lower if kw in text)
            score += p.get("relevance_score", 0)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:8]]
