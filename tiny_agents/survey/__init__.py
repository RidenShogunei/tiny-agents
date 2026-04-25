"""Survey package — research survey/review paper generation pipeline.

Built on top of the generic multi-agent core (tiny_agents.core).
This package is NOT part of the core framework — it is one possible
application demonstrating how to compose agents and tools into a pipeline.

Architecture:
    tiny_agents.core          — BaseAgent, Orchestrator, SessionContext, tools
    tiny_agents.agents       — Generic agents (Router, Coder, Critic, etc.)
    tiny_agents.survey       — Survey-specific pipeline (this package)
        agents/               — Survey agents (PlanningAgent, SearchAgent, etc.)
        tools/                — Survey tools (ArxivTool, OpenAlexTool, etc.)
        survey_pipeline.py    — Pipeline orchestrator
        paper_kg.py           — Paper knowledge graph
        similarity.py          — Title/text similarity utilities
"""

from tiny_agents.survey.survey_pipeline import SurveyPipeline
from tiny_agents.survey.agents import (
    PlanningAgent,
    SearchAgent,
    ReaderAgent,
    WriterAgent,
    SynthesizerAgent,
    CitationAgent,
)
from tiny_agents.survey.paper_kg import KnowledgeGraph as PaperKnowledgeGraph
from tiny_agents.survey.similarity import title_similarity

__all__ = [
    # Pipeline
    "SurveyPipeline",
    # Agents
    "PlanningAgent",
    "SearchAgent",
    "ReaderAgent",
    "WriterAgent",
    "SynthesizerAgent",
    "CitationAgent",
    # Knowledge graph
    "PaperKnowledgeGraph",
    # Utilities
    "title_similarity",
]
