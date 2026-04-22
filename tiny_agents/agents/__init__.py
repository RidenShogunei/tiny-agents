"""Generic agents — not domain-specific.

These agents are task-type agnostic (routing, coding, reviewing, etc.)
and can be reused across different pipelines.
"""

from tiny_agents.agents.router import RouterAgent
from tiny_agents.agents.coder import CoderAgent
from tiny_agents.agents.critic import CriticAgent
from tiny_agents.agents.vl_perception import VLPerceptionAgent
from tiny_agents.agents.tool_reasoner import ToolReasonerAgent

__all__ = [
    "RouterAgent",
    "CoderAgent",
    "CriticAgent",
    "VLPerceptionAgent",
    "ToolReasonerAgent",
]
