"""CandidateManager — tracks current best answer, scores, and candidate history.

This avoids field drift and circular dependencies when multiple modules
(reasoner, verifier, controller) need to query answer state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CandidateAnswer:
    """A candidate answer with its quality score."""
    text: str
    score: float
    step_id: int
    agent_name: str


class CandidateManager:
    """Manages current best answer and candidate history for the budget loop."""

    def __init__(self):
        self.candidates: List[CandidateAnswer] = []
        self.best: Optional[CandidateAnswer] = None
        self._scores_by_agent: Dict[str, List[float]] = {}

    def reset(self) -> None:
        """Reset for a new problem."""
        self.candidates.clear()
        self.best = None
        self._scores_by_agent.clear()

    def add_candidate(
        self,
        text: str,
        score: float,
        step_id: int,
        agent_name: str,
    ) -> Tuple[CandidateAnswer, bool]:
        """
        Add a new candidate answer.

        Returns:
            (candidate, is_new_best)
        """
        candidate = CandidateAnswer(
            text=text,
            score=score,
            step_id=step_id,
            agent_name=agent_name,
        )
        self.candidates.append(candidate)

        if agent_name not in self._scores_by_agent:
            self._scores_by_agent[agent_name] = []
        self._scores_by_agent[agent_name].append(score)

        is_new_best = False
        if self.best is None or score > self.best.score:
            self.best = candidate
            is_new_best = True

        return candidate, is_new_best

    def get_best_text(self) -> Optional[str]:
        return self.best.text if self.best else None

    def get_best_score(self) -> float:
        return self.best.score if self.best else 0.0

    def get_best_step_id(self) -> int:
        return self.best.step_id if self.best else -1

    def get_score_by_agent(self, agent_name: str) -> List[float]:
        """Get all scores produced by a specific agent."""
        return self._scores_by_agent.get(agent_name, [])

    def get_latest_candidate(self) -> Optional[CandidateAnswer]:
        """Get the most recently added candidate."""
        return self.candidates[-1] if self.candidates else None

    def get_disagreement_score(self) -> float:
        """
        Fraction of candidates that differ from current best.
        Higher = more disagreement among agents.
        """
        if not self.candidates or self.best is None:
            return 0.0
        if len(self.candidates) < 2:
            return 0.0

        divergent = sum(
            1 for c in self.candidates
            if c.text.strip() != self.best.text.strip()
        )
        return divergent / len(self.candidates)

    def get_all_texts(self) -> List[str]:
        """Return all candidate answer texts."""
        return [c.text for c in self.candidates]

    def get_answer_history_summary(self) -> Dict:
        """Summary of answer evolution for logging."""
        return {
            "num_candidates": len(self.candidates),
            "best_score": self.get_best_score(),
            "best_step_id": self.get_best_step_id(),
            "best_agent": self.best.agent_name if self.best else None,
            "disagreement": self.get_disagreement_score(),
            "agents_answered": list(self._scores_by_agent.keys()),
        }
