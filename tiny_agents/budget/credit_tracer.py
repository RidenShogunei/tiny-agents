"""CreditTracer — hierarchical credit signal computation.

Core principle (per user correction):
  - msg_gain = Q_t - Q_{t-1} is the PRIMARY signal
  - agent_credit = sum(msg_gains by agent) is DERIVED from msg_gains
  - step_count_ratio is demoted to agent_activity_ratio (debug only, NOT credit)
"""

from __future__ import annotations

import time
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tiny_agents.budget.types import (
    ActionType,
    CollaborationStep,
    CreditStats,
)


class CreditTracer:
    """Computes hierarchical credit statistics from collaboration trajectory."""

    def __init__(
        self,
        k_marginal: int = 3,
        k_slope: int = 5,
        eps: float = 1e-9,
    ):
        """
        Args:
            k_marginal: window size for moving average of msg_gains
            k_slope: window size for computing gain slope
            eps: small constant to avoid log(0)
        """
        self.k_marginal = k_marginal
        self.k_slope = k_slope
        self.eps = eps

        self.trajectory: List[CollaborationStep] = []
        self._msg_gains: List[float] = []         # Q_t values over time
        self._quality_scores: List[float] = []    # raw Q_t sequence
        self._msg_gains_per_agent: Dict[str, List[float]] = defaultdict(list)

    def reset(self) -> None:
        """Reset state for a new problem."""
        self.trajectory.clear()
        self._msg_gains.clear()
        self._quality_scores.clear()
        self._msg_gains_per_agent.clear()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_step(
        self,
        step: CollaborationStep,
        quality_score: float,
    ) -> None:
        """
        Add a completed collaboration step and its quality score Q_t.

        Args:
            step: the collaboration step that was executed
            quality_score: Q_t, the verifier/proxy quality score after this step
        """
        self.trajectory.append(step)

        # Compute msg_gain = Q_t - Q_{t-1}
        prev_q = self._quality_scores[-1] if self._quality_scores else 0.0
        msg_gain = quality_score - prev_q
        step.msg_gain = msg_gain

        self._quality_scores.append(quality_score)
        self._msg_gains.append(msg_gain)
        self._msg_gains_per_agent[step.agent_name].append(msg_gain)

        # Aggregate agent_credit from msg_gains
        step.agent_credit = self._compute_agent_credit(step.agent_name)

    def compute_stats(self) -> CreditStats:
        """Compute full credit statistics from current trajectory."""
        if not self.trajectory:
            return CreditStats()

        n = len(self.trajectory)

        # Message-level stats
        msg_gain = self._msg_gains[-1] if self._msg_gains else 0.0
        msg_gain_ma = self._moving_average(self._msg_gains, self.k_marginal)
        msg_gain_slope = self._compute_slope(self._msg_gains, self.k_slope)

        # Agent-level credit: aggregated from msg_gains
        agent_credit = self._compute_all_agent_credits()

        # Agent activity ratio (DEBUG ONLY — NOT credit)
        agent_activity = self._compute_activity_ratios()

        # Derived signals
        credit_entropy = self._compute_entropy(agent_credit)
        credit_concentration = self._compute_concentration(agent_credit)

        # Trajectory-level
        disagreement = self._compute_disagreement()
        uncertainty = self._compute_uncertainty()

        return CreditStats(
            msg_gain=msg_gain,
            msg_gain_ma=msg_gain_ma,
            msg_gain_slope=msg_gain_slope,
            agent_credit=agent_credit,
            agent_activity_ratio=agent_activity,
            credit_entropy=credit_entropy,
            credit_concentration=credit_concentration,
            disagreement=disagreement,
            uncertainty=uncertainty,
        )

    def get_agent_credit_dict(self) -> Dict[str, float]:
        """Return normalized {agent_name: credit} dict."""
        return self._compute_all_agent_credits()

    # ── Internal methods ──────────────────────────────────────────────────────

    def _compute_agent_credit(self, agent_name: str) -> float:
        """Sum of all msg_gains produced by this agent."""
        return sum(self._msg_gains_per_agent.get(agent_name, []))

    def _compute_all_agent_credits(self) -> Dict[str, float]:
        """Compute agent_credit for all agents (sum of msg_gains)."""
        result = {}
        for agent_name, gains in self._msg_gains_per_agent.items():
            result[agent_name] = sum(gains)
        return result

    def _compute_activity_ratios(self) -> Dict[str, float]:
        """Compute step count ratios per agent. DEBUG ONLY."""
        if not self.trajectory:
            return {}
        n = len(self.trajectory)
        counts = defaultdict(int)
        for step in self.trajectory:
            counts[step.agent_name] += 1
        return {k: v / n for k, v in counts.items()}

    def _compute_entropy(self, credit_dict: Dict[str, float]) -> float:
        """Compute Shannon entropy H(credit) over agents."""
        total = sum(credit_dict.values())
        if total <= self.eps:
            return 0.0
        probs = [v / total for v in credit_dict.values()]
        h = 0.0
        for p in probs:
            if p > self.eps:
                h -= p * math.log(p)
        return h

    def _compute_concentration(self, credit_dict: Dict[str, float]) -> float:
        """Compute credit concentration: max_i credit_i / total."""
        total = sum(credit_dict.values())
        if total <= self.eps:
            return 0.0
        return max(credit_dict.values()) / total

    def _compute_disagreement(self) -> float:
        """
        Fraction of steps where answer_candidate differs from the best answer.
        We compare each answer_candidate to the most recent one.
        """
        candidates = [
            s.answer_candidate for s in self.trajectory
            if s.answer_candidate is not None
        ]
        if len(candidates) < 2:
            return 0.0
        divergent = sum(
            1 for i in range(1, len(candidates))
            if candidates[i] != candidates[i - 1]
        )
        return divergent / (len(candidates) - 1)

    def _compute_uncertainty(self) -> float:
        """
        Uncertainty derived from msg_gain_ma and trajectory consistency.
        High uncertainty when msg_gain_ma is near zero (diminishing returns).
        """
        if not self._msg_gains:
            return 1.0
        ma = self._moving_average(self._msg_gains, self.k_marginal)
        # Normalize: if ma is close to 0, uncertainty high; if ma is high, low uncertainty
        # We use a sigmoid-like mapping: uncertainty = 1 - sigmoid(ma * scale)
        # Simplified: uncertainty = 1 / (1 + |ma| * 10)
        return 1.0 / (1.0 + abs(ma) * 10.0)

    def _moving_average(self, values: List[float], k: int) -> float:
        """Simple moving average of last k values."""
        if not values:
            return 0.0
        window = values[-k:]
        return sum(window) / len(window)

    def _compute_slope(self, values: List[float], k: int) -> float:
        """
        Compute linear slope of last k values using ordinary least squares.
        Returns rate of change of msg_gain over steps.
        """
        if len(values) < 2:
            return 0.0
        window = values[-k:]
        n = len(window)
        if n < 2:
            return 0.0
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(window) / n
        num = sum((x[i] - x_mean) * (window[i] - y_mean) for i in range(n))
        den = sum((x[i] - x_mean) ** 2 for i in range(n))
        if abs(den) < self.eps:
            return 0.0
        return num / den
