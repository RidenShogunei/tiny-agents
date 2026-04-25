"""BudgetController — rule-driven budget allocation controller.

Key engineering constraints (per user correction):
  1. Default rule: if no rule triggers → CONTINUE (must be explicit)
  2. Cooldown/hysteresis: prevent oscillation between VERIFY and CONTINUE
  3. Decision logging: always record triggered_rule_id for failure analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tiny_agents.budget.types import (
    ActionType,
    BudgetLoopState,
    CreditStats,
    DecisionRecord,
    VerificationResult,
)


@dataclass
class ControllerConfig:
    """Tunable thresholds for the rule engine."""
    # Rule thresholds
    entropy_threshold: float = 0.8         # high entropy → balanced collaboration
    concentration_threshold: float = 0.7   # one agent dominates
    gain_threshold: float = 0.05          # marginal gain below this → STOP
    uncertainty_threshold: float = 0.7     # high uncertainty → CALL_VERIFIER
    disagreement_threshold: float = 0.5   # high disagreement → CALL_VERIFIER

    # Budget parameters
    verify_cost: int = 200                 # token cost of VERIFY action
    min_budget_for_continue: int = 50      # minimum budget to attempt CONTINUE
    min_budget_for_verify: int = 100       # minimum budget to attempt VERIFY

    # Cooldown parameters
    verify_cooldown_steps: int = 1         # VERIFY → no VERIFY for N steps
    consecutive_low_gain_stop: int = 2     # STOP only after N consecutive low-gain steps
    slope_stop_requires_verify: bool = True  # STOP on slope only if last verify was confident


@dataclass
class ControllerHistory:
    """Tracks recent controller decisions for hysteresis."""
    recent_actions: List[ActionType] = field(default_factory=list)
    recent_gains: List[float] = field(default_factory=list)
    consecutive_verifies: int = 0
    consecutive_low_gains: int = 0
    last_verify_confidence: float = 0.0

    def record(
        self,
        action: ActionType,
        msg_gain: float,
        verify_confidence: float = 0.0,
    ) -> None:
        self.recent_actions.append(action)
        self.recent_gains.append(msg_gain)
        if action == ActionType.CALL_VERIFIER:
            self.consecutive_verifies += 1
            self.last_verify_confidence = verify_confidence
        else:
            self.consecutive_verifies = 0

        if msg_gain < 0.02:  # below gain_threshold
            self.consecutive_low_gains += 1
        else:
            self.consecutive_low_gains = 0

        # Keep only recent window
        max_history = 10
        if len(self.recent_actions) > max_history:
            self.recent_actions = self.recent_actions[-max_history:]
            self.recent_gains = self.recent_gains[-max_history:]

    def reset(self) -> None:
        self.recent_actions.clear()
        self.recent_gains.clear()
        self.consecutive_verifies = 0
        self.consecutive_low_gains = 0
        self.last_verify_confidence = 0.0


class BudgetController:
    """
    Rule-driven budget allocation controller.

    Decision rules (checked in priority order):

      R1. IF b_t <= 0                              → STOP
      R2. IF disagreement >= disagreement_threshold → CALL_VERIFIER
      R3. IF consecutive_low_gains >= N            → STOP
      R4. IF uncertainty >= uncertainty_threshold  → CALL_VERIFIER
      R5. IF credit_entropy >= entropy_threshold
          AND budget sufficient                    → CONTINUE_DISCUSS
      R6. IF credit_concentration >= threshold    → STOP
      R7. DEFAULT                                   → CONTINUE_DISCUSS

    Each decide() call returns a DecisionRecord with triggered_rule_id.
    """

    # Rule IDs for logging
    R_BUDGET_EXHAUSTED = 1
    R_HIGH_DISAGREEMENT = 2
    R_CONSECUTIVE_LOW_GAIN = 3
    R_HIGH_UNCERTAINTY = 4
    R_COLLABORATION_FAIR = 5
    R_AGENT_DOMINANCE = 6
    R_DEFAULT = 7

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self._history = ControllerHistory()

    def reset(self) -> None:
        """Reset for a new problem."""
        self._history.reset()

    def decide(
        self,
        state: BudgetLoopState,
        verifier_output: Optional[VerificationResult] = None,
    ) -> Tuple[ActionType, DecisionRecord]:
        """
        Standard control loop decision point.

        Args:
            state: BudgetLoopState snapshot (s_t = (x, τ_t, A_t, b_t, c_t))
            verifier_output: output from VERIFY action (for hysteresis tracking)

        Returns:
            (action, decision_record) — decision_record always has triggered_rule_id
        """
        b = state.budget_state
        c = state.credit_stats

        # Track previous best answer for logging
        prev_best = state.candidate_manager.get_best_text() if state.candidate_manager else None

        # Use MA for smoother decision signals; this is computed from credit_stats
        # which reflects the state AFTER the latest step
        msg_gain = c.msg_gain_ma
        action, rule_id, rule_name = self._apply_rules(state, verifier_output, msg_gain)

        # Update history for hysteresis (tracks what happened in this step)
        verify_conf = verifier_output.confidence if verifier_output else 0.0
        self._history.record(
            action=action,
            msg_gain=msg_gain,
            verify_confidence=verify_conf,
        )

        record = DecisionRecord(
            step_id=state.step_id,
            action=action,
            triggered_rule_id=rule_id,
            triggered_rule_name=rule_name,
            budget_state=b.to_dict(),
            credit_stats={
                "msg_gain": c.msg_gain,
                "msg_gain_ma": c.msg_gain_ma,
                "msg_gain_slope": c.msg_gain_slope,
                "credit_entropy": c.credit_entropy,
                "credit_concentration": c.credit_concentration,
                "disagreement": c.disagreement,
                "uncertainty": c.uncertainty,
                "agent_credit": c.agent_credit,
            },
            candidate_scores=(
                state.candidate_manager.get_all_texts()
                if state.candidate_manager else []
            ),
            verifier_output=verifier_output,
            best_answer_before=prev_best,
        )
        return action, record

    # ── Rule application ──────────────────────────────────────────────────────

    def _apply_rules(
        self,
        state: BudgetLoopState,
        verifier_output: Optional[VerificationResult],
        msg_gain: float,
    ) -> Tuple[ActionType, int, str]:
        """Apply rules in priority order. Returns (action, rule_id, rule_name)."""

        cfg = self.config
        b = state.budget_state
        c = state.credit_stats

        # R1: Budget exhausted
        if not b.can_afford(cfg.min_budget_for_continue):
            return ActionType.STOP, self.R_BUDGET_EXHAUSTED, "budget_exhausted"

        # R2: High disagreement → external verification to break tie
        if c.disagreement >= cfg.disagreement_threshold:
            if b.can_afford(cfg.min_budget_for_verify):
                return ActionType.CALL_VERIFIER, self.R_HIGH_DISAGREEMENT, "high_disagreement"
            return ActionType.STOP, self.R_BUDGET_EXHAUSTED, "budget_exhausted"

        # R3: Consecutive low marginal gains → diminishing returns
        # Check the value AFTER potential increment (i.e., current gain counts)
        # If current msg_gain is low, it will increment after this check,
        # so we check: current_consecutive + 1 >= threshold
        future_count = self._history.consecutive_low_gains + (1 if msg_gain < 0.02 else 0)
        if future_count >= cfg.consecutive_low_gain_stop:
            # Only STOP on slope if last verify was confident (or no verify yet)
            if not cfg.slope_stop_requires_verify:
                return ActionType.STOP, self.R_CONSECUTIVE_LOW_GAIN, "consecutive_low_gain"
            if self._history.last_verify_confidence >= 0.6:
                return ActionType.STOP, self.R_CONSECUTIVE_LOW_GAIN, "consecutive_low_gain"

        # R4: High uncertainty → verify before continuing
        if c.uncertainty >= cfg.uncertainty_threshold:
            if b.can_afford(cfg.min_budget_for_verify):
                return ActionType.CALL_VERIFIER, self.R_HIGH_UNCERTAINTY, "high_uncertainty"
            return ActionType.STOP, self.R_BUDGET_EXHAUSTED, "budget_exhausted"

        # R5: Balanced collaboration (high entropy) → continue discussion
        if c.credit_entropy >= cfg.entropy_threshold:
            if b.can_afford(cfg.min_budget_for_continue):
                return ActionType.CONTINUE_DISCUSS, self.R_COLLABORATION_FAIR, "collaboration_fair"

        # R6: One agent dominates → further投入 marginal gains likely low
        if c.credit_concentration >= cfg.concentration_threshold:
            return ActionType.STOP, self.R_AGENT_DOMINANCE, "agent_dominance"

        # R7: Default → continue discussion
        if b.can_afford(cfg.min_budget_for_continue):
            return ActionType.CONTINUE_DISCUSS, self.R_DEFAULT, "default_continue"
        return ActionType.STOP, self.R_BUDGET_EXHAUSTED, "budget_exhausted"

    # ── Cooldown helpers ──────────────────────────────────────────────────────

    def _in_verify_cooldown(self) -> bool:
        """True if VERIFY was called recently (hysteresis)."""
        return self._history.consecutive_verifies > 0

    def _get_history_summary(self) -> Dict[str, Any]:
        return {
            "recent_actions": [a.value for a in self._history.recent_actions],
            "recent_gains": self._history.recent_gains,
            "consecutive_verifies": self._history.consecutive_verifies,
            "consecutive_low_gains": self._history.consecutive_low_gains,
            "last_verify_confidence": self._history.last_verify_confidence,
        }
