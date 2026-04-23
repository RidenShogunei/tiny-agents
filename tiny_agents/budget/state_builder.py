"""StateBuilder — independent state construction layer.

Prevents field drift and circular dependencies between
CreditTracer, CandidateManager, BudgetController, and BudgetOrchestrator.
All modules consume state from here; none build it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from tiny_agents.budget.types import (
    ActionType,
    BudgetState,
    CreditStats,
)

if TYPE_CHECKING:
    from tiny_agents.budget.candidate_manager import CandidateManager
    from tiny_agents.budget.credit_tracer import CreditTracer


@dataclass
class BudgetLoopState:
    """
    Complete state snapshot at each control loop iteration.
    
    s_t = (x, τ_t, A_t, b_t, c_t) as defined in the paper.
    
    This is the ONLY state object passed to BudgetController.decide().
    """
    problem: Dict[str, Any]         # x
    trajectory: List[Any]            # τ_t  (list of CollaborationStep)
    active_agents: List[str]          # A_t
    budget_state: BudgetState         # b_t
    credit_stats: CreditStats        # c_t
    step_id: int = 0                # current step counter
    remaining_steps: int = 0         # steps remaining in budget (estimate)
    
    # Additional read-only context for convenience
    candidate_manager: Optional[Any] = None  # set by orchestrator before decide()


class StateBuilder:
    """
    Constructs and maintains BudgetLoopState across the control loop.
    
    Usage:
        builder = StateBuilder(problem, budget)
        builder.reset()
        
        # After each step:
        builder.build_state(
            trajectory=trajectory,
            active_agents=["reasoner", "critic"],
            credit_stats=credit_tracer.compute_stats(),
            candidate_manager=candidate_mgr,
        )
        state = builder.get_state()
        
        # Pass state to controller:
        action = controller.decide(state)
    """

    def __init__(self, problem: Dict[str, Any], total_budget: int):
        """
        Args:
            problem: the input problem dict
            total_budget: token-equivalent budget B
        """
        self._problem = problem
        self._total_budget = total_budget
        self._step_id = 0
        self._state: Optional[BudgetLoopState] = None

    def reset(self) -> None:
        """Reset for a new problem."""
        self._step_id = 0
        self._state = None

    def build_state(
        self,
        trajectory: List[Any],
        active_agents: List[str],
        credit_stats: CreditStats,
        candidate_manager: Optional["CandidateManager"] = None,
        budget_state: Optional[BudgetState] = None,
    ) -> BudgetLoopState:
        """
        Build the current iteration state snapshot.

        Args:
            trajectory: list of CollaborationStep
            active_agents: names of currently active agents
            credit_stats: CreditStats from CreditTracer
            candidate_manager: CandidateManager instance
            budget_state: existing BudgetState (creates new if None)
        """
        self._step_id += 1

        if budget_state is None:
            budget_state = BudgetState(total_budget=self._total_budget)
            # Deduct for any already-spent steps
            spent = sum(s.token_cost for s in trajectory)
            budget_state.deduct(spent)

        # Estimate remaining steps based on avg cost per step
        avg_cost_per_step = 100  # rough estimate; will be refined by actual data
        remaining = budget_state.remaining
        remaining_steps = remaining // avg_cost_per_step if remaining > 0 else 0

        self._state = BudgetLoopState(
            problem=self._problem,
            trajectory=trajectory,
            active_agents=active_agents,
            budget_state=budget_state,
            credit_stats=credit_stats,
            step_id=self._step_id,
            remaining_steps=remaining_steps,
            candidate_manager=candidate_manager,
        )
        return self._state

    def get_state(self) -> BudgetLoopState:
        """Get the current state snapshot. Raises if not yet built."""
        if self._state is None:
            raise RuntimeError("StateBuilder.get_state() called before build_state()")
        return self._state

    def get_step_id(self) -> int:
        return self._step_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state for logging."""
        if self._state is None:
            return {}
        return {
            "step_id": self._state.step_id,
            "problem": self._state.problem,
            "budget_state": self._state.budget_state.to_dict(),
            "credit_stats": self._credit_stats_to_dict(self._state.credit_stats),
            "active_agents": self._state.active_agents,
            "remaining_steps": self._state.remaining_steps,
        }

    def _credit_stats_to_dict(self, cs: CreditStats) -> Dict[str, Any]:
        return {
            "msg_gain": cs.msg_gain,
            "msg_gain_ma": cs.msg_gain_ma,
            "msg_gain_slope": cs.msg_gain_slope,
            "agent_credit": cs.agent_credit,
            "agent_activity_ratio": cs.agent_activity_ratio,
            "credit_entropy": cs.credit_entropy,
            "credit_concentration": cs.credit_concentration,
            "disagreement": cs.disagreement,
            "uncertainty": cs.uncertainty,
        }
