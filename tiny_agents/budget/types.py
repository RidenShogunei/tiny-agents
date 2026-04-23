"""Core types for budget-aware multi-agent collaboration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ── Action Types ─────────────────────────────────────────────────────────────

class ActionType(Enum):
    """Budget controller action space."""
    CONTINUE_DISCUSS = "CONTINUE_DISCUSS"
    CALL_VERIFIER = "CALL_VERIFIER"
    STOP = "STOP"


class VerdictType(Enum):
    """Verifier three-valued output."""
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNCERTAIN = "UNCERTAIN"


# ── Collaboration Step ────────────────────────────────────────────────────────

@dataclass
class CollaborationStep:
    """A single atomic collaboration action in the trajectory."""
    step_id: int
    agent_name: str
    action_type: ActionType
    content: str
    answer_candidate: Optional[str] = None
    token_cost: int = 0
    latency_ms: float = 0.0
    timestamp: float = 0.0

    # Credit fields (populated by CreditTracer)
    msg_gain: float = 0.0          # C_msg_t = Q_t - Q_{t-1}
    agent_credit: float = 0.0      # aggregated from msg_gains


# ── Credit Stats ─────────────────────────────────────────────────────────────

@dataclass
class CreditStats:
    """
    Hierarchical credit statistics computed from trajectory.
    
    msg_gain_* are primary signals.
    agent_credit is derived by aggregating msg_gains per agent.
    agent_activity_ratio is a debug feature only (NOT credit).
    """
    # Message-level (primary)
    msg_gain: float = 0.0           # Q_t - Q_{t-1}, latest step
    msg_gain_ma: float = 0.0         # moving average of last K msg_gains
    msg_gain_slope: float = 0.0      # linear slope of recent msg_gains

    # Agent-level (aggregated from msg_gains)
    agent_credit: Dict[str, float] = field(default_factory=dict)
    
    # Debug feature only — NOT credit, rename from step_share
    agent_activity_ratio: Dict[str, float] = field(default_factory=dict)

    # Derived signals
    credit_entropy: float = 0.0     # H(agent_credit) — high = balanced
    credit_concentration: float = 0.0  # max_i credit_i — high = one agent dominates

    # Trajectory-level
    disagreement: float = 0.0        # fraction of divergent answer candidates
    uncertainty: float = 1.0         # current answer uncertainty (from verifier or consistency)


# ── Budget State ─────────────────────────────────────────────────────────────

@dataclass
class BudgetState:
    """Tracks token-equivalent budget consumption."""
    total_budget: int
    remaining: int = 0
    spent: int = 0
    latency_ms: float = 0.0

    def __post_init__(self):
        # Only initialize remaining/spent if not explicitly provided (i.e., at default 0)
        # This allows BudgetState(total_budget=1000, remaining=500) to work
        if self.remaining == 0:
            self.remaining = self.total_budget
        if self.spent == 0 and self.remaining == self.total_budget:
            self.spent = 0

    def can_afford(self, cost: int) -> bool:
        return self.remaining >= cost

    def deduct(self, cost: int, latency: float = 0.0) -> None:
        self.remaining = max(0, self.remaining - cost)
        self.spent += cost
        self.latency_ms += latency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "remaining": self.remaining,
            "spent": self.spent,
            "latency_ms": self.latency_ms,
        }


# ── Verification Result ──────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Output from a verifier agent."""
    verdict: VerdictType
    confidence: float             # [0.0, 1.0]
    quality_score: float          # Q_t, used as msg_gain signal
    feedback: str = ""
    token_cost: int = 0
    latency_ms: float = 0.0


# ── Run Result ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Final output from BudgetOrchestrator.run()."""
    answer: str
    trajectory: List[CollaborationStep]
    credit_stats: CreditStats
    budget_state: BudgetState
    stop_reason: str
    is_correct: bool = False


# ── Decision Record ──────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """Log entry for a single BudgetController.decide() call."""
    step_id: int
    action: ActionType
    triggered_rule_id: Optional[int] = None
    triggered_rule_name: Optional[str] = None
    # Input features
    budget_state: Dict[str, Any] = field(default_factory=dict)
    credit_stats: Dict[str, Any] = field(default_factory=dict)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    # Output
    verifier_output: Optional[VerificationResult] = None
    msg_gain_before: float = 0.0
    msg_gain_after: float = 0.0
    best_answer_before: Optional[str] = None
    best_answer_after: Optional[str] = None


# ── Budget Loop State ─────────────────────────────────────────────────────────

@dataclass
class BudgetLoopState:
    """
    Complete state snapshot at each control loop iteration.

    s_t = (x, τ_t, A_t, b_t, c_t) as defined in the paper.

    This is the ONLY state object passed to BudgetController.decide().
    """
    problem: Dict[str, Any]         # x
    trajectory: List  # τ_t  (list of CollaborationStep)
    active_agents: List[str]          # A_t
    budget_state: BudgetState         # b_t
    credit_stats: CreditStats        # c_t
    step_id: int = 0                # current step counter
    remaining_steps: int = 0         # steps remaining in budget (estimate)
    # Additional read-only context for convenience
    candidate_manager: Optional[Any] = None  # set by orchestrator before decide()
