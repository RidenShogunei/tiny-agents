"""ExperimentLogger — structured logging for budget-aware experiments.

Logs per decision:
  - state_features
  - credit_features
  - chosen_action
  - triggered_rule_id
  - cost
  - best_answer_before/after
  - verifier_output
  - msg_gain_before/after

Enables failure analysis: which rules fire, which never fire, etc.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tiny_agents.budget.types import (
    ActionType,
    BudgetState,
    CollaborationStep,
    CreditStats,
    DecisionRecord,
    RunResult,
    VerificationResult,
    VerdictType,
)


class ExperimentLogger:
    """Collects and persists experiment results and decision logs."""

    def __init__(self, output_dir: str = "./budget_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._run_id = uuid.uuid4().hex[:8]
        self._run_timestamp = datetime.now().isoformat()
        self._results: List[Dict[str, Any]] = []
        self._decision_logs: List[Dict[str, Any]] = []
        self._current_problem_id: Optional[str] = None

    def start_problem(self, problem_id: str) -> None:
        """Mark the start of a new problem in the current run."""
        self._current_problem_id = problem_id

    def log_decision(
        self,
        problem_id: str,
        decision: DecisionRecord,
        cost_spent: int,
        best_answer_after: Optional[str],
        trajectory_length: int,
    ) -> None:
        """
        Log a single BudgetController.decide() call.
        
        Required for failure analysis:
          - which rules trigger most often
          - which rules never fire
          - correlation between rule and final correctness
        """
        record = {
            "problem_id": problem_id,
            "run_id": self._run_id,
            "timestamp": datetime.now().isoformat(),
            # Decision
            "action": decision.action.value,
            "triggered_rule_id": decision.triggered_rule_id,
            "triggered_rule_name": decision.triggered_rule_name,
            # Budget
            "budget_state": decision.budget_state,
            "cost_spent": cost_spent,
            # Credit features
            "credit_stats": decision.credit_stats,
            # Candidate state
            "candidate_scores": decision.candidate_scores,
            "best_answer_before": decision.best_answer_before,
            "best_answer_after": best_answer_after,
            # Verifier
            "verifier_output": _verifier_to_dict(decision.verifier_output),
            # Trajectory state
            "trajectory_length": trajectory_length,
            "msg_gain_before": decision.msg_gain_before,
            "msg_gain_after": decision.msg_gain_after,
        }
        self._decision_logs.append(record)

    def log_run(
        self,
        result: RunResult,
        problem_id: str,
        ground_truth: str,
        problem_text: str,
    ) -> None:
        """Log the final result of a complete BudgetOrchestrator.run()."""
        record = {
            "problem_id": problem_id,
            "run_id": self._run_id,
            "run_timestamp": self._run_timestamp,
            # Problem
            "problem_text": problem_text,
            "ground_truth": ground_truth,
            # Result
            "predicted_answer": result.answer,
            "is_correct": result.is_correct,
            "stop_reason": result.stop_reason,
            # Budget
            "budget_total": result.budget_state.total_budget,
            "budget_spent": result.budget_state.spent,
            "budget_remaining": result.budget_state.remaining,
            "latency_ms": result.budget_state.latency_ms,
            # Credit stats (final)
            "final_credit_stats": _credit_stats_to_dict(result.credit_stats),
            # Trajectory
            "num_steps": len(result.trajectory),
            "agents_used": list(set(s.agent_name for s in result.trajectory)),
            # Dollar cost estimate (rough: $0.01 per 1K tokens)
            "estimated_dollar_cost": result.budget_state.spent / 100_000 * 0.01,
        }
        self._results.append(record)

    def log_decision_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics over all logged decisions.
        Useful for rule analysis and failure diagnosis.
        """
        if not self._decision_logs:
            return {}

        from collections import Counter

        rule_counter = Counter(
            r["triggered_rule_name"] for r in self._decision_logs
        )
        action_counter = Counter(r["action"] for r in self._decision_logs)

        return {
            "total_decisions": len(self._decision_logs),
            "rule_trigger_counts": dict(rule_counter),
            "action_counts": dict(action_counter),
            "rules_never_triggered": self._find_never_triggered_rules(rule_counter),
        }

    def _find_never_triggered_rules(self, triggered: Dict[str, int]) -> List[str]:
        all_rules = [
            "budget_exhausted", "high_disagreement", "consecutive_low_gain",
            "high_uncertainty", "collaboration_fair", "agent_dominance", "default_continue",
        ]
        return [r for r in all_rules if r not in triggered]

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all completed runs."""
        if not self._results:
            return {}

        n = len(self._results)
        n_correct = sum(1 for r in self._results if r["is_correct"])
        total_spent = sum(r["budget_spent"] for r in self._results)
        total_latency = sum(r["latency_ms"] for r in self._results)

        # Budget efficiency: accuracy per token
        avg_spent = total_spent / n if n > 0 else 0
        avg_latency = total_latency / n if n > 0 else 0

        return {
            "n_problems": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n > 0 else 0.0,
            "avg_budget_spent": avg_spent,
            "avg_latency_ms": avg_latency,
            "decision_stats": self.log_decision_stats(),
        }

    def save(self, suffix: str = "") -> str:
        """
        Persist all logs to disk.
        
        Returns the output file paths.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{suffix}" if suffix else ""

        # Save run results
        results_path = self.output_dir / f"runs_{self._run_id}{suffix}.jsonl"
        with open(results_path, "w") as f:
            for r in self._results:
                f.write(json.dumps(r) + "\n")

        # Save decision logs
        decisions_path = self.output_dir / f"decisions_{self._run_id}{suffix}.jsonl"
        with open(decisions_path, "w") as f:
            for d in self._decision_logs:
                f.write(json.dumps(d) + "\n")

        return str(results_path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _verifier_to_dict(v: Optional[VerificationResult]) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    return {
        "verdict": v.verdict.value,
        "confidence": v.confidence,
        "quality_score": v.quality_score,
        "feedback": v.feedback,
        "token_cost": v.token_cost,
        "latency_ms": v.latency_ms,
    }


def _credit_stats_to_dict(cs: CreditStats) -> Dict[str, Any]:
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
