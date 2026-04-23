"""BudgetOrchestrator — budget-aware multi-agent collaboration orchestrator.

Implements the standard budgeted control loop:
    s_t = (x, τ_t, A_t, b_t, c_t)
          │
          ▼
    BudgetController.decide(s_t, c_t)   ← BEFORE action!
          │
     action
          │
     ├─► STOP:      return best_answer
     │
     ├─► CALL_VERIFIER:
     │      verifier_output = verifier.run()
     │      update τ_t, c_t, b_t
     │      continue
     │
     └─► CONTINUE_DISCUSS:
            Orchestrator.step() → single atomic action
            update τ_t, c_t, b_t
            continue

Key design (per user correction):
  - Orchestrator.step() executes ONE atomic action, not full pipeline
  - BudgetController decides BEFORE execution
  - VERIFY is a real action that produces a new observation
  - Agent selection in step() is FIXED SCHEDULE (not hidden intelligence)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from tiny_agents.budget.candidate_manager import CandidateManager
from tiny_agents.budget.credit_tracer import CreditTracer
from tiny_agents.budget.controller import BudgetController
from tiny_agents.budget.logger import ExperimentLogger
from tiny_agents.budget.state_builder import StateBuilder
from tiny_agents.budget.types import (
    ActionType,
    BudgetState,
    BudgetLoopState,
    CollaborationStep,
    CreditStats,
    RunResult,
    VerificationResult,
    VerdictType,
)
from tiny_agents.agents.verifier import VerifierAgent


class BudgetOrchestrator:
    """
    Budget-aware multi-agent collaboration orchestrator.

    Usage:
        orch = BudgetOrchestrator(...)
        result = await orch.run(problem={"question": "..."}, budget=5000)
    """

    # Phase 1 fixed agent schedule (no hidden intelligence)
    # (agent_name, action_type) pairs in execution order
    PHASE1_SCHEDULE: List[Tuple[str, str]] = [
        ("reasoner", "REASONER_STEP"),   # Step 1: initial reasoning
        ("critic", "CRITIC_STEP"),       # Step 2: critique
        ("reasoner", "REASONER_STEP"),   # Step 3: respond to critique
        ("critic", "CRITIC_STEP"),       # Step 4: re-evaluate
        ("reasoner", "REASONER_STEP"),   # Step 5: final reasoning
    ]

    def __init__(
        self,
        llm_backend: Any,                    # vLLM backend for all agents
        reasoner_agent_name: str = "reasoner",
        critic_agent_name: str = "critic",
        verifier_agent: Optional[VerifierAgent] = None,
        credit_tracer: Optional[CreditTracer] = None,
        budget_controller: Optional[BudgetController] = None,
        agent_backends: Optional[Dict[str, Any]] = None,  # agent -> backend
        experiment_logger: Optional[ExperimentLogger] = None,
    ):
        """
        Args:
            llm_backend: default vLLM backend for agents
            reasoner_agent_name: name of the reasoner agent
            critic_agent_name: name of the critic agent
            verifier_agent: VerifierAgent instance (can be None for Phase 1 no-verify)
            credit_tracer: CreditTracer instance
            budget_controller: BudgetController instance
            agent_backends: dict mapping agent_name -> specific backend
            experiment_logger: ExperimentLogger for structured logging
        """
        self.llm_backend = llm_backend
        self.reasoner_name = reasoner_agent_name
        self.critic_name = critic_agent_name
        self.verifier = verifier_agent
        self.credit_tracer = credit_tracer or CreditTracer()
        self.controller = budget_controller or BudgetController()
        self.agent_backends = agent_backends or {}
        self.logger = experiment_logger

        # Candidate management
        self.candidate_mgr = CandidateManager()

        # Agent schedule state
        self._schedule_index = 0
        self._has_seen_verifier_output = False  # for post-verify reasoner responses

    # ── Main entry ─────────────────────────────────────────────────────────────

    async def run(
        self,
        problem: Dict[str, Any],
        budget: int,
        ground_truth: Optional[str] = None,
        problem_id: Optional[str] = None,
        task_type: str = "math",
    ) -> RunResult:
        """
        Execute budget-aware multi-agent collaboration.

        Args:
            problem: {"question": "...", ...}
            budget: token-equivalent budget B
            ground_truth: for final correctness evaluation (optional)
            problem_id: for logging (optional)
            task_type: "math" | "gsm8k" | "exact"

        Returns:
            RunResult with answer, trajectory, credit stats, budget state
        """
        # Initialize
        problem_id = problem_id or f"p_{time.time():.0f}"
        if self.logger:
            self.logger.start_problem(problem_id)

        self.credit_tracer.reset()
        self.controller.reset()
        self.candidate_mgr.reset()
        self._schedule_index = 0
        self._has_seen_verifier_output = False

        budget_state = BudgetState(total_budget=budget)
        trajectory: List[CollaborationStep] = []
        state_builder = StateBuilder(problem, budget)

        step_id = 0
        stop_reason = "max_iterations"
        last_verifier_output: Optional[VerificationResult] = None

        # Initial quality score Q_0 = 0
        current_quality = 0.0

        while budget_state.can_afford(50):  # minimum cost to do anything

            # ── Build state for controller ────────────────────────────────────
            credit_stats = self.credit_tracer.compute_stats()
            state = state_builder.build_state(
                trajectory=trajectory,
                active_agents=[self.reasoner_name, self.critic_name],
                credit_stats=credit_stats,
                candidate_manager=self.candidate_mgr,
                budget_state=budget_state,
            )

            # ── Controller decides BEFORE execution ────────────────────────────
            action, decision_record = self.controller.decide(state, last_verifier_output)

            # ── Execute action ─────────────────────────────────────────────────
            if action == ActionType.STOP:
                stop_reason = decision_record.triggered_rule_name or "controller_stop"
                break

            elif action == ActionType.CALL_VERIFIER:
                step_id += 1
                verify_result = await self._execute_verify(
                    problem=problem["question"],
                    trajectory=trajectory,
                )
                last_verifier_output = verify_result
                current_quality = verify_result.quality_score

                # VERIFY is a real action: creates a step and updates credit
                verify_step = CollaborationStep(
                    step_id=step_id,
                    agent_name="verifier",
                    action_type=ActionType.CALL_VERIFIER,
                    content=verify_result.feedback,
                    answer_candidate=self.candidate_mgr.get_best_text(),
                    token_cost=verify_result.token_cost,
                    latency_ms=verify_result.latency_ms,
                )
                trajectory.append(verify_step)
                self.credit_tracer.add_step(verify_step, current_quality)
                budget_state.deduct(verify_result.token_cost, verify_result.latency_ms)

                # Update candidate manager with verifier quality
                if self.candidate_mgr.best:
                    _, _ = self.candidate_mgr.add_candidate(
                        text=self.candidate_mgr.best.text,
                        score=current_quality,
                        step_id=step_id,
                        agent_name="verifier",
                    )

                # Log decision
                if self.logger:
                    self.logger.log_decision(
                        problem_id=problem_id,
                        decision=decision_record,
                        cost_spent=verify_result.token_cost,
                        best_answer_after=self.candidate_mgr.get_best_text(),
                        trajectory_length=len(trajectory),
                    )

                self._has_seen_verifier_output = True
                continue

            elif action == ActionType.CONTINUE_DISCUSS:
                step_id += 1

                # Execute one atomic collaboration step
                step = await self._execute_atomic_step(
                    problem=problem,
                    budget_state=budget_state,
                )

                trajectory.append(step)

                # Update candidate if step produced an answer
                if step.answer_candidate:
                    _, _ = self.candidate_mgr.add_candidate(
                        text=step.answer_candidate,
                        score=current_quality,  # quality updated by verifier or stays same
                        step_id=step_id,
                        agent_name=step.agent_name,
                    )

                # CreditTracer update (quality score is from last verifier or 0 if first step)
                self.credit_tracer.add_step(step, current_quality)
                budget_state.deduct(step.token_cost, step.latency_ms)

                # Log decision
                if self.logger:
                    self.logger.log_decision(
                        problem_id=problem_id,
                        decision=decision_record,
                        cost_spent=step.token_cost,
                        best_answer_after=self.candidate_mgr.get_best_text(),
                        trajectory_length=len(trajectory),
                    )

                # After verifier feedback, reset to allow controller to re-evaluate
                if self._has_seen_verifier_output:
                    last_verifier_output = None
                    self._has_seen_verifier_output = False

                continue

        # ── Final result ──────────────────────────────────────────────────────
        final_answer = self.candidate_mgr.get_best_text() or ""

        # Final correctness (deterministic)
        is_correct = False
        if ground_truth and final_answer:
            is_correct = VerifierAgent.check_final_correctness(
                final_answer, ground_truth, task_type
            )

        result = RunResult(
            answer=final_answer,
            trajectory=trajectory,
            credit_stats=self.credit_tracer.compute_stats(),
            budget_state=budget_state,
            stop_reason=stop_reason,
            is_correct=is_correct,
        )

        if self.logger:
            self.logger.log_run(
                result=result,
                problem_id=problem_id,
                ground_truth=ground_truth or "",
                problem_text=problem.get("question", ""),
            )

        return result

    # ── Atomic step execution ─────────────────────────────────────────────────

    async def _execute_atomic_step(
        self,
        problem: Dict[str, Any],
        budget_state: BudgetState,
    ) -> CollaborationStep:
        """
        Execute ONE atomic collaboration action.

        Phase 1: fixed schedule, no hidden intelligence.
        Schedule: REASONER → CRITIC → REASONER → CRITIC → REASONER → ...

        After VERIFY: reasoner responds to verifier feedback.

        Returns:
            CollaborationStep with content, token_cost, answer_candidate, etc.
        """
        start = time.time()
        step_id = len(self.credit_tracer.trajectory) + 1

        # Determine which agent to run based on fixed schedule
        if self._has_seen_verifier_output and self.candidate_mgr.best:
            # After verifier feedback: reasoner responds
            agent_name = self.reasoner_name
            action_type = "REASONER_STEP"
            instruction = f"Based on the verifier's feedback, revise your answer if needed.\n\nVerifier said: {self.candidate_mgr.best.text}"
        else:
            # Fixed schedule
            agent_name, action_type = self._get_next_in_schedule()

        # Build prompt for the agent
        prompt = self._build_step_prompt(
            problem=problem,
            agent_name=agent_name,
            action_type=action_type,
        )

        # Call agent
        backend = self.agent_backends.get(agent_name, self.llm_backend)
        response = await self._call_agent(backend, agent_name, prompt)

        latency = (time.time() - start) * 1000
        token_cost = self._estimate_token_cost(prompt, response)

        # Extract answer candidate from response
        answer_candidate = self._extract_answer(response)

        return CollaborationStep(
            step_id=step_id,
            agent_name=agent_name,
            action_type=ActionType.CONTINUE_DISCUSS,
            content=response,
            answer_candidate=answer_candidate,
            token_cost=token_cost,
            latency_ms=latency,
        )

    def _get_next_in_schedule(self) -> Tuple[str, str]:
        """
        Phase 1 fixed schedule. No hidden intelligence.

        Schedule pattern:
          odd steps  → reasoner
          even steps → critic

        Returns (agent_name, action_type)
        """
        current_step = len(self.credit_tracer.trajectory) + 1

        if current_step % 2 == 1:
            return (self.reasoner_name, "REASONER_STEP")
        else:
            return (self.critic_name, "CRITIC_STEP")

    def _build_step_prompt(
        self,
        problem: Dict[str, Any],
        agent_name: str,
        action_type: str,
    ) -> str:
        """Build the prompt for a single atomic step."""
        question = problem.get("question", "")

        if agent_name == self.reasoner_name:
            if "critic" in action_type.lower():
                return f"""You are a reasoning agent. Given the problem and any critique, provide your reasoning and updated answer.

Problem: {question}

Provide your reasoning and final answer. Format your final answer clearly."""
            else:
                return f"""You are a reasoning agent. Solve the problem step by step.

Problem: {question}

Provide your reasoning and final answer."""

        elif agent_name == self.critic_name:
            return f"""You are a critical reviewer. Evaluate the current reasoning.

Problem: {question}

Review the reasoning and provide critique. Identify any errors or weaknesses."""

        return f"Problem: {question}\n\n"

    async def _call_agent(
        self,
        backend: Any,
        agent_name: str,
        prompt: str,
    ) -> str:
        """Call an agent's LLM backend."""
        messages = [{"role": "user", "content": prompt}]
        response = backend.chat(messages)
        if hasattr(response, "choices"):
            return response.choices[0].message.content
        return str(response)

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract final answer from agent response."""
        # Simple extraction: look for "The answer is X" or last line with content
        # Phase 1: just return the full response as candidate
        # Better extraction can be added later
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[-1] if lines else response[:200]

    async def _execute_verify(
        self,
        problem: str,
        trajectory: List[CollaborationStep],
    ) -> VerificationResult:
        """Execute VERIFY action — produces a new observation."""
        if self.verifier is None:
            # No verifier: return UNCERTAIN with default values
            return VerificationResult(
                verdict=VerdictType.UNCERTAIN,
                confidence=0.5,
                quality_score=0.5,
                feedback="No verifier available",
                token_cost=50,
                latency_ms=0.0,
            )

        current_best = self.candidate_mgr.get_best_text() or ""
        result = await self.verifier.verify(
            problem=problem,
            candidate_answer=current_best,
            trajectory=trajectory,
        )
        return result

    def _estimate_token_cost(self, prompt: str, response: str) -> int:
        """Rough token estimation."""
        return (len(prompt) + len(response)) // 4
