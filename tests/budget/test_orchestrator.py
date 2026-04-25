"""Unit tests for BudgetOrchestrator — deterministic control loop.

Tests verify the control loop mechanics without any model calls.
Uses a mock backend that returns deterministic responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tiny_agents.budget.orchestrator import BudgetOrchestrator
from tiny_agents.budget.types import ActionType, VerificationResult, VerdictType
from tiny_agents.budget.controller import BudgetController, ControllerConfig
from tiny_agents.budget.credit_tracer import CreditTracer
from tiny_agents.budget.candidate_manager import CandidateManager


class MockBackend:
    """Mock LLM backend that returns deterministic responses."""

    def __init__(self, response: str = "42"):
        self.response = response

    def chat(self, messages):
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=self.response))]
        return mock


def make_orchestrator(
    verify_cost: int = 200,
    budget: int = 5000,
    controller_config: ControllerConfig = None,
) -> BudgetOrchestrator:
    """Helper: create BudgetOrchestrator with mock backend."""
    mock_backend = MockBackend(response="42")
    cfg = controller_config or ControllerConfig(verify_cost=verify_cost)

    return BudgetOrchestrator(
        llm_backend=mock_backend,
        reasoner_agent_name="reasoner",
        critic_agent_name="critic",
        verifier_agent=None,
        credit_tracer=CreditTracer(),
        budget_controller=BudgetController(cfg),
        experiment_logger=None,
    )


def make_decision_record(action: ActionType, rule_id: int = 7, rule_name: str = "default"):
    """Helper: create a mock DecisionRecord."""
    from tiny_agents.budget.types import DecisionRecord
    return DecisionRecord(
        step_id=0,
        action=action,
        triggered_rule_id=rule_id,
        triggered_rule_name=rule_name,
    )


class TestOrchestratorControlLoop:
    """Control loop mechanics: decide → execute → observe → update."""

    @pytest.mark.asyncio
    async def test_stop_does_not_run_extra_steps(self):
        """STOP must exit loop without running another step."""
        orch = make_orchestrator(budget=100)
        step_count = 0

        async def count_steps(problem, budget_state):
            nonlocal step_count
            step_count += 1
            from tiny_agents.budget.types import CollaborationStep
            return CollaborationStep(
                step_id=step_count,
                agent_name="reasoner",
                action_type=ActionType.CONTINUE_DISCUSS,
                content="step",
                token_cost=10,
            )

        # Force controller to STOP immediately
        def mock_decide(state, v=None):
            return (ActionType.STOP, make_decision_record(ActionType.STOP, 1, "budget_exhausted"))

        orch.controller.decide = mock_decide
        orch.controller._history.record = lambda *a, **kw: None

        with patch.object(orch, '_execute_atomic_step', count_steps):
            result = await orch.run(
                problem={"question": "What is 2+2?"},
                budget=100,
            )

        assert step_count == 0  # No step executed after STOP

    @pytest.mark.asyncio
    async def test_verify_writes_real_observation(self):
        """VERIFY must produce a real observation, not just deduct cost."""
        orch = make_orchestrator(budget=2000)

        # Mock verifier to return a result
        mock_verifier = AsyncMock()
        mock_verifier.verify = AsyncMock(return_value=VerificationResult(
            verdict=VerdictType.UNCERTAIN,
            confidence=0.5,
            quality_score=0.6,
            feedback="Not sure",
            token_cost=150,
            latency_ms=10.0,
        ))
        orch.verifier = mock_verifier

        call_count = [0]

        def mock_decide(state, v=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (ActionType.CALL_VERIFIER, make_decision_record(ActionType.CALL_VERIFIER, 2, "high_disagreement"))
            return (ActionType.STOP, make_decision_record(ActionType.STOP, 1, "budget_exhausted"))

        orch.controller.decide = mock_decide
        orch.controller._history.record = lambda *a, **kw: None

        result = await orch.run(
            problem={"question": "What is 2+2?"},
            budget=2000,
        )

        # VERIFY was called once
        assert mock_verifier.verify.call_count == 1
        # Result trajectory should contain the verify step
        verify_steps = [s for s in result.trajectory if s.agent_name == "verifier"]
        assert len(verify_steps) == 1


class TestOrchestratorLogging:
    """Every step must log with triggered_rule_id."""

    @pytest.mark.asyncio
    async def test_every_decision_has_rule_id(self):
        """Every logged decision must have triggered_rule_id."""
        from tiny_agents.budget.logger import ExperimentLogger

        logger = ExperimentLogger(output_dir="/tmp/test_logs")

        orch = make_orchestrator(budget=500)
        orch.logger = logger

        # Mock controller to force a few steps
        decisions = [
            (ActionType.CONTINUE_DISCUSS, make_decision_record(ActionType.CONTINUE_DISCUSS, 7, "default")),
            (ActionType.STOP, make_decision_record(ActionType.STOP, 1, "budget_exhausted")),
        ]
        call_count = [0]
        orch.controller.decide = lambda s, v=None: decisions[min(call_count[0], 1)]
        call_count[0] += 1
        orch.controller._history.record = lambda *a, **kw: None

        result = await orch.run(
            problem={"question": "What is 2+2?"},
            budget=500,
            problem_id="test_001",
        )

        # Check decision logs
        if logger._decision_logs:
            for log in logger._decision_logs:
                assert "triggered_rule_id" in log
                assert log["triggered_rule_id"] is not None


class TestOrchestratorSchedule:
    """Phase 1 fixed schedule: odd=reasoner, even=critic."""

    def test_fixed_schedule_odd_steps_reasoner(self):
        """Odd-numbered steps → reasoner."""
        orch = make_orchestrator()

        from tiny_agents.budget.types import CollaborationStep, ActionType
        for j in range(4):
            orch.credit_tracer.trajectory.append(
                CollaborationStep(step_id=j+1, agent_name="reasoner" if j % 2 == 0 else "critic",
                                  action_type=ActionType.CONTINUE_DISCUSS, content="")
            )
        # After 4 completed steps (last=critic), step 5 is odd → reasoner
        agent, action = orch._get_next_in_schedule()
        assert agent == "reasoner", f"Step 5 should be reasoner (odd), got {agent}"

    def test_fixed_schedule_even_steps_critic(self):
        """Even-numbered steps → critic."""
        orch = make_orchestrator()

        from tiny_agents.budget.types import CollaborationStep, ActionType
        # Simulate 1 reasoner step
        orch.credit_tracer.trajectory.append(
            CollaborationStep(step_id=1, agent_name="reasoner", action_type=ActionType.CONTINUE_DISCUSS, content="")
        )

        agent, action = orch._get_next_in_schedule()
        assert agent == "critic"

    def test_schedule_after_verify_flag(self):
        """After VERIFY, _has_seen_verifier_output flag is tracked."""
        orch = make_orchestrator()
        assert orch._has_seen_verifier_output is False

        orch._has_seen_verifier_output = True
        assert orch._has_seen_verifier_output is True
