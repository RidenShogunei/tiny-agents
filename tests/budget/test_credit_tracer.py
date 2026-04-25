"""Unit tests for CreditTracer — deterministic, no model calls."""

import pytest
from tiny_agents.budget.credit_tracer import CreditTracer
from tiny_agents.budget.types import ActionType, CollaborationStep


def make_step(
    step_id: int,
    agent_name: str,
    content: str = "test",
    answer_candidate: str = None,
    token_cost: int = 100,
) -> CollaborationStep:
    """Helper: create a CollaborationStep with defaults."""
    return CollaborationStep(
        step_id=step_id,
        agent_name=agent_name,
        action_type=ActionType.CONTINUE_DISCUSS,
        content=content,
        answer_candidate=answer_candidate,
        token_cost=token_cost,
        latency_ms=10.0,
        timestamp=0.0,
    )


class TestCreditTracerMsgGain:
    """msg_gain = Q_t - Q_{t-1} is the primary signal."""

    def test_first_step_msg_gain_is_Q0(self):
        """First step: msg_gain = Q_0 - 0 = Q_0."""
        ct = CreditTracer()
        step = make_step(1, "reasoner")
        ct.add_step(step, quality_score=0.8)

        stats = ct.compute_stats()
        assert stats.msg_gain == 0.8  # Q_0 - 0

    def test_msg_gain_positive_when_Q_increases(self):
        """Higher Q_t → positive msg_gain."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.8)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.9)

        stats = ct.compute_stats()
        # msg_gain is Q_t - Q_{t-1}
        # last Q_t = 0.9, prev Q_{t-1} = 0.8 → msg_gain = 0.1
        assert stats.msg_gain == pytest.approx(0.1)
        # msg_gain_ma should reflect recent trend
        assert stats.msg_gain_ma > 0

    def test_msg_gain_zero_when_Q_flat(self):
        """No improvement → msg_gain ≈ 0."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.6)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.6)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.6)

        stats = ct.compute_stats()
        assert abs(stats.msg_gain) < 0.01  # near zero

    def test_msg_gain_negative_when_Q_drops(self):
        """Q_t decreases → negative msg_gain."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.8)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.5)

        stats = ct.compute_stats()
        assert stats.msg_gain == pytest.approx(-0.3)


class TestCreditTracerAgentCredit:
    """agent_credit = sum of msg_gains by agent. NOT step count."""

    def test_agent_credit_accumulates_msg_gains(self):
        """agent_credit = sum of its msg_gains."""
        ct = CreditTracer()

        # reasoner: Q 0→0.5→0.8 → msg_gains = 0.5, 0.3
        ct.add_step(make_step(1, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.8)
        # critic: Q 0.8→0.9 → msg_gain = 0.1
        ct.add_step(make_step(3, "critic"), quality_score=0.9)

        stats = ct.compute_stats()

        # reasoner credit = 0.5 + 0.3 = 0.8
        assert stats.agent_credit["reasoner"] == pytest.approx(0.8)
        # critic credit = 0.1
        assert stats.agent_credit["critic"] == pytest.approx(0.1)

    def test_agent_activity_ratio_is_debug_only(self):
        """agent_activity_ratio = step count ratio, NOT credit."""
        ct = CreditTracer()

        # 3 reasoner steps, 1 critic step
        # Q stays at 0.5 → msg_gains all 0 except first step = 0.5
        ct.add_step(make_step(1, "reasoner"), quality_score=0.0)  # Q: 0→0, msg_gain=0
        ct.add_step(make_step(2, "reasoner"), quality_score=0.0)  # Q: 0→0, msg_gain=0
        ct.add_step(make_step(3, "reasoner"), quality_score=0.0)  # Q: 0→0, msg_gain=0
        ct.add_step(make_step(4, "critic"), quality_score=0.0)   # Q: 0→0, msg_gain=0

        stats = ct.compute_stats()

        # Activity ratio: reasoner=3/4=0.75, critic=1/4=0.25
        assert stats.agent_activity_ratio["reasoner"] == pytest.approx(0.75)
        assert stats.agent_activity_ratio["critic"] == pytest.approx(0.25)

        # But credit is 0 because Q didn't change (all msg_gains = 0)
        assert stats.agent_credit["reasoner"] == pytest.approx(0.0)
        assert stats.agent_credit["critic"] == pytest.approx(0.0)

    def test_agent_credit_and_activity_are_different(self):
        """Credit and activity must not be confused."""
        ct = CreditTracer()

        # reasoner: 4 steps, but all with same Q (no gain)
        # First step: Q goes 0→0.5 → msg_gain=0.5 (initialization effect)
        # Subsequent steps: Q stays at 0.5 → msg_gain=0
        for i in range(1, 5):
            ct.add_step(make_step(i, "reasoner"), quality_score=0.5)

        stats = ct.compute_stats()

        # activity: reasoner=1.0 (4/4 steps)
        assert stats.agent_activity_ratio["reasoner"] == pytest.approx(1.0)

        # Credit: only the first step gets msg_gain=0.5 (Q: 0→0.5)
        # Remaining 3 steps have msg_gain=0
        # Total reasoner credit = 0.5
        assert stats.agent_credit["reasoner"] == pytest.approx(0.5)


class TestCreditTracerMarginalGain:
    """Marginal gain tracking."""

    def test_msg_gain_slope_positive_when_improving(self):
        """Positive slope when Q is increasing."""
        ct = CreditTracer()

        # Need at least k_slope=5 values for meaningful slope
        ct.add_step(make_step(1, "reasoner"), quality_score=0.1)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.3)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(4, "reasoner"), quality_score=0.7)
        ct.add_step(make_step(5, "reasoner"), quality_score=0.9)

        stats = ct.compute_stats()
        assert stats.msg_gain_slope > 0  # improving

    def test_msg_gain_slope_negative_when_declining(self):
        """Negative slope when Q is decreasing."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.9)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.7)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(4, "reasoner"), quality_score=0.3)
        ct.add_step(make_step(5, "reasoner"), quality_score=0.1)

        stats = ct.compute_stats()
        assert stats.msg_gain_slope < 0  # declining

    def test_msg_gain_slope_flat_when_stable(self):
        """Near-zero slope when Q is flat."""
        ct = CreditTracer()

        for i in range(1, 8):
            ct.add_step(make_step(i, "reasoner"), quality_score=0.6)

        stats = ct.compute_stats()
        assert abs(stats.msg_gain_slope) < 0.01


class TestCreditTracerDerived:
    """Derived signals: entropy, concentration, disagreement, uncertainty."""

    def test_credit_entropy_high_when_balanced(self):
        """High entropy when agents contribute equally."""
        ct = CreditTracer()

        # Two agents with equal positive credit
        ct.add_step(make_step(1, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(2, "critic"), quality_score=0.7)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.9)
        ct.add_step(make_step(4, "critic"), quality_score=1.0)

        stats = ct.compute_stats()
        # Both agents have non-zero credit → entropy > 0
        assert stats.credit_entropy > 0.0
        # If one agent dominates → entropy close to 0
        # With balanced contributions → higher entropy
        assert stats.credit_entropy < 1.0  # reasonable upper bound

    def test_credit_concentration_one_agent_dominates(self):
        """Concentration = 1.0 when single agent has all credit."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.8)
        ct.add_step(make_step(3, "reasoner"), quality_score=1.0)
        # critic produces no gain
        ct.add_step(make_step(4, "critic"), quality_score=1.0)

        stats = ct.compute_stats()
        # reasoner has all the gain → concentration → 1.0
        assert stats.credit_concentration > 0.9

    def test_disagreement_increases_with_different_candidates(self):
        """Disagreement tracks divergent answer candidates."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner", answer_candidate="42"), quality_score=0.5)
        ct.add_step(make_step(2, "critic", answer_candidate="42"), quality_score=0.5)
        ct.add_step(make_step(3, "reasoner", answer_candidate="42"), quality_score=0.5)

        # All same → disagreement = 0
        stats = ct.compute_stats()
        assert stats.disagreement == 0.0

        # Now introduce a different answer
        ct.add_step(make_step(4, "reasoner", answer_candidate="37"), quality_score=0.5)

        stats = ct.compute_stats()
        # Some answers differ from most recent → disagreement > 0
        assert stats.disagreement > 0.0

    def test_uncertainty_high_when_no_improvement(self):
        """High uncertainty when msg_gain_ma is near zero."""
        ct = CreditTracer()

        # Flat Q → no improvement → high uncertainty
        for i in range(1, 6):
            ct.add_step(make_step(i, "reasoner"), quality_score=0.5)

        stats = ct.compute_stats()
        # uncertainty = 1 / (1 + |ma| * 10), flat Q → ma ≈ 0 → uncertainty ≈ 1.0
        assert stats.uncertainty > 0.9

    def test_uncertainty_low_when_gaining(self):
        """Low uncertainty when showing steady improvement."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.1)
        ct.add_step(make_step(2, "reasoner"), quality_score=0.3)
        ct.add_step(make_step(3, "reasoner"), quality_score=0.5)
        ct.add_step(make_step(4, "reasoner"), quality_score=0.7)

        stats = ct.compute_stats()
        # Steady gains → low uncertainty
        assert stats.uncertainty < 0.5


class TestCreditTracerReset:
    """Reset between problems."""

    def test_reset_clears_all_state(self):
        """reset() clears trajectory and computed stats."""
        ct = CreditTracer()

        ct.add_step(make_step(1, "reasoner"), quality_score=0.8)
        ct.add_step(make_step(2, "critic"), quality_score=0.9)

        assert len(ct.trajectory) == 2

        ct.reset()

        assert len(ct.trajectory) == 0
        assert len(ct._msg_gains) == 0
        assert len(ct._quality_scores) == 0
        assert len(ct._msg_gains_per_agent) == 0

        stats = ct.compute_stats()
        assert stats.msg_gain == 0.0
        assert stats.agent_credit == {}
