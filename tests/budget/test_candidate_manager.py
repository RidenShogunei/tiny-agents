"""Unit tests for CandidateManager — deterministic answer tracking."""

import pytest
from tiny_agents.budget.candidate_manager import CandidateAnswer, CandidateManager


class TestCandidateManagerBestAnswer:
    """Best answer tracking with score."""

    def test_first_candidate_becomes_best(self):
        """First candidate is automatically the best."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")

        assert mgr.get_best_text() == "42"
        assert mgr.get_best_score() == 0.5

    def test_higher_score_replaces_best(self):
        """Higher score → replaces current best."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="37", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="42", score=0.8, step_id=2, agent_name="reasoner")

        assert mgr.get_best_text() == "42"
        assert mgr.get_best_score() == 0.8

    def test_lower_score_does_not_replace_best(self):
        """Lower score does not replace best."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.8, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="37", score=0.5, step_id=2, agent_name="reasoner")

        assert mgr.get_best_text() == "42"
        assert mgr.get_best_score() == 0.8

    def test_equal_score_keeps_first(self):
        """Equal score: tie-break = first one wins (stable)."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="42", score=0.5, step_id=2, agent_name="critic")

        # First one kept
        assert mgr.get_best_text() == "42"
        assert mgr.get_best_step_id() == 1

    def test_add_candidate_returns_is_new_best(self):
        """add_candidate returns whether this is the new best."""
        mgr = CandidateManager()
        mgr.reset()

        _, is_best = mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        assert is_best is True

        _, is_best = mgr.add_candidate(text="37", score=0.3, step_id=2, agent_name="critic")
        assert is_best is False

        _, is_best = mgr.add_candidate(text="99", score=0.9, step_id=3, agent_name="reasoner")
        assert is_best is True


class TestCandidateManagerDisagreement:
    """Answer disagreement tracking."""

    def test_no_disagreement_single_candidate(self):
        """Single candidate → disagreement = 0."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")

        assert mgr.get_disagreement_score() == 0.0

    def test_no_disagreement_all_same(self):
        """All candidates same → disagreement = 0."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="42", score=0.6, step_id=2, agent_name="critic")
        mgr.add_candidate(text="42", score=0.8, step_id=3, agent_name="reasoner")

        assert mgr.get_disagreement_score() == 0.0

    def test_disagreement_with_different_answers(self):
        """Different answers → disagreement > 0."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="37", score=0.6, step_id=2, agent_name="critic")
        mgr.add_candidate(text="42", score=0.8, step_id=3, agent_name="reasoner")

        # 1 out of 3 candidates differs from best "42"
        assert mgr.get_disagreement_score() == pytest.approx(1/3)

    def test_disagreement_half_different(self):
        """Half of candidates differ → disagreement = 0.5."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="37", score=0.6, step_id=2, agent_name="critic")

        assert mgr.get_disagreement_score() == 0.5


class TestCandidateManagerByAgent:
    """Scores tracked per agent."""

    def test_scores_by_agent(self):
        """Can retrieve all scores for a specific agent."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="37", score=0.6, step_id=2, agent_name="critic")
        mgr.add_candidate(text="99", score=0.7, step_id=3, agent_name="reasoner")

        reasoner_scores = mgr.get_score_by_agent("reasoner")
        critic_scores = mgr.get_score_by_agent("critic")

        assert reasoner_scores == [0.5, 0.7]
        assert critic_scores == [0.6]

    def test_unknown_agent_returns_empty(self):
        """Unknown agent → empty list."""
        mgr = CandidateManager()
        mgr.reset()

        assert mgr.get_score_by_agent("unknown") == []


class TestCandidateManagerReset:
    """Reset between problems."""

    def test_reset_clears_all(self):
        """reset() clears candidates and best."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.8, step_id=1, agent_name="reasoner")
        assert mgr.get_best_text() == "42"

        mgr.reset()

        assert mgr.get_best_text() is None
        assert len(mgr.candidates) == 0


class TestCandidateManagerSummary:
    """Answer history summary for logging."""

    def test_summary_contains_expected_fields(self):
        """Summary dict has all expected fields."""
        mgr = CandidateManager()
        mgr.reset()

        mgr.add_candidate(text="42", score=0.5, step_id=1, agent_name="reasoner")
        mgr.add_candidate(text="37", score=0.6, step_id=2, agent_name="critic")

        summary = mgr.get_answer_history_summary()

        assert summary["num_candidates"] == 2
        assert summary["best_score"] == 0.6
        assert summary["best_step_id"] == 2
        assert summary["best_agent"] == "critic"
        assert summary["disagreement"] == 0.5
        assert "reasoner" in summary["agents_answered"]
        assert "critic" in summary["agents_answered"]
