"""VerifierAgent — dual-use verifier with three-valued output.

Key design (per user correction):
  - Final correctness: use deterministic checker (exact match / numeric match)
  - Control signal Qt: LLM verifier with three-valued output (CORRECT / INCORRECT / UNCERTAIN)

The LLM verifier produces:
  - verdict: CORRECT / INCORRECT / UNCERTAIN
  - confidence: [0.0, 1.0] continuous estimate
  - quality_score: Q_t used as msg_gain signal
  - feedback: natural language explanation

UNCERTAIN is important: it tells the controller that spending more budget
might be worthwhile, rather than definitively stopping.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from tiny_agents.budget.types import VerificationResult, VerdictType


class VerifierAgent:
    """
    Phase 1 verifier with dual purpose.

    1. Control signal Qt: LLM-based, three-valued output
    2. Final correctness: deterministic (exact match / numeric match)

    The LLM verifier does NOT serve as ground truth evaluator.
    """

    def __init__(self, backend: Any, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.backend = backend
        self.model_name = model_name

    async def verify(
        self,
        problem: str,
        candidate_answer: str,
        trajectory: List[Any],
    ) -> VerificationResult:
        """
        Produce a quality assessment for the current candidate answer.

        Returns VerificationResult with three-valued verdict and confidence.
        Used as the Q_t signal for CreditTracer.

        Args:
            problem: the original problem text
            candidate_answer: current answer to evaluate
            trajectory: full collaboration trajectory (for context)
        """
        start = time.time()

        # Build context from trajectory
        trajectory_text = self._build_trajectory_context(trajectory)

        prompt = self._build_verifier_prompt(problem, candidate_answer, trajectory_text)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse three-valued verdict
        verdict, confidence, quality_score, feedback = self._parse_response(response)

        latency = (time.time() - start) * 1000
        token_cost = self._estimate_token_cost(prompt, response)

        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            quality_score=quality_score,
            feedback=feedback,
            token_cost=token_cost,
            latency_ms=latency,
        )

    # ── Deterministic final checkers ─────────────────────────────────────────

    @staticmethod
    def check_final_correctness(
        predicted: str,
        ground_truth: str,
        task_type: str = "math",
    ) -> bool:
        """
        Deterministic final correctness check.
        Used as the actual ground truth evaluation, NOT the LLM verifier.

        Supported task types:
          - "math": numeric match or symbolic equivalence
          - "gsm8k": parseable number extraction
          - "exact": exact string match
        """
        if task_type == "exact":
            return predicted.strip() == ground_truth.strip()

        if task_type == "gsm8k" or task_type == "math":
            pred_nums = _extract_numbers(predicted)
            gt_nums = _extract_numbers(ground_truth)
            if pred_nums and gt_nums:
                # For GSM8K/MATH: compare the final numeric answer
                return pred_nums[-1] == gt_nums[-1]
            return predicted.strip() == ground_truth.strip()

        return predicted.strip() == ground_truth.strip()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_trajectory_context(self, trajectory: List[Any]) -> str:
        """Build natural language context from trajectory for the verifier."""
        if not trajectory:
            return "No prior collaboration steps."
        lines = []
        for step in trajectory[-5:]:  # last 5 steps for brevity
            agent = step.agent_name
            content = step.content[:200]  # truncate long messages
            lines.append(f"[{agent}]: {content}")
        return "\n".join(lines)

    def _build_verifier_prompt(
        self,
        problem: str,
        candidate_answer: str,
        trajectory_context: str,
    ) -> str:
        return f"""You are evaluating whether a candidate answer is correct for the given problem.

Problem: {problem}

Recent collaboration history:
{trajectory_context}

Candidate answer to evaluate: {candidate_answer}

Evaluate this answer carefully. Consider:
- Is the reasoning sound?
- Is the final answer correct?
- Are there any errors or oversights?

Respond with ONLY a JSON object in this exact format:
{{"verdict": "CORRECT", "confidence": 0.95, "feedback": "brief explanation"}}
OR
{{"verdict": "INCORRECT", "confidence": 0.90, "feedback": "brief explanation"}}
OR
{{"verdict": "UNCERTAIN", "confidence": 0.50, "feedback": "brief explanation"}}

Choose UNCERTAIN when the answer could be right but you're not confident enough to say CORRECT, and it could go either way. This is the most useful state for deciding whether to continue reasoning.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend. Returns raw response text."""
        # Use the vLLM backend's chat interface
        messages = [{"role": "user", "content": prompt}]
        response = self.backend.chat(messages)
        if hasattr(response, "choices"):
            return response.choices[0].message.content
        return str(response)

    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into verdict, confidence, quality_score, feedback."""
        try:
            # Try to extract JSON
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                import json
                obj = json.loads(json_match.group())
            else:
                obj = {}

            verdict_str = obj.get("verdict", "UNCERTAIN").upper()
            if verdict_str not in ("CORRECT", "INCORRECT", "UNCERTAIN"):
                verdict_str = "UNCERTAIN"

            verdict = VerdictType[verdict_str]
            confidence = float(obj.get("confidence", 0.5))
            feedback = str(obj.get("feedback", ""))

            # quality_score Q_t: map verdict to [0, 1] scale
            # CONFIDENT CORRECT → high score, UNCERTAIN → medium, CONFIDENT INCORRECT → low
            quality_score = self._verdict_to_quality(verdict, confidence)

            return verdict, confidence, quality_score, feedback

        except Exception:
            # Fallback: treat as uncertain
            return VerdictType.UNCERTAIN, 0.5, 0.5, "Failed to parse verifier response"

    def _verdict_to_quality(self, verdict: VerdictType, confidence: float) -> float:
        """Map verdict + confidence to a [0, 1] quality score Q_t."""
        if verdict == VerdictType.CORRECT:
            # Map [0.5, 1.0] confidence to [0.5, 1.0] quality
            return 0.5 + 0.5 * confidence
        elif verdict == VerdictType.INCORRECT:
            # Map [0.5, 1.0] confidence to [0.0, 0.5] quality
            return 0.5 - 0.5 * confidence
        else:  # UNCERTAIN
            # UNCERTAIN with any confidence → medium score (encourages continued reasoning)
            return 0.5

    def _estimate_token_cost(self, prompt: str, response: str) -> int:
        """Rough token estimation (prompt + response) / 4 chars per token."""
        return (len(prompt) + len(response)) // 4


# ── Number extraction helper ─────────────────────────────────────────────────

def _extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text, useful for math answer comparison."""
    # Match integers and decimals
    pattern = r'-?\d+(?:\.\d+)?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers
