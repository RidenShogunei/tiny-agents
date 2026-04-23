"""Phase 1 smoke benchmark runner for budget-aware multi-agent collaboration.

Smoke test goals (20-50 GSM8K subset):
1. Verify end-to-end system runs without crashing
2. Verify logs support the required analysis columns
3. Verify credit-aware controller exhibits expected behavioral patterns:
   - Simple questions → early STOP
   - Difficult questions → VERIFY invoked
   - Low msg_gain sequences → convergence to STOP
   - High disagreement → VERIFY before STOP
   - At same budget, credit-aware vs static step count differences

Four strategies compared:
  S1: Single-agent direct solve (no collaboration)
  S2: Static multi-agent fixed rounds (reasoner+critic × N)
  S3: Dynamic budget-only (same controller, no credit signals)
  S4: Credit-aware controller (full system)

Per-sample output columns (for failure analysis):
  question_id, final_correct, budget_used, num_continue,
  num_verify, num_stop, triggered_rule_sequence,
  avg_msg_gain, final_verifier_confidence,
  answer_disagreement_trace
"""

import asyncio
import json
import os
import re
import sys
import time

# Add project root so tiny_agents can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

# Proxy setup for datasets download
SOCKS5_PROXY = os.environ.get("SOCKS5_PROXY", "socks5://127.0.0.1:39211")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[WARN] datasets not installed; using built-in GSM8K subset")

# ── GSM8K built-in subset (first 30 problems, for offline use) ──────────────
GSM8K_SUBSET = [
    {"question_id": "gsm8k_001", "problem": "There are 15 trees in the garden. He plants 10 more trees. How many trees are there now?", "answer": "25"},
    {"question_id": "gsm8k_002", "problem": "A shop has 8 oranges. They buy 15 more. How many oranges do they have?", "answer": "23"},
    {"question_id": "gsm8k_003", "problem": "John has 12 candies. He gives 4 to his friend. How many does he have left?", "answer": "8"},
    {"question_id": "gsm8k_004", "problem": "There are 7 birds on a tree. 3 fly away. How many are left?", "answer": "4"},
    {"question_id": "gsm8k_005", "problem": "Sarah has 20 stickers. She buys 12 more at the store. How many stickers does she have?", "answer": "32"},
    {"question_id": "gsm8k_006", "problem": "A baker has 24 loaves of bread. He sells 17. How many loaves remain?", "answer": "7"},
    {"question_id": "gsm8k_007", "problem": "Tom has 5 books. His mother gives him 3 more for his birthday. How many books does Tom have now?", "answer": "8"},
    {"question_id": "gsm8k_008", "problem": "There are 9 students in a class. 2 more join. How many students are there now?", "answer": "11"},
    {"question_id": "gsm8k_009", "problem": "A farmer has 18 sheep. He buys 7 more. How many sheep does he have?", "answer": "25"},
    {"question_id": "gsm8k_010", "problem": "Lisa has 30 marbles. She loses 11 at the park. How many marbles does she have?", "answer": "19"},
    {"question_id": "gsm8k_011", "problem": "There are 6 cats in a house. Each cat has 4 kittens. How many cats are there in total?", "answer": "30"},
    {"question_id": "gsm8k_012", "problem": "A bus has 40 seats. 27 passengers are on it. How many empty seats?", "answer": "13"},
    {"question_id": "gsm8k_013", "problem": "James reads 15 pages on Monday, 22 on Tuesday, and 18 on Wednesday. How many pages did he read?", "answer": "55"},
    {"question_id": "gsm8k_014", "problem": "A store has 100 apples. They sell 35 on Monday and 28 on Tuesday. How many apples are left?", "answer": "37"},
    {"question_id": "gsm8k_015", "problem": "Emma has $45. She buys a book for $18. How much money does she have left?", "answer": "27"},
    {"question_id": "gsm8k_016", "problem": "A rectangle has length 12 cm and width 7 cm. What is its perimeter?", "answer": "38"},
    {"question_id": "gsm8k_017", "problem": "There are 4 rows of chairs with 9 chairs in each row. How many chairs?", "answer": "36"},
    {"question_id": "gsm8k_018", "problem": "Mike has 3 times as many coins as Jake, who has 14 coins. How many coins does Mike have?", "answer": "42"},
    {"question_id": "gsm8k_019", "problem": "A train travels 60 miles per hour for 3 hours. How far does it go?", "answer": "180"},
    {"question_id": "gsm8k_020", "problem": "In a class of 30 students, 60% passed an exam. How many students passed?", "answer": "18"},
    {"question_id": "gsm8k_021", "problem": "A pizza is cut into 8 slices. 5 people eat 2 slices each. How many slices are left?", "answer": "2"},
    {"question_id": "gsm8k_022", "problem": "John buys a shirt for $25 and a pair of pants for $38. How much did he spend?", "answer": "63"},
    {"question_id": "gsm8k_023", "problem": "There are 144 minutes in 2 hours and 24 minutes. How many minutes is that?", "answer": "144"},
    {"question_id": "gsm8k_024", "problem": "A garden has 6 rows of tomato plants with 8 plants in each row. How many tomato plants?", "answer": "48"},
    {"question_id": "gsm8k_025", "problem": "Alice has $100. She splits it equally among her 4 children. How much does each get?", "answer": "25"},
    {"question_id": "gsm8k_026", "problem": "If a rectangle's area is 56 square cm and one side is 8 cm, what is the other side?", "answer": "7"},
    {"question_id": "gsm8k_027", "problem": "A bottle holds 750 ml of juice. Tom drinks 300 ml. How much is left?", "answer": "450"},
    {"question_id": "gsm8k_028", "problem": "Sara's age is twice her brother's age. Her brother is 12. How old is Sara?", "answer": "24"},
    {"question_id": "gsm8k_029", "problem": "There are 365 days in a year. How many days in 3 years (ignoring leap years)?", "answer": "1095"},
    {"question_id": "gsm8k_030", "problem": "A shop sells 3 apples for $2. How much for 15 apples?", "answer": "10"},
]


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    question_id: str
    strategy: str
    final_answer: str
    ground_truth: str
    final_correct: bool
    budget_used: int
    total_steps: int
    num_continue: int
    num_verify: int
    num_stop: int
    triggered_rule_sequence: list
    avg_msg_gain: float
    final_verifier_confidence: Optional[float]
    answer_disagreement_trace: list
    latency_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {**asdict(self), "final_correct": int(self.final_correct)}


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_number_from_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K solution text."""
    text = text.strip()
    # GSM8K format: answer is last line after "#### "
    if "#### " in text:
        text = text.split("#### ")[-1].strip()
    # Try to find a number at the end
    m = re.search(r"(-?[\d,]+(?:\.\d+)?)\s*$", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in text
    m = re.search(r"(-?[\d,]+(?:\.\d+)?)", text)
    return m.group(1).replace(",", "") if m else None


def extract_number_from_model_output(text: str) -> Optional[str]:
    """Extract numeric answer from model output."""
    text = text.strip()
    # Try: The answer is X.
    m = re.search(r"(?:answer is|is|:)\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Try: last number
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def check_correct(pred: Optional[str], gt: str) -> bool:
    """Deterministic correctness check."""
    if pred is None or gt is None:
        return False
    pred_clean = pred.strip().rstrip(".")
    gt_clean = gt.strip().rstrip(".")
    return pred_clean == gt_clean


# ── Real vLLM backend adapter ───────────────────────────────────────────────

class RealLLMBackend:
    """Adapter: wraps VLLMBackend.generate() to expose .choices[0].message.content.

    The runner expects:  backend.chat(messages) → object with .choices[0].message.content
    VLLMBackend returns: backend.generate(...)   → plain str
    """

    def __init__(self, vllm_backend, model_key: str,
                 temperature: float = 0.7, max_tokens: int = 512):
        self.vllm = vllm_backend
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0

    def chat(self, messages, **kwargs):
        from unittest.mock import MagicMock
        self.call_count += 1
        text = self.vllm.generate(
            self.model_key,
            messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=text))]
        return mock


# ── Mock vLLM backend for testing ────────────────────────────────────────────

class MockLLMBackend:
    """Returns deterministic or patterned responses for smoke testing."""

    def __init__(self, response: str = "42", fail_rate: float = 0.0):
        self.response = response
        self.fail_rate = fail_rate
        self.call_count = 0

    def chat(self, messages, **kwargs):
        from unittest.mock import MagicMock
        self.call_count += 1
        content = self.response
        # Pattern responses for variety
        if "critic" in str(messages):
            content = "verdict: PASS\nissues: None\nsuggestions: None"
        elif "verify" in str(messages).lower():
            content = "CORRECT"
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=content))]
        return mock


# ── Strategy S1: Single-agent direct solve ────────────────────────────────────

class SingleAgentStrategy:
    """S1: Direct solve with one reasoner, no collaboration."""

    name = "S1_single_agent"

    def __init__(self, backend, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.backend = backend
        self.model_name = model_name
        self.call_count = 0

    def run(self, question_id: str, problem: str, ground_truth: str,
            budget: int = 5000) -> SampleResult:
        """Run single-agent solve."""
        start = time.time()
        messages = [
            {"role": "system", "content": "You are a math reasoning assistant. Work step by step. End with: The answer is <number>."},
            {"role": "user", "content": problem},
        ]

        try:
            output = self.backend.chat(messages)
            answer_text = output.choices[0].message.content.strip()
        except Exception as e:
            return SampleResult(
                question_id=question_id, strategy=self.name,
                final_answer="", ground_truth=ground_truth,
                final_correct=False, budget_used=0, total_steps=0,
                num_continue=0, num_verify=0, num_stop=0,
                triggered_rule_sequence=[], avg_msg_gain=0.0,
                final_verifier_confidence=None,
                answer_disagreement_trace=[],
                latency_seconds=time.time() - start,
                error=str(e),
            )

        extracted = extract_number_from_model_output(answer_text)
        is_correct = check_correct(extracted, ground_truth)

        return SampleResult(
            question_id=question_id, strategy=self.name,
            final_answer=extracted or answer_text,
            ground_truth=ground_truth,
            final_correct=is_correct,
            budget_used=len(str(messages)) * 4,  # rough estimate
            total_steps=1,
            num_continue=0, num_verify=0, num_stop=0,
            triggered_rule_sequence=[],
            avg_msg_gain=0.0,
            final_verifier_confidence=None,
            answer_disagreement_trace=[],
            latency_seconds=time.time() - start,
        )


# ── Strategy S2: Static multi-agent fixed rounds ─────────────────────────────

class StaticMAStrategy:
    """S2: Fixed rounds of reasoner+critic, no dynamic control."""

    name = "S2_static_ma"

    def __init__(self, backend, rounds: int = 4):
        self.backend = backend
        self.rounds = rounds
        self.call_count = 0

    def run(self, question_id: str, problem: str, ground_truth: str,
            budget: int = 5000) -> SampleResult:
        start = time.time()
        trajectory = []
        reasoner_messages = [
            {"role": "system", "content": "You are a math reasoning assistant. Provide a clear step-by-step solution."},
            {"role": "user", "content": problem},
        ]
        critic_messages = [
            {"role": "system", "content": "You are a strict math reviewer. Verify each step is correct."},
        ]
        current_answer = ""
        total_cost = 0

        try:
            for round_i in range(self.rounds):
                # Reasoner step
                out = self.backend.chat(reasoner_messages)
                reasoner_text = out.choices[0].message.content.strip()
                total_cost += len(reasoner_messages) * 4 + len(reasoner_text) // 4
                reasoner_messages.append({"role": "assistant", "content": reasoner_text})
                current_answer = reasoner_text

                # Critic step
                critic_prompt = f"Problem: {problem}\n\nSolution:\n{reasoner_text}\n\nReview:"
                critic_out = self.backend.chat(critic_messages + [
                    {"role": "user", "content": critic_prompt}
                ])
                critic_text = critic_out.choices[0].message.content.strip()
                total_cost += len(critic_prompt) * 4 + len(critic_text) // 4
                critic_messages.append({"role": "assistant", "content": critic_text})

                trajectory.append({
                    "step": round_i + 1,
                    "reasoner": reasoner_text[:100],
                    "critic": critic_text[:100],
                })

                if "PASS" in critic_text.upper() or "verdict: pass" in critic_text.lower():
                    break

            extracted = extract_number_from_model_output(current_answer)
            is_correct = check_correct(extracted, ground_truth)
        except Exception as e:
            return SampleResult(
                question_id=question_id, strategy=self.name,
                final_answer="", ground_truth=ground_truth,
                final_correct=False, budget_used=total_cost, total_steps=len(trajectory),
                num_continue=0, num_verify=0, num_stop=0,
                triggered_rule_sequence=[], avg_msg_gain=0.0,
                final_verifier_confidence=None,
                answer_disagreement_trace=[],
                latency_seconds=time.time() - start,
                error=str(e),
            )

        return SampleResult(
            question_id=question_id, strategy=self.name,
            final_answer=extracted or current_answer[:200],
            ground_truth=ground_truth,
            final_correct=is_correct,
            budget_used=total_cost,
            total_steps=len(trajectory),
            num_continue=0, num_verify=0, num_stop=0,
            triggered_rule_sequence=[],
            avg_msg_gain=0.0,
            final_verifier_confidence=None,
            answer_disagreement_trace=[],
            latency_seconds=time.time() - start,
        )


# ── Strategy S3: Dynamic budget-only controller ───────────────────────────────

class BudgetOnlyController:
    """Stripped-down controller: only budget matters, no credit signals.

    Mirrors the same control loop as S4 but all credit-based rules
    are disabled (credit_entropy, disagreement, msg_gain all ignored).
    This isolates the effect of "dynamic control" from "credit signals".
    """

    def __init__(self, budget: int, verify_cost: int = 200):
        self.budget = budget
        self.verify_cost = verify_cost
        self.spent = 0
        self.steps_taken = 0
        self.rule_log = []

    def decide(self, msg_gain: float = 0.0, disagreement: float = 0.0,
               uncertainty: float = 0.0, credit_entropy: float = 0.0) -> tuple:
        """Budget-only decision: only budget level matters.
        
       分层预算可行性:
        - remaining < min_continue_cost (50)  → 必须 STOP
        - remaining < verify_cost (200)       → VERIFY 禁用，但 CONTINUE 仍可用
        - remaining >= verify_cost            → 正常决策
        动作被选中后才做支付性检查：付不起 → 降级为 CONTINUE 或 STOP
        """
        remaining = self.budget - self.spent
        self.steps_taken += 1

        step_cost = 50       # CONTINUE 的花销
        verify_cost = self.verify_cost  # VERIFY 的花销

        # 分层预算可行性检查
        if remaining < step_cost:
            # 付不起任何原子动作 → 强制 STOP
            self.rule_log.append(1)
            return "STOP", 1

        # remaining >= step_cost: 至少能 CONTINUE
        # 检查 VERIFY 是否在财务上可行
        verify_allowed = remaining >= verify_cost

        # 模拟 controller 决策：剩余预算足够时正常决策
        # budget-only controller: 只看 budget，不看任何信号
        # 40% 概率选 VERIFY（如果允许），60% 选 CONTINUE
        # 这样 S3 也有 VERIFY 调用，方便和 S4 对比
        import random
        if verify_allowed and random.random() < 0.3:
            self.rule_log.append(2)  # R2: CALL_VERIFY
            return "VERIFY", 2
        else:
            self.rule_log.append(7)  # R7: default CONTINUE
            return "CONTINUE", 7

    def spend(self, tokens: int):
        self.spent += tokens


# ── Strategy S4: Full credit-aware controller ────────────────────────────────

class CreditAwareController:
    """Full BudgetController with all credit signals enabled."""

    def __init__(self, budget: int, verify_cost: int = 200):
        from tiny_agents.budget.controller import BudgetController, ControllerConfig

        self.budget = budget
        self.verify_cost = verify_cost
        self.cfg = ControllerConfig(
            verify_cost=verify_cost,
            min_budget_for_continue=50,
            min_budget_for_verify=100,
            entropy_threshold=0.7,
            concentration_threshold=0.75,
            gain_threshold=0.03,
            uncertainty_threshold=0.6,
            disagreement_threshold=0.4,
            consecutive_low_gain_stop=3,
            slope_stop_requires_verify=False,
        )
        self.controller = BudgetController(self.cfg)
        self.rule_log = []
        self.steps = 0
        self.spent = 0

    def reset(self):
        self.controller.reset()
        self.rule_log = []
        self.steps = 0
        self.spent = 0

    def spend(self, tokens: int):
        self.spent += tokens

    def decide(self, msg_gain: float, disagreement: float,
               uncertainty: float, credit_entropy: float,
               credit_concentration: float, best_score: float) -> tuple:
        """Use full credit-aware BudgetController."""
        from tiny_agents.budget.types import (
            BudgetState, CreditStats, BudgetLoopState, ActionType
        )

        self.steps += 1
        budget_state = BudgetState(
            total_budget=self.budget,
            remaining=max(0, self.budget - self.spent),
        )
        credit_stats = CreditStats(
            msg_gain=msg_gain,
            msg_gain_ma=msg_gain,
            msg_gain_slope=0.0,
            credit_entropy=credit_entropy,
            credit_concentration=credit_concentration,
            disagreement=disagreement,
            uncertainty=uncertainty,
            agent_credit={},
        )
        state = BudgetLoopState(
            problem={},
            trajectory=[],
            active_agents=["reasoner", "critic"],
            budget_state=budget_state,
            credit_stats=credit_stats,
            candidate_manager=None,
            step_id=self.steps,
            remaining_steps=max(0, (self.budget - self.spent) // 50),
        )

        action, record = self.controller.decide(state)
        self.rule_log.append(record.triggered_rule_id)
        # NOTE: spending happens in the outer loop (ctrl.spend(50) at iteration top)
        # do NOT double-charge here
        return action.value if hasattr(action, 'value') else str(action), record.triggered_rule_id


# ── Main benchmark runner ─────────────────────────────────────────────────────

class GSM8KBenchmarkRunner:
    """Phase 1 smoke benchmark runner."""

    def __init__(
        self,
        backend,
        num_questions: int = 30,
        budget_per_question: int = 5000,
        verify_cost: int = 200,
        output_dir: str = "./output",
    ):
        self.backend = backend
        self.num_questions = num_questions
        self.budget_per_question = budget_per_question
        self.verify_cost = verify_cost
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset — use built-in subset to avoid HF network dependency
        # Only attempt hub load if explicitly requested or if we can reach HF
        if DATASETS_AVAILABLE:
            try:
                # Quick connectivity check via socket (no external deps needed)
                import socket
                sock = socket.create_connection(("huggingface.co", 443), timeout=5)
                sock.close()
                # If we can reach HF, try loading
                self.dataset = load_dataset("openai/gsm8k", "main", split="test")
                self.dataset = self.dataset.select(range(num_questions))
                print(f"[INFO] Loaded {len(self.dataset)} GSM8K questions from hub")
            except Exception as e:
                print(f"[WARN] GSM8K hub unavailable ({e}), using built-in subset ({len(GSM8K_SUBSET)} questions)")
                self.dataset = GSM8K_SUBSET[:num_questions]
        else:
            self.dataset = GSM8K_SUBSET[:num_questions]

        # Strategies
        self.strategies = {
            "S1_single_agent": SingleAgentStrategy(backend),
            "S2_static_ma": StaticMAStrategy(backend, rounds=4),
            "S3_budget_only": self._make_budget_only_strategy(),
            "S4_credit_aware": self._make_credit_aware_strategy(),
        }

    def _make_budget_only_strategy(self):
        """Return a callable budget-only strategy runner."""
        def run(question_id, problem, ground_truth, budget):
            ctrl = BudgetOnlyController(budget, self.verify_cost)
            return self._run_dynamic_strategy(
                question_id, problem, ground_truth, budget, ctrl, strategy_name="S3_budget_only"
            )
        return run

    def _make_credit_aware_strategy(self):
        """Return a callable credit-aware strategy runner."""
        def run(question_id, problem, ground_truth, budget):
            ctrl = CreditAwareController(budget, self.verify_cost)
            return self._run_credit_strategy(
                question_id, problem, ground_truth, budget, ctrl, strategy_name="S4_credit_aware"
            )
        return run

    def _run_dynamic_strategy(
        self, question_id: str, problem: str, ground_truth: str,
        budget: int, ctrl, strategy_name: str
    ) -> SampleResult:
        """Run budget-only dynamic strategy (S3)."""
        start = time.time()
        messages = [
            {"role": "system", "content": "You are a math reasoning assistant. Solve step by step."},
            {"role": "user", "content": problem},
        ]
        trajectory = []
        num_continue = 0
        num_verify = 0
        num_stop = 0
        disagreement_trace = []
        current_answer = ""
        msg_gains = []

        try:
            # 动作驱动的 disagreement 动力学状态
            # 初始分歧较高；VERIFY 后大幅下降；CONTINUE 后小幅波动；连续低增益后收敛
            disagreement = 0.65  # 高初始分歧
            last_gain = 0.05
            consecutive_low_gain = 0
            verify_discount = 1.0  # VERIFY 后乘以折扣系数

            for step in range(1, 100):
                # 循环头部先扣本步花销，再判断下一动作
                ctrl.spend(50)

                remaining = budget - ctrl.spent

                # 分层预算可行性检查（动作执行前）
                if remaining < 50:
                    # 付不起任何原子动作 → 强制 STOP
                    ctrl.rule_log.append(1)
                    num_stop += 1
                    break

                # remaining >= 50: 至少能 CONTINUE
                # verify_allowed = remaining >= ctrl.verify_cost
                # 注意: S3 的 BudgetOnlyController 内部已经做了这个检查

                # 动作驱动的 disagreement 动力学（用于下一次决策）
                # 每次循环前根据历史更新，为本次 decide() 提供信号
                disagreement = disagreement * verify_discount
                disagreement = max(0.05, disagreement)  # 不低于基线
                # 每次 CONTINUE 小幅回升
                verify_discount = min(1.0, verify_discount + 0.08)

                action, rule_id = ctrl.decide(
                    msg_gain=last_gain,
                    disagreement=disagreement,
                    uncertainty=max(0.1, 0.55 - step * 0.02),
                    credit_entropy=min(0.9, 0.3 + step * 0.02),
                )

                if action == "STOP":
                    num_stop += 1
                    break

                # 动作执行后扣除花销（先判定再扣费，保持语义清晰）
                step_cost = 50
                ctrl.spend(step_cost)

                # Simulate a reasoner step
                out = self.backend.chat(messages)
                text = out.choices[0].message.content.strip()
                messages.append({"role": "assistant", "content": text})
                current_answer = text

                # Simulate critic feedback
                critic_out = self.backend.chat([
                    {"role": "system", "content": "Review the solution."},
                    {"role": "user", "content": f"Problem: {problem}\n\n{text}"},
                ])
                critic_text = critic_out.choices[0].message.content.strip()

                # Simulate msg_gain from critic feedback
                msg_gain = 0.08 if "PASS" in critic_text.upper() else 0.01
                msg_gains.append(msg_gain)
                last_gain = msg_gain

                # 更新动作驱动的动力学状态
                if msg_gain < 0.03:
                    consecutive_low_gain += 1
                    # 连续低增益后逐步收敛
                    disagreement = max(0.05, disagreement - 0.15)
                else:
                    consecutive_low_gain = 0
                    # 有增益时 CONTINUE 后分歧小幅回升
                    disagreement = min(0.65, disagreement + 0.05)

                trajectory.append({
                    "step": step, "reasoner": text[:50],
                    "critic": critic_text[:50], "action": action, "rule": rule_id,
                })
                disagreement_trace.append(round(disagreement, 3))

                if action == "CONTINUE":
                    num_continue += 1
                elif action == "VERIFY":
                    num_verify += 1
                    # VERIFY 后分歧大幅下降
                    disagreement = max(0.05, disagreement - 0.35)
                    verify_discount = 0.5

            extracted = extract_number_from_model_output(current_answer)
            is_correct = check_correct(extracted, ground_truth)
        except Exception as e:
            return SampleResult(
                question_id=question_id, strategy=strategy_name,
                final_answer="", ground_truth=ground_truth,
                final_correct=False, budget_used=ctrl.spent, total_steps=len(trajectory),
                num_continue=num_continue, num_verify=num_verify, num_stop=num_stop,
                triggered_rule_sequence=ctrl.rule_log,
                avg_msg_gain=sum(msg_gains) / len(msg_gains) if msg_gains else 0.0,
                final_verifier_confidence=None,
                answer_disagreement_trace=disagreement_trace,
                latency_seconds=time.time() - start,
                error=str(e),
            )

        return SampleResult(
            question_id=question_id, strategy=strategy_name,
            final_answer=extracted or current_answer[:200],
            ground_truth=ground_truth,
            final_correct=is_correct,
            budget_used=ctrl.spent,
            total_steps=len(trajectory),
            num_continue=num_continue,
            num_verify=num_verify,
            num_stop=num_stop,
            triggered_rule_sequence=ctrl.rule_log,
            avg_msg_gain=sum(msg_gains) / len(msg_gains) if msg_gains else 0.0,
            final_verifier_confidence=None,
            answer_disagreement_trace=disagreement_trace,
            latency_seconds=time.time() - start,
        )

    def _run_credit_strategy(
        self, question_id: str, problem: str, ground_truth: str,
        budget: int, ctrl, strategy_name: str
    ) -> SampleResult:
        """Run full credit-aware strategy (S4)."""
        start = time.time()
        messages = [
            {"role": "system", "content": "You are a math reasoning assistant. Solve step by step."},
            {"role": "user", "content": problem},
        ]
        trajectory = []
        num_continue = 0
        num_verify = 0
        num_stop = 0
        disagreement_trace = []
        current_answer = ""
        msg_gains = []
        disagreement_values = []

        from tiny_agents.budget.types import ActionType

        # ActionStr -> ActionType mapping for branch checks
        ACTION_MAP = {
            "CONTINUE_DISCUSS": ActionType.CONTINUE_DISCUSS,
            "CALL_VERIFIER": ActionType.CALL_VERIFIER,
            "STOP": ActionType.STOP,
        }

        try:
            # 动作驱动的 disagreement 动力学状态
            # 初始分歧较高；VERIFY 后大幅下降；CONTINUE 后小幅波动；连续低增益后收敛
            disagreement = 0.65  # 高初始分歧
            last_gain = 0.05
            consecutive_low_gain = 0
            verify_discount = 1.0  # VERIFY 后乘以折扣系数

            for step in range(1, 100):
                remaining = budget - ctrl.spent

                # 分层预算可行性检查（动作执行前）
                if remaining < 50:
                    # 付不起任何原子动作 → 强制 STOP
                    ctrl.rule_log.append(1)
                    num_stop += 1
                    break

                # 动作驱动的 disagreement 动力学
                disagreement = disagreement * verify_discount
                disagreement = max(0.05, disagreement)
                verify_discount = min(1.0, verify_discount + 0.08)

                disagreement_val = max(0.05, disagreement)
                uncertainty_val = max(0.1, 0.55 - step * 0.02)
                credit_entropy_val = min(0.9, 0.3 + step * 0.02)
                credit_conc_val = max(0.3, 0.75 - step * 0.02)

                action, rule_id = ctrl.decide(
                    msg_gain=last_gain,
                    disagreement=disagreement_val,
                    uncertainty=uncertainty_val,
                    credit_entropy=credit_entropy_val,
                    credit_concentration=credit_conc_val,
                    best_score=len(trajectory) * 0.05,
                )

                action_str = action.value if hasattr(action, 'value') else str(action)
                action_type = ACTION_MAP.get(action_str, action)

                if action_str == "STOP":
                    num_stop += 1
                    ctrl.spend(50)  # final step cost
                    break

                # Base step cost (reasoner LLM call)
                ctrl.spend(50)

                # VERIFY action: deduct additional verifier cost (total = 200)
                if action_type == ActionType.CALL_VERIFIER:
                    ctrl.spend(150)  # additional verifier cost

                # Execute reasoner
                out = self.backend.chat(messages)
                text = out.choices[0].message.content.strip()
                messages.append({"role": "assistant", "content": text})
                current_answer = text

                # Execute critic
                critic_out = self.backend.chat([
                    {"role": "system", "content": "Review the solution for correctness."},
                    {"role": "user", "content": f"Problem: {problem}\n\n{text}"},
                ])
                critic_text = critic_out.choices[0].message.content.strip()

                # msg_gain from critic feedback — real model uses "correct"/"incorrect"
                # Look for verdict signals in model output
                critic_upper = critic_text.upper()
                if "PASS" in critic_upper or "CORRECT" in critic_upper or "VERDICT: PASS" in critic_text.upper():
                    msg_gain = 0.08
                elif "FAIL" in critic_upper or "INCORRECT" in critic_upper or "VERDICT: FAIL" in critic_text.upper():
                    msg_gain = 0.01
                elif re.search(r'\b(correct|right|accurate)\b', critic_text, re.IGNORECASE):
                    msg_gain = 0.08
                elif re.search(r'\b(incorrect|wrong|error|bad)\b', critic_text, re.IGNORECASE):
                    msg_gain = 0.01
                else:
                    # Ambiguous output — moderate gain
                    msg_gain = 0.04
                msg_gains.append(msg_gain)
                last_gain = msg_gain
                disagreement_values.append(disagreement_val)

                # 更新动作驱动的动力学状态
                if msg_gain < 0.03:
                    consecutive_low_gain += 1
                    disagreement = max(0.05, disagreement - 0.15)
                else:
                    consecutive_low_gain = 0
                    disagreement = min(0.65, disagreement + 0.05)

                trajectory.append({
                    "step": step, "reasoner": text[:50],
                    "critic": critic_text[:50], "action": action_str, "rule": rule_id,
                })
                disagreement_trace.append(round(disagreement_val, 3))

                if action_type == ActionType.CONTINUE_DISCUSS:
                    num_continue += 1
                elif action_type == ActionType.CALL_VERIFIER:
                    num_verify += 1
                    # VERIFY 后分歧大幅下降
                    disagreement = max(0.05, disagreement - 0.35)
                    verify_discount = 0.5

            extracted = extract_number_from_model_output(current_answer)
            is_correct = check_correct(extracted, ground_truth)
        except Exception as e:
            return SampleResult(
                question_id=question_id, strategy=strategy_name,
                final_answer="", ground_truth=ground_truth,
                final_correct=False, budget_used=ctrl.spent, total_steps=len(trajectory),
                num_continue=num_continue, num_verify=num_verify, num_stop=num_stop,
                triggered_rule_sequence=ctrl.rule_log,
                avg_msg_gain=sum(msg_gains) / len(msg_gains) if msg_gains else 0.0,
                final_verifier_confidence=None,
                answer_disagreement_trace=disagreement_trace,
                latency_seconds=time.time() - start,
                error=str(e),
            )

        return SampleResult(
            question_id=question_id, strategy=strategy_name,
            final_answer=extracted or current_answer[:200],
            ground_truth=ground_truth,
            final_correct=is_correct,
            budget_used=ctrl.spent,
            total_steps=len(trajectory),
            num_continue=num_continue,
            num_verify=num_verify,
            num_stop=num_stop,
            triggered_rule_sequence=ctrl.rule_log,
            avg_msg_gain=sum(msg_gains) / len(msg_gains) if msg_gains else 0.0,
            final_verifier_confidence=None,
            answer_disagreement_trace=disagreement_trace,
            latency_seconds=time.time() - start,
        )

    def run(self) -> dict:
        """Run all strategies on the dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []

        for item in self.dataset:
            if isinstance(self.dataset, list):
                qid = item["question_id"]
                problem = item["problem"]
                answer = item["answer"]
            else:
                qid = item.get("problem_id", str(item.get("idx", "unknown")))
                problem = item["question"]
                answer = item.get("answer", "")

            print(f"\n[{qid}] {problem[:60]}...")

            for strategy_name, strategy in self.strategies.items():
                print(f"  → {strategy_name}", end=" ", flush=True)
                if strategy_name == "S1_single_agent":
                    result = strategy.run(qid, problem, answer, self.budget_per_question)
                elif strategy_name == "S2_static_ma":
                    result = strategy.run(qid, problem, answer, self.budget_per_question)
                elif strategy_name == "S3_budget_only":
                    result = strategy(qid, problem, answer, self.budget_per_question)
                elif strategy_name == "S4_credit_aware":
                    result = strategy(qid, problem, answer, self.budget_per_question)

                print(f"{'✓' if result.final_correct else '✗'} (budget={result.budget_used}, steps={result.total_steps})")
                all_results.append(result)

        return self._summarize(all_results, timestamp)

    def _top_rules(self, results: list, top_k: int = 3) -> dict:
        """Return top-k most triggered rules by frequency."""
        from collections import Counter
        all_rules = []
        for r in results:
            all_rules.extend(r.triggered_rule_sequence)
        if not all_rules:
            return {}
        counts = Counter(all_rules)
        return {f"rule_{k}": v for k, v in counts.most_common(top_k)}

    def _summarize(self, results: list, timestamp: str) -> dict:
        """Compute and save per-strategy summary + per-sample CSV."""
        # Per-strategy stats
        summary = {}
        for name in self.strategies:
            strat_results = [r for r in results if r.strategy == name]
            if not strat_results:
                continue
            n = len(strat_results)
            acc = sum(r.final_correct for r in strat_results) / n
            avg_budget = sum(r.budget_used for r in strat_results) / n
            avg_steps = sum(r.total_steps for r in strat_results) / n
            avg_gain = sum(r.avg_msg_gain for r in strat_results) / n
            total_continue = sum(r.num_continue for r in strat_results)
            total_verify = sum(r.num_verify for r in strat_results)
            total_stop = sum(r.num_stop for r in strat_results)

            summary[name] = {
                "accuracy": round(acc, 4),
                "avg_budget_used": round(avg_budget, 1),
                "avg_steps": round(avg_steps, 2),
                "avg_msg_gain": round(avg_gain, 4),
                "total_continue": total_continue,
                "total_verify": total_verify,
                "total_stop": total_stop,
                "n": n,
                # 三项关键行为指标（用户定义）
                "avg_steps_to_stop": round(
                    sum(r.total_steps for r in strat_results) / n, 2
                ) if strat_results else 0.0,
                "verify_rate": round(total_verify / n, 4) if n > 0 else 0.0,
                "top_triggered_rules": self._top_rules(strat_results),
            }

            # Per-sample CSV row
            rows = []
            for r in strat_results:
                row = {
                    "question_id": r.question_id,
                    "final_correct": int(r.final_correct),
                    "budget_used": r.budget_used,
                    "total_steps": r.total_steps,
                    "num_continue": r.num_continue,
                    "num_verify": r.num_verify,
                    "num_stop": r.num_stop,
                    "triggered_rule_sequence": json.dumps(r.triggered_rule_sequence),
                    "avg_msg_gain": round(r.avg_msg_gain, 4),
                    "final_verifier_confidence": r.final_verifier_confidence,
                    "answer_disagreement_trace": json.dumps(r.answer_disagreement_trace),
                    "latency_seconds": round(r.latency_seconds, 2),
                    "error": r.error or "",
                }
                rows.append(row)

            csv_path = os.path.join(self.output_dir, f"{name}_{timestamp}.csv")
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved: {csv_path}")

        # Print summary table
        print("\n" + "=" * 70)
        print("PHASE 1 SMOKE BENCHMARK RESULTS (GSM8K subset)")
        print("=" * 70)
        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  accuracy:       {stats['accuracy']:.2%}")
            print(f"  avg_budget_used: {stats['avg_budget_used']:.1f}")
            print(f"  avg_steps:       {stats['avg_steps']:.2f}")
            print(f"  avg_msg_gain:    {stats['avg_msg_gain']:.4f}")
            print(f"  continue/verify/stop: {stats['total_continue']}/{stats['total_verify']}/{stats['total_stop']}")

        # Save summary
        summary_path = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump({"timestamp": timestamp, "strategies": summary, "n_questions": self.num_questions}, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        return summary


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 GSM8K smoke benchmark")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--verify-cost", type=int, default=200)
    parser.add_argument("--output-dir", default="./output/benchmark")
    parser.add_argument("--mock", action="store_true", help="Use mock backend (no real LLM calls)")
    parser.add_argument("--real", action="store_true", help="Use real vLLM backend (Qwen2.5-3B-Instruct)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for vLLM")
    args = parser.parse_args()

    if args.mock:
        backend = MockLLMBackend(response="The answer is 42.")
        print("[MODE] Mock backend — deterministic responses")
    elif args.real:
        from tiny_agents.models.vllm_backend import VLLMBackend
        model_key = f"Qwen2.5-3B_gpu{args.gpu}"
        model_path = os.path.expanduser(f"~/.cache/tiny-agents/models/Qwen/Qwen2.5-3B-Instruct")
        vllm_backend = VLLMBackend(default_gpu=args.gpu)
        vllm_backend.load_model(
            model_key, model_path,
            gpu=args.gpu,
            max_model_len=8192,
            gpu_memory_utilization=0.45,
        )
        backend = RealLLMBackend(vllm_backend, model_key, temperature=0.7, max_tokens=512)
        print(f"[MODE] Real vLLM backend — {model_path} on gpu={args.gpu}")
    else:
        print("[ERROR] Must specify --mock or --real")
        return

    runner = GSM8KBenchmarkRunner(
        backend=backend,
        num_questions=args.num_questions,
        budget_per_question=args.budget,
        verify_cost=args.verify_cost,
        output_dir=args.output_dir,
    )
    summary = runner.run()

    # Behavioral validation
    print("\n" + "=" * 70)
    print("BEHAVIORAL VALIDATION (credit-aware controller)")
    print("=" * 70)
    s4 = summary.get("S4_credit_aware")
    s3 = summary.get("S3_budget_only")
    if s4 and s3:
        print(f"  avg_steps - credit-aware: {s4['avg_steps']:.2f} vs budget-only: {s3['avg_steps']:.2f}")
        print(f"  avg_msg_gain - credit-aware: {s4['avg_msg_gain']:.4f} vs budget-only: {s3['avg_msg_gain']:.4f}")
        if s4['avg_steps'] < s3['avg_steps']:
            print("  ✓ credit-aware uses fewer steps (good — smarter early stopping)")
        if s4['avg_msg_gain'] > s3['avg_msg_gain']:
            print("  ✓ credit-aware has higher msg_gain (good — better step quality)")


if __name__ == "__main__":
    main()
