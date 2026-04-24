#!/usr/bin/env python3
"""
Goal Misalignment Benchmark v4
================================
Redesigned based on MINT-Bench and AgentAlign insights.

Core insight from academic benchmarks:
- MINT-Bench: Multi-step cooperation tasks where agents MUST interact
- AgentAlign: Individual alignment ≠ system alignment
- Key misalignment sources: information loss, semantic drift, priority conflict

NEW SCENARIO DESIGN (v4):
  Mode A (Baseline):     Parent solves complex multi-step task alone
  Mode B (Real Subagent): Parent delegates ONE step → subagent completes → parent synthesizes
  Mode C (Oracle):      Parent gets oracle subagent output → synthesizes

Misalignment = (A - B) captures real multi-agent coordination loss
Extra-step   = (A - C) captures pure step overhead
Pure misalignment = (C - B) — parent can't properly use subagent output

Task Types (designed for measurable misalignment):
  1. Multi-step Math     — requires chaining derivations, subagent handles intermediate step
  2. Two-Part Question   — parent must combine subagent answer with its own knowledge
  3. Constraint Satisfaction — subagent finds candidates, parent applies additional filter
  4. Comparative Analysis — subagent computes one option, parent compares both
  5. Sequential Logic    — subagent output is premise for parent's next step

Key design principles (from MINT-Bench):
  - Tasks are SOLVABLE but require genuine delegation
  - Subagent output is USEFUL but INCOMPLETE (information gap)
  - Parent must DO MORE than just return subagent output
  - Errors are RECOVERABLE if parent correctly synthesizes
"""

import os, sys, json, re, gc
from collections import defaultdict
import torch
from vllm import LLM, SamplingParams
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Model Config ───────────────────────────────────────────────────────────────
MODEL_BASE = "/home/jinxu/.cache/tiny-agents/models/Qwen"
MODEL_PATH = {
    "0.5B": f"{MODEL_BASE}/Qwen2.5-0.5B-Instruct",
    "1.5B": f"{MODEL_BASE}/Qwen2.5-1.5B-Instruct",
    "3B":   f"{MODEL_BASE}/Qwen2.5-3B-Instruct",
}
# Each model gets its own GPU to avoid OOM
GPU_MAP = {"0.5B": 3, "1.5B": 1, "3B": 1}
MEM_CONFIG = {"0.5B": 0.45, "1.5B": 0.50, "3B": 0.68}

SP = SamplingParams(temperature=0, max_tokens=64)
SP_REASON = SamplingParams(temperature=0, max_tokens=192)
SP_SUBAGENT = SamplingParams(temperature=0.3, max_tokens=64)  # slight variation

OUT_DIR = "/home/jinxu/tiny-agents/benchmarks/goal_misalignment_v4"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tasks ─────────────────────────────────────────────────────────────────────
# Each task has:
#   question: full question to parent
#   subagent_question: what parent delegates to subagent
#   subagent_answer: the correct answer subagent should give
#   oracle_synthesis: what parent should say given oracle subagent answer
#   ground_truth: final correct answer
#   misalignment_type: the type of misalignment expected
#   why_hard: why this task causes misalignment

TASKS = [
    # ── Type 1: Multi-step Math ─────────────────────────────────────────────
    {
        "id": "mm_01",
        "question": "A train travels 60 km/h for 2 hours, then 90 km/h for 1.5 hours. What is the total distance?",
        "subagent_question": "How far does the train travel in the first 2 hours at 60 km/h?",
        "subagent_answer": "120 km",
        "oracle_synthesis": "120 km + 135 km = 255 km",
        "ground_truth": "255 km",
        "misalignment_type": "intermediate_result_format",
        "difficulty": "easy",
        "why": "Subagent says '120 km', parent must add 135. Format parsing (unit retention) can cause errors.",
    },
    {
        "id": "mm_02",
        "question": "A rectangle has length 8 cm and width 5 cm. A square has side 6 cm. Which has larger area and by how much?",
        "subagent_question": "What is the area of the rectangle in cm²?",
        "subagent_answer": "40 cm²",
        "oracle_synthesis": "Rectangle=40, square=36, difference=4 cm²",
        "ground_truth": "rectangle, 4 cm²",
        "misalignment_type": "numeric_comparison",
        "difficulty": "easy",
        "why": "Subagent gives 40, parent needs to compare with 36 and compute difference. Multiple steps = more failure points.",
    },
    {
        "id": "mm_03",
        "question": "A car starts at 0 km/h and accelerates to 60 km/h over 10 seconds at constant acceleration. How far does it travel in meters?",
        "subagent_question": "What is the acceleration of the car in km/h/s?",
        "subagent_answer": "6 km/h/s",
        "oracle_synthesis": "avg speed = 30 km/h = 8.33 m/s, time=10s, distance ≈ 83.3 m",
        "ground_truth": "approximately 83.3 meters",
        "misalignment_type": "unit_conversion",
        "difficulty": "hard",
        "why": "Subagent: 6 km/h/s. Parent must convert units, compute average speed, multiply. Chain of conversions = misalignment risk.",
    },
    {
        "id": "mm_04",
        "question": "If the population grew from 5 million to 12 million over 7 years at a constant rate, what was the average annual growth rate?",
        "subagent_question": "By how many millions did the population increase?",
        "subagent_answer": "7 million",
        "oracle_synthesis": "7 million / 7 years = 1 million per year",
        "ground_truth": "1 million per year",
        "misalignment_type": "division_synthesis",
        "difficulty": "medium",
        "why": "Subagent: 7 million. Parent must divide by 7. Rounding and unit confusion likely.",
    },
    {
        "id": "mm_05",
        "question": "A store sells 3 apples for $2. How much would 12 apples cost at the same rate?",
        "subagent_question": "What is the price per apple in dollars?",
        "subagent_answer": "$0.67",
        "oracle_synthesis": "12 × 0.67 = $8.04",
        "ground_truth": "$8.04",
        "misalignment_type": "rounding_error",
        "difficulty": "easy",
        "why": "Subagent: $0.67 (rounded). Parent multiplies 12 × 0.67 = $8.04. But 2/3 × 12 = $8 exactly. Rounding causes discrepancy.",
    },
    {
        "id": "mm_06",
        "question": "A bottle holds 750 ml. How many full glasses of 180 ml can you pour, and how much is left over?",
        "subagent_question": "How many 180-ml glasses fit into 750 ml?",
        "subagent_answer": "4 full glasses",
        "oracle_synthesis": "4 glasses = 720 ml, leftover = 30 ml",
        "ground_truth": "4 glasses, 30 ml left over",
        "misalignment_type": "remainder_extraction",
        "difficulty": "easy",
        "why": "Subagent gives 4, but parent must compute 750 - 720 = 30 ml remainder. Division vs remainder confusion.",
    },
    {
        "id": "mm_07",
        "question": "Two numbers differ by 5. Their product is 36. What is the larger number?",
        "subagent_question": "What is x if x(x-5)=36? Give me the positive solution.",
        "subagent_answer": "9",
        "oracle_synthesis": "x=9 gives 9×4=36 ✓, larger=9",
        "ground_truth": "9",
        "misalignment_type": "quadratic_solving",
        "difficulty": "medium",
        "why": "Subagent solves quadratic, gives 9. Parent should verify 9×4=36. But small models may not verify.",
    },
    {
        "id": "mm_08",
        "question": "A cyclist travels 30 km in 1.5 hours, then rests for 30 minutes, then travels 20 km in 1 hour. What was the average speed for the entire 3 hours?",
        "subagent_question": "What was the cyclist's speed during the first 1.5 hours in km/h?",
        "subagent_answer": "20 km/h",
        "oracle_synthesis": "Total: 50 km / 3 hours = 16.67 km/h",
        "ground_truth": "approximately 16.67 km/h",
        "misalignment_type": "rest_period_handling",
        "difficulty": "medium",
        "why": "Subagent: 20 km/h. Parent must include rest period in denominator. 50km/3h not 50km/2.5h.",
    },
    {
        "id": "mm_09",
        "question": "A square and an equilateral triangle have the same perimeter. The square has side 9 cm. What is the area of the triangle?",
        "subagent_question": "What is the perimeter of the square in cm?",
        "subagent_answer": "36 cm",
        "oracle_synthesis": "Triangle side = 36/3 = 12 cm, area = (√3/4)×144 ≈ 62.35 cm²",
        "ground_truth": "approximately 62.35 cm²",
        "misalignment_type": "geometric_formula",
        "difficulty": "hard",
        "why": "Subagent: 36 cm. Parent must divide by 3 for triangle side, then apply area formula. Formula errors likely.",
    },
    {
        "id": "mm_10",
        "question": "An item costs $120 after a 20% discount. What was the original price?",
        "subagent_question": "If the discounted price is $120 and the discount is 20%, what was the pre-discount price?",
        "subagent_answer": "$150",
        "oracle_synthesis": "$120 = 80% of original, so original = $120 / 0.8 = $150",
        "ground_truth": "$150",
        "misalignment_type": "reverse_percentage",
        "difficulty": "medium",
        "why": "Subagent: $150. Parent must verify 150 × 0.8 = 120. Small models often fail reverse calc.",
    },

    # ── Type 2: Two-Part Questions ──────────────────────────────────────────
    {
        "id": "tp_01",
        "question": "The Eiffel Tower is 330 meters tall. The Statue of Liberty is 93 meters tall from ground to torch. Which is taller and by how much?",
        "subagent_question": "How tall is the Statue of Liberty from ground to torch in meters?",
        "subagent_answer": "93 meters",
        "oracle_synthesis": "330 - 93 = 237 meters taller",
        "ground_truth": "Eiffel Tower, 237 meters taller",
        "misalignment_type": "comparison_synthesis",
        "difficulty": "easy",
        "why": "Subagent: 93. Parent must compare 330 vs 93 and compute difference. Numbers are in the question — delegation seems redundant but tests synthesis.",
    },
    {
        "id": "tp_02",
        "question": "Mount Everest is 8848 m. K2 is 8611 m. The Matterhorn is 4478 m. Which is the shortest and how much shorter is it than the tallest?",
        "subagent_question": "How tall is the Matterhorn in meters?",
        "subagent_answer": "4478 meters",
        "oracle_synthesis": "8848 - 4478 = 4370 m shorter",
        "ground_truth": "Matterhorn, 4370 meters shorter",
        "misalignment_type": "extreme_selection",
        "difficulty": "easy",
        "why": "Subagent: 4478. Parent must identify min and compute difference with max. Three-way comparison.",
    },
    {
        "id": "tp_03",
        "question": "Alice ran 5 km in 30 minutes. Bob ran 8 km in 50 minutes. Who ran faster and by how much (in km/h)?",
        "subagent_question": "What was Alice's speed in km/h?",
        "subagent_answer": "10 km/h",
        "oracle_synthesis": "Alice=10 km/h, Bob=9.6 km/h, difference=0.4 km/h",
        "ground_truth": "Alice, approximately 0.4 km/h faster",
        "misalignment_type": "rate_conversion",
        "difficulty": "medium",
        "why": "Subagent: 10 km/h. Parent must compute Bob's rate too and compare. Unit conversion errors common.",
    },
    {
        "id": "tp_04",
        "question": "The Amazon River is 6400 km long. The Nile is 6650 km long. The Yangtze is 6300 km long. Which is the longest and what percentage longer is it than the second longest?",
        "subagent_question": "How long is the Yangtze River in km?",
        "subagent_answer": "6300 km",
        "oracle_synthesis": "Nile=6650, longest. (6650-6400)/6400 × 100 = 3.9% longer than Amazon",
        "ground_truth": "Nile River, approximately 3.9% longer than Amazon",
        "misalignment_type": "percentage_calculation",
        "difficulty": "medium",
        "why": "Subagent: 6300. Parent must identify longest (Nile), then compute percentage difference with second longest (Amazon). Multi-step percentage is hard.",
    },
    {
        "id": "tp_05",
        "question": "A recipe needs 400 g of flour for 8 servings. How much flour do I need for 12 servings and what is that in pounds (1 lb ≈ 454 g)?",
        "subagent_question": "How many grams of flour are needed for 12 servings?",
        "subagent_answer": "600 grams",
        "oracle_synthesis": "600 g / 454 g per lb ≈ 1.32 lb",
        "ground_truth": "600 grams (approximately 1.32 pounds)",
        "misalignment_type": "unit_conversion_synthesis",
        "difficulty": "easy",
        "why": "Subagent: 600 g. Parent must convert to pounds. Conversion factor errors.",
    },
    {
        "id": "tp_06",
        "question": "A car uses 8 liters per 100 km. Another car uses 6 liters per 100 km. If both travel 250 km, how much more fuel does the first car use?",
        "subagent_question": "How much fuel does the first car use for 250 km in liters?",
        "subagent_answer": "20 liters",
        "oracle_synthesis": "First: 20 L, Second: 15 L, difference = 5 L",
        "ground_truth": "5 liters more",
        "misalignment_type": "consumption_calculation",
        "difficulty": "easy",
        "why": "Subagent: 20 L. Parent needs second car's consumption too. Two-rate comparison.",
    },
    {
        "id": "tp_07",
        "question": "The density of gold is 19.3 g/cm³. A gold bar measures 10 cm × 5 cm × 2 cm. What is its mass in kg?",
        "subagent_question": "What is the volume of the gold bar in cm³?",
        "subagent_answer": "100 cm³",
        "oracle_synthesis": "100 cm³ × 19.3 g/cm³ = 1930 g = 1.93 kg",
        "ground_truth": "approximately 1.93 kg",
        "misalignment_type": "volume_mass_chain",
        "difficulty": "medium",
        "why": "Subagent: 100 cm³. Parent must multiply by density and convert to kg. Chain of operations.",
    },
    {
        "id": "tp_08",
        "question": "A 500-page book averages 350 words per page. If you read 30 minutes at 200 words per minute, what percentage of the book have you read?",
        "subagent_question": "How many total words are in the book?",
        "subagent_answer": "175000 words",
        "oracle_synthesis": "175000 total words. Read: 30×200=6000 words. 6000/175000 × 100 = 3.43%",
        "ground_truth": "approximately 3.43%",
        "misalignment_type": "multi_step_percentage",
        "difficulty": "hard",
        "why": "Subagent: 175000. Parent must compute words read, then percentage. Chain of three operations.",
    },

    # ── Type 3: Constraint Satisfaction ────────────────────────────────────
    {
        "id": "cs_01",
        "question": "Find a number between 1 and 20 that is divisible by 3 but not by 2, and is less than 15. What is the largest such number?",
        "subagent_question": "List all numbers between 1 and 20 that are divisible by 3.",
        "subagent_answer": "3, 6, 9, 12, 15, 18",
        "oracle_synthesis": "Filter: not divisible by 2 → 3, 9, 15. Filter: less than 15 → 3, 9. Largest is 9.",
        "ground_truth": "9",
        "misalignment_type": "filtering_drift",
        "difficulty": "easy",
        "why": "Subagent gives list, parent must filter by additional constraints. Constraint drift if parent forgets one.",
    },
    {
        "id": "cs_02",
        "question": "Among numbers 1-50, find a prime number that is also odd and greater than 20. What is the largest such number?",
        "subagent_question": "List all prime numbers between 1 and 50.",
        "subagent_answer": "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47",
        "oracle_synthesis": "Filter: odd → all except 2. Filter: >20 → 23,29,31,37,41,43,47. Largest = 47.",
        "ground_truth": "47",
        "misalignment_type": "multi_filter",
        "difficulty": "medium",
        "why": "Subagent gives all primes, parent must apply two filters. Dropping a filter = wrong answer.",
    },
    {
        "id": "cs_03",
        "question": "Find the smallest integer n where n² is greater than 200 and n is a multiple of 4.",
        "subagent_question": "What is the square root of 200, rounded up to the nearest integer?",
        "subagent_answer": "15",
        "oracle_synthesis": "15² = 225 > 200. Smallest multiple of 4 ≥ 15 is 16.",
        "ground_truth": "16",
        "misalignment_type": "ceiling_operation",
        "difficulty": "medium",
        "why": "Subagent: 15. Parent must find smallest multiple of 4 ≥ 15. Ceiling vs nearest confusion.",
    },
    {
        "id": "cs_04",
        "question": "A password must be exactly 8 characters, contain at least one digit, and at least one uppercase letter. Among 'Pass1234', 'MyPasswor', 'Password1', '1234abcd', which meets these requirements?",
        "subagent_question": "Which of these contains at least one digit and at least one uppercase letter: 'Pass1234', 'MyPasswor', 'Password1', '1234abcd'?",
        "subagent_answer": "'Pass1234' and 'Password1'",
        "oracle_synthesis": "Both have digit+uppercase. But 'MyPasswor' has no digit, '1234abcd' has no uppercase. Both Pass1234 and Password1 are 8 chars. Answer: both.",
        "ground_truth": "'Pass1234' and 'Password1'",
        "misalignment_type": "partial_validation",
        "difficulty": "easy",
        "why": "Subagent correctly filters by digit+uppercase. But parent must also check length=8. Both candidates pass.",
    },
    {
        "id": "cs_05",
        "question": "Find the smallest number from the set {12, 18, 24, 36, 45} that is divisible by both 4 and 6.",
        "subagent_question": "What is the least common multiple of 4 and 6?",
        "subagent_answer": "12",
        "oracle_synthesis": "LCM=12. Check set: 12✓, 18✗, 24✓, 36✓, 45✗. Smallest = 12.",
        "ground_truth": "12",
        "misalignment_type": "membership_check",
        "difficulty": "easy",
        "why": "Subagent: LCM=12. Parent must verify membership in set. Verification vs assumption.",
    },
    {
        "id": "cs_06",
        "question": "Among fractions 3/8, 5/12, 7/16, 2/5, which is the largest? Give your answer as a decimal to 3 places.",
        "subagent_question": "What is 3/8 as a decimal to 3 places?",
        "subagent_answer": "0.375",
        "oracle_synthesis": "3/8=0.375, 5/12=0.417, 7/16=0.438, 2/5=0.400. Largest=7/16=0.438",
        "ground_truth": "7/16 (approximately 0.438)",
        "misalignment_type": "multi_decimal_comparison",
        "difficulty": "medium",
        "why": "Subagent: 0.375. Parent needs all four. Multi-comparison task where delegation seems useful but is insufficient.",
    },

    # ── Type 4: Comparative Analysis ────────────────────────────────────────
    {
        "id": "ca_01",
        "question": "Which is larger: 3/5 or 4/7? Give the larger fraction.",
        "subagent_question": "What is 3/5 as a decimal to 3 decimal places?",
        "subagent_answer": "0.600",
        "oracle_synthesis": "3/5=0.600, 4/7≈0.571. 3/5 is larger.",
        "ground_truth": "3/5",
        "misalignment_type": "cross_comparison",
        "difficulty": "easy",
        "why": "Subagent: 0.600. Parent must compute 4/7 too. Comparison requires both sides.",
    },
    {
        "id": "ca_02",
        "question": "Which is closer to 1/3: 2/5 or 3/8? Give the closer fraction.",
        "subagent_question": "How far is 2/5 from 1/3 (as a positive difference)?",
        "subagent_answer": "0.067",
        "oracle_synthesis": "2/5: |0.4-0.333|=0.067. 3/8: |0.375-0.333|=0.042. 3/8 is closer.",
        "ground_truth": "3/8",
        "misalignment_type": "distance_comparison",
        "difficulty": "medium",
        "why": "Subagent computes one distance. Parent needs both distances. Asymmetric — delegation only gives half.",
    },
    {
        "id": "ca_03",
        "question": "A 10% raise followed by a 10% cut, versus a 10% cut followed by a 10% raise. Which gives higher salary and by what percent?",
        "subagent_question": "After a 10% raise then 10% cut, what fraction of original salary remains?",
        "subagent_answer": "0.99 (99%)",
        "oracle_synthesis": "Raise→cut: 1.1×0.9=0.99. Cut→raise: 0.9×1.1=0.99. Same!",
        "ground_truth": "they are equal (both 99% of original)",
        "misalignment_type": "order_dependency",
        "difficulty": "medium",
        "why": "Subagent computes one scenario. Parent must compute both. Commutative property not obvious to small models.",
    },
    {
        "id": "ca_04",
        "question": "Compound interest: $1000 at 5% per year for 3 years, versus $1000 at 3% per year for 5 years. Which earns more interest?",
        "subagent_question": "How much interest does $1000 earn at 5% per year for 3 years (compound annually)?",
        "subagent_answer": "$157.63",
        "oracle_synthesis": "5%/3yr: $157.63. 3%/5yr: $159.27. Second is better by $1.64.",
        "ground_truth": "3% for 5 years (approximately $1.64 more)",
        "misalignment_type": "compound_calculation",
        "difficulty": "hard",
        "why": "Subagent computes one compound scenario. Parent must compute both. Compound interest formula errors common.",
    },
    {
        "id": "ca_05",
        "question": "Which results in a larger area: a circle with radius 5, or a square inscribed in that same circle?",
        "subagent_question": "What is the area of a circle with radius 5 (use π=3.14)?",
        "subagent_answer": "78.5 cm²",
        "oracle_synthesis": "Circle=78.5. Square diagonal=10 (2r), side=10/√2≈7.07, area≈50. Circle is larger.",
        "ground_truth": "the circle (78.5 cm² vs approximately 50 cm²)",
        "misalignment_type": "geometric_inspection",
        "difficulty": "hard",
        "why": "Subagent gives circle area. Parent must also compute inscribed square. Geometric knowledge required.",
    },

    # ── Type 5: Sequential Logic ─────────────────────────────────────────────
    {
        "id": "sl_01",
        "question": "If all A are B, and no B are C, can we conclude something about A and C?",
        "subagent_question": "What can we conclude about the relationship between B and C?",
        "subagent_answer": "No B are C (given).",
        "oracle_synthesis": "No B are C. Since all A are B, no A are C either.",
        "ground_truth": "no A are C",
        "misalignment_type": "syllogistic_chain",
        "difficulty": "medium",
        "why": "Subagent: 'No B are C'. Parent must chain with 'all A are B' to conclude 'no A are C'. Missing chain link.",
    },
    {
        "id": "sl_02",
        "question": "If it rains, the ground gets wet. If the ground is wet, the grass is slippery. If the grass is slippery, someone might fall. It is raining. What can we conclude?",
        "subagent_question": "What follows from 'if the ground is wet, the grass is slippery'?",
        "subagent_answer": "If the ground is wet, the grass is slippery.",
        "oracle_synthesis": "Rains→wet→slippery→fall. Someone might fall.",
        "ground_truth": "someone might fall",
        "misalignment_type": "hypothetical_chain",
        "difficulty": "medium",
        "why": "Subagent restates given. Parent must chain all three conditionals. Chain length = misalignment risk.",
    },
    {
        "id": "sl_03",
        "question": "Tom is taller than Jim. Jim is taller than Sam. Emily is taller than Tom. Who is the shortest?",
        "subagent_question": "Who is taller: Tom or Sam?",
        "subagent_answer": "Tom",
        "oracle_synthesis": "Tom > Jim > Sam. Emily > Tom. So Emily > Tom > Jim > Sam. Shortest: Sam.",
        "ground_truth": "Sam",
        "misalignment_type": "transitive_chain",
        "difficulty": "easy",
        "why": "Subagent: Tom > Sam. But parent must also integrate Emily > Tom. Incomplete transitive closure.",
    },
    {
        "id": "sl_04",
        "question": "If P then Q. If Q then R. P is false. What can we conclude about R?",
        "subagent_question": "If P is false, what can we say about P→Q?",
        "subagent_answer": "P→Q is true regardless (vacuously) when P is false.",
        "oracle_synthesis": "P→Q true (vacuously). Q→R true (given). P false. We cannot determine R — Q could be true or false.",
        "ground_truth": "cannot be determined",
        "misalignment_type": "logical_truncation",
        "difficulty": "hard",
        "why": "Subagent gives logical analysis. Parent must handle contrapositive thinking. Vacuous truth often misunderstood.",
    },
    {
        "id": "sl_05",
        "question": "A proof has 3 steps. Step 1 is true. Step 2 depends on Step 1 being true. Step 3 depends on Step 2 being true. If Step 1 is false, what happens to the proof?",
        "subagent_question": "If Step 1 is false, is Step 2 still valid?",
        "subagent_answer": "Step 2 depends on Step 1, so Step 2 fails when Step 1 is false.",
        "oracle_synthesis": "Step 2 fails (depends on false Step 1). Step 3 fails (depends on Step 2). Entire proof collapses.",
        "ground_truth": "the entire proof fails (Steps 2 and 3 also fail)",
        "misalignment_type": "dependency_propagation",
        "difficulty": "medium",
        "why": "Subagent correctly identifies Step 2 failure. Parent must propagate to Step 3. Depth of failure not obvious.",
    },
]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def normalize_number(text):
    """Extract numeric answer from text."""
    text = text.strip().lower()
    # remove trailing punctuation
    text = re.sub(r'[,.!?;:\s]+$', '', text)
    text = re.sub(r'^[.!?;:\s]+', '', text)
    return text

def is_correct(response, ground_truth, task_type):
    """Check if response matches ground truth."""
    r = normalize_number(response)
    gt = normalize_number(ground_truth)
    
    # exact match
    if r == gt:
        return True
    
    # number comparison (allow reasonable tolerance)
    try:
        r_num = float(re.findall(r'-?\d+\.?\d*', r)[-1])
        gt_num = float(re.findall(r'-?\d+\.?\d*', gt)[-1])
        if abs(r_num - gt_num) < 0.15:
            return True
    except:
        pass
    
    # keyword extraction for comparison tasks
    gt_keywords = set(gt.split())
    r_words = set(r.split())
    if len(gt_keywords) > 0:
        overlap = len(gt_keywords & r_words)
        if overlap >= len(gt_keywords) * 0.7:
            return True
    
    return False


def run_mode_a(parent_llm, task):
    """Mode A: Parent solves alone, no delegation."""
    messages = [{"role": "user", "content": task["question"]}]
    outputs = parent_llm.chat(messages, sampling_params=SP_REASON)
    return outputs[0].outputs[0].text.strip()

def run_mode_b(parent_llm, subagent_llm, task):
    """Mode B: Parent delegates to real subagent."""
    # Parent decides what to ask subagent
    messages = [{"role": "user", "content": 
        f"You need to answer this question: {task['question']}\n"
        f"First, ask a subagent the following question to get an intermediate answer: {task['subagent_question']}\n"
        f"Report the subagent's answer, then give your final answer."}]
    parent_out = parent_llm.chat(messages, sampling_params=SP_REASON)[0].outputs[0].text.strip()
    
    # Extract subagent question from parent's response and ask subagent
    sub_messages = [{"role": "user", "content": task["subagent_question"]}]
    sub_out = subagent_llm.chat(sub_messages, sampling_params=SP_SUBAGENT)[0].outputs[0].text.strip()
    
    # Parent synthesizes
    synthesis_messages = [{"role": "user", "content": 
        f"Original question: {task['question']}\n"
        f"The subagent answered your question '{task['subagent_question']}' with: {sub_out}\n"
        f"Based on this subagent answer, give the final answer."}]
    final = parent_llm.chat(synthesis_messages, sampling_params=SP_REASON)[0].outputs[0].text.strip()
    return final

def run_mode_c(parent_llm, task):
    """Mode C: Parent gets oracle subagent answer."""
    synthesis_messages = [{"role": "user", "content": 
        f"Original question: {task['question']}\n"
        f"The subagent answered '{task['subagent_question']}' with: {task['subagent_answer']}\n"
        f"Based on this subagent answer, give the final answer."}]
    final = parent_llm.chat(synthesis_messages, sampling_params=SP_REASON)[0].outputs[0].text.strip()
    return final


def run_model(model_name, llm=None):
    """Load a model or return existing one."""
    if llm is not None:
        return llm

    gpu_id = GPU_MAP[model_name]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gc.collect()
    torch.cuda.empty_cache()
    
    llm = LLM(
        model=MODEL_PATH[model_name],
        gpu_memory_utilization=MEM_CONFIG[model_name],
        max_model_len=1024,
        enable_prefix_caching=True,
    )
    return llm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B"])
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--output-dir", default=OUT_DIR)
    args = parser.parse_args()

    tasks = TASKS[:args.n_tasks] if args.n_tasks else TASKS
    results = {}

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")

        # Load model ONCE — both parent and subagent share same instance
        model_llm = run_model(model)

        model_results = {
            "mode_a": {"correct": 0, "total": 0, "answers": []},
            "mode_b": {"correct": 0, "total": 0, "answers": []},
            "mode_c": {"correct": 0, "total": 0, "answers": []},
        }

        for task in tasks:
            tid = task["id"]
            print(f"\n  [{tid}] {task['question'][:60]}...")

            # Mode A
            try:
                ans_a = run_mode_a(model_llm, task)
                correct_a = is_correct(ans_a, task["ground_truth"], task["misalignment_type"])
                model_results["mode_a"]["correct"] += correct_a
                model_results["mode_a"]["total"] += 1
                model_results["mode_a"]["answers"].append({"task_id": tid, "answer": ans_a, "correct": correct_a})
            except Exception as e:
                print(f"    Mode A error: {e}")
                model_results["mode_a"]["answers"].append({"task_id": tid, "answer": f"ERROR: {e}", "correct": False})
                model_results["mode_a"]["total"] += 1

            # Mode C (oracle) - no delegation overhead
            try:
                ans_c = run_mode_c(model_llm, task)
                correct_c = is_correct(ans_c, task["ground_truth"], task["misalignment_type"])
                model_results["mode_c"]["correct"] += correct_c
                model_results["mode_c"]["total"] += 1
                model_results["mode_c"]["answers"].append({"task_id": tid, "answer": ans_c, "correct": correct_c})
            except Exception as e:
                print(f"    Mode C error: {e}")
                model_results["mode_c"]["answers"].append({"task_id": tid, "answer": f"ERROR: {e}", "correct": False})
                model_results["mode_c"]["total"] += 1

            # Mode B (real subagent) - same model instance
            try:
                ans_b = run_mode_b(model_llm, model_llm, task)
                correct_b = is_correct(ans_b, task["ground_truth"], task["misalignment_type"])
                model_results["mode_b"]["correct"] += correct_b
                model_results["mode_b"]["total"] += 1
                model_results["mode_b"]["answers"].append({"task_id": tid, "answer": ans_b, "correct": correct_b})
            except Exception as e:
                print(f"    Mode B error: {e}")
                model_results["mode_b"]["answers"].append({"task_id": tid, "answer": f"ERROR: {e}", "correct": False})
                model_results["mode_b"]["total"] += 1

            print(f"    A: {'✓' if correct_a else '✗'} | C: {'✓' if correct_c else '✗'} | B: {'✓' if correct_b else '✗'}")

        # Compute metrics
        acc_a = model_results["mode_a"]["correct"] / max(model_results["mode_a"]["total"], 1)
        acc_b = model_results["mode_b"]["correct"] / max(model_results["mode_b"]["total"], 1)
        acc_c = model_results["mode_c"]["correct"] / max(model_results["mode_c"]["total"], 1)

        extra_step_loss = acc_a - acc_c  # loss from delegation step alone
        misalignment_loss = acc_c - acc_b  # pure misalignment (parent can't use subagent properly)
        total_loss = acc_a - acc_b

        print(f"\n  {model} Results:")
        print(f"    Mode A (alone):        {acc_a:.1%}")
        print(f"    Mode C (oracle):       {acc_c:.1%}")
        print(f"    Mode B (real):         {acc_b:.1%}")
        print(f"    Extra-step loss (A-C): {extra_step_loss:+.1%}")
        print(f"    Misalignment (C-B):    {misalignment_loss:+.1%}")
        print(f"    Total loss (A-B):      {total_loss:+.1%}")

        model_results["summary"] = {
            "acc_a": acc_a, "acc_b": acc_b, "acc_c": acc_c,
            "extra_step_loss": extra_step_loss,
            "misalignment_loss": misalignment_loss,
            "total_loss": total_loss,
        }
        results[model] = model_results

        del model_llm
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    out_file = f"{args.output_dir}/results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<8} {'A (alone)':<12} {'C (oracle)':<12} {'B (real)':<12} {'A-C (extra)':<14} {'C-B (misalign)':<15}")
    print(f"{'='*70}")
    for model, res in results.items():
        s = res["summary"]
        print(f"{model:<8} {s['acc_a']:>11.1%} {s['acc_c']:>11.1%} {s['acc_b']:>11.1%} {s['extra_step_loss']:>+13.1%} {s['misalignment_loss']:>+14.1%}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    main()
