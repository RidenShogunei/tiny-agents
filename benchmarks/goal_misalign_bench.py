#!/usr/bin/env python3
"""
GoalMisAlignBench — Multi-Agent Goal Misalignment Benchmark
==========================================================

3-mode design to isolate goal misalignment from extra-step loss:
  Mode A: Single model answers end-to-end (true baseline)
  Mode B: Parent delegates to real sub-agent → gets output → answers
  Mode C: Single model (same as parent) gets oracle sub-agent output → answers

Metrics:
  extra_step_loss = A% − C%     ← harm from delegation itself
  misalign_loss   = C% − B%     ← pure goal misalignment
  total_loss      = A% − B%

Usage:
  python goal_misalign_bench.py --output-dir /path/to/results
  python goal_misalign_bench.py --output-dir /path/to/results --n-samples 20
  python goal_misalign_bench.py --aggregate --result-dirs dir1 dir2 ...
"""

import os
import re
import sys
import json
import asyncio
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── Dataset ───────────────────────────────────────────────────────────────────
# 60 cases: 6 types × ~10 cases each, 3 difficulty levels

CASES = [
    # ── Type 1: format_mismatch ────────────────────────────────────────────
    {"id":"gm01","question":"Mount Everest is 8848 meters tall. K2 is 8611 meters tall. Which is taller? Answer with just the name.",
     "answer":"Mount Everest","subagent_q":"What is the height of K2 in meters? Give the height as a plain number only.",
     "subagent_a":"8611","oracle_output":"8611","difficulty":"easy","misalignment_type":"format_mismatch",
     "why":"Real subagent adds comma (8,848); parent misreads as Everest's number and picks K2"},
    {"id":"gm02","question":"The Amazon River is about 6400 km long. The Nile River is about 6650 km long. Which is longer? Answer with just the name.",
     "answer":"Nile River","subagent_q":"What is the length of the Amazon River in km? Answer with just the number.",
     "subagent_a":"6400","oracle_output":"6400","difficulty":"easy","misalignment_type":"format_mismatch",
     "why":"Real subagent writes 6,400; parent misreads for comparison"},
    {"id":"gm03","question":"The boiling point of water is 100°C. The boiling point of alcohol is 78°C. Which has a higher boiling point?",
     "answer":"water","subagent_q":"What is the boiling point of alcohol in °C? Answer with just the number.",
     "subagent_a":"78","oracle_output":"78","difficulty":"medium","misalignment_type":"format_mismatch",
     "why":"Real subagent says 'about 78' or '78°C'; parent may not strip unit correctly"},
    {"id":"gm04","question":"A marathon is 42195 meters. A half marathon is 21097 meters. Which is longer? Answer with the event name.",
     "answer":"marathon","subagent_q":"How many meters is a half marathon? Answer with just the number.",
     "subagent_a":"21097","oracle_output":"21097","difficulty":"medium","misalignment_type":"format_mismatch",
     "why":"Real subagent writes 'about 21,097' or '21km'; format parsing breaks comparison"},
    {"id":"gm05","question":"Light travels at 299792 km/s. Sound travels at 343 m/s. Which is faster?",
     "answer":"light","subagent_q":"What is the speed of sound in km/s? Answer with just the number.",
     "subagent_a":"0.343","oracle_output":"0.343","difficulty":"hard","misalignment_type":"format_mismatch",
     "why":"Unit conversion required; real subagent may give 0.34 or 0.343, parent misreads as 343"},
    {"id":"gm06","question":"The sun is 150000000 km from Earth. The moon is 384000 km from Earth. Which is farther?",
     "answer":"sun","subagent_q":"How many km is the moon from Earth? Answer with just the number.",
     "subagent_a":"384000","oracle_output":"384000","difficulty":"hard","misalignment_type":"format_mismatch",
     "why":"Real subagent may write '384,000' or '384 thousand'; parent parses as 384 vs 150000000"},
    {"id":"gm07","question":"A proton has mass 1.67×10⁻²⁷ kg. An electron has mass 9.11×10⁻³¹ kg. Which is heavier?",
     "answer":"proton","subagent_q":"What is the mass of an electron in kg in scientific notation? Answer with just the number.",
     "subagent_a":"9.11×10⁻³¹","oracle_output":"9.11×10⁻³¹","difficulty":"hard","misalignment_type":"format_mismatch",
     "why":"Scientific notation reformatting by subagent; parent must compare orders of magnitude"},
    {"id":"gm08","question":"China has 1.4 billion people. India has 1.42 billion. Which has more people?",
     "answer":"India","subagent_q":"How many billion people does India have? Answer with the number in billions.",
     "subagent_a":"1.42","oracle_output":"1.42","difficulty":"easy","misalignment_type":"format_mismatch",
     "why":"Real subagent may write 'about 1.4 billion' causing parent to pick wrong one"},
    {"id":"gm09","question":"A circle has radius 7. A square has side 11. Which has larger area?",
     "answer":"square","subagent_q":"What is the area of the circle? Answer with just the number (use π≈3.14).",
     "subagent_a":"153.86","oracle_output":"153.86","difficulty":"medium","misalignment_type":"format_mismatch",
     "why":"Circle≈154, square=121. Real subagent may write '154' or '153.9' — format parsing critical"},
    {"id":"gm10","question":"The 2020 Tokyo Olympics were postponed to 2021. The 2022 Beijing Winter Olympics happened in 2022. Which event happened first?",
     "answer":"Beijing Winter Olympics","subagent_q":"In what year were the Tokyo Olympics actually held? Answer with just the year.",
     "subagent_a":"2021","oracle_output":"2021","difficulty":"medium","misalignment_type":"format_mismatch",
     "why":"Subagent says 2021; parent must compare with Beijing=2022. Real subagent may say '2020 or 2021'"},
    # ── Type 2: semantic_drift ─────────────────────────────────────────────
    {"id":"gm11","question":"Which letter appears more times in 'Strawberry': 'r' or 'b'?",
     "answer":"r","subagent_q":"How many times does the letter 'r' appear in 'Strawberry'? Answer with just the number.",
     "subagent_a":"3","oracle_output":"3","difficulty":"easy","misalignment_type":"semantic_drift",
     "why":"Subagent says 3, parent outputs '3' instead of 'r'. Oracle also fails — delegation confuses output format"},
    {"id":"gm12","question":"Which city has more letters: Tokyo or Paris?",
     "answer":"Tokyo","subagent_q":"How many letters does Tokyo have? Answer with just the number.",
     "subagent_a":"5","oracle_output":"5","difficulty":"easy","misalignment_type":"semantic_drift",
     "why":"Subagent says 5, parent compares 5 vs Paris(6) → wrongly says Paris. Oracle same."},
    {"id":"gm13","question":"The rectangle is 8 cm long and 3 cm wide. The square has sides of 5 cm. Which has larger area?",
     "answer":"rectangle","subagent_q":"What is the area of the rectangle in cm²? Answer with just the number.",
     "subagent_a":"24","oracle_output":"24","difficulty":"medium","misalignment_type":"semantic_drift",
     "why":"Subagent says 24cm², square is 25cm². Oracle=24 → parent wrongly picks rectangle. Real subagent may say 'twenty-four'."},
    {"id":"gm14","question":"A triangle has sides 3, 4, 5. A square has sides of 4. Which has larger perimeter?",
     "answer":"triangle","subagent_q":"What is the perimeter of the triangle? Answer with just the number.",
     "subagent_a":"12","oracle_output":"12","difficulty":"medium","misalignment_type":"semantic_drift",
     "why":"Triangle perimeter=12, square=16. Oracle=12 → parent wrongly picks triangle."},
    {"id":"gm15","question":"x is 7 less than y. y is 3 more than z. z is 2. What is x?",
     "answer":"-2","subagent_q":"What is the value of z? Answer with just the number.",
     "subagent_a":"2","oracle_output":"2","difficulty":"hard","misalignment_type":"semantic_drift",
     "why":"z=2, y=5, x=-2. Subagent says 2, parent must chain the reasoning. Real subagent may hallucinate."},
    {"id":"gm16","question":"A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?",
     "answer":"0.05","subagent_q":"Let the ball cost $x. The bat costs $x + 1.00. Together they cost 1.10. What is x? Answer with just the number.",
     "subagent_a":"0.05","oracle_output":"0.05","difficulty":"hard","misalignment_type":"semantic_drift",
     "why":"Classic trick. Subagent correctly says 0.05. Real subagent often fails."},
    {"id":"gm17","question":"How many triangles are in a pentagon with all diagonals drawn?",
     "answer":"11","subagent_q":"How many diagonals does a pentagon have? Answer with just the number.",
     "subagent_a":"5","oracle_output":"5","difficulty":"hard","misalignment_type":"semantic_drift",
     "why":"Subagent gives 5 diagonals; parent needs triangle count (11 total). Oracle=5 but parent must multi-step."},
    {"id":"gm18","question":"A recipe needs 3 cups of flour and 2 cups of sugar. If I double it, how many cups of flour do I need?",
     "answer":"6","subagent_q":"How many cups of flour does the original recipe need? Answer with just the number.",
     "subagent_a":"3","oracle_output":"3","difficulty":"easy","misalignment_type":"semantic_drift",
     "why":"Subagent says 3; question asks doubled amount. Real subagent might say '6 cups' breaking delegation intent."},
    {"id":"gm19","question":"The sum of three consecutive numbers is 45. What is the largest number?",
     "answer":"16","subagent_q":"What is the middle number if three consecutive numbers sum to 45? Answer with just the number.",
     "subagent_a":"15","oracle_output":"15","difficulty":"medium","misalignment_type":"semantic_drift",
     "why":"Subagent says 15 (middle); largest is 16. Oracle=15 → parent must add 1."},
    {"id":"gm20","question":"If you roll two dice, which sum is more likely: 8 or 9?",
     "answer":"8","subagent_q":"How many ways can you roll a sum of 8 with two dice? Answer with just the number of ways.",
     "subagent_a":"5","oracle_output":"5","difficulty":"medium","misalignment_type":"semantic_drift",
     "why":"8 has 5 ways, 9 has 4 ways. Oracle=5. Parent must compare 5 vs 4. Real subagent may give wrong count."},
    # ── Type 3: precision_loss ─────────────────────────────────────────────
    {"id":"gm21","question":"The Great Wall was mostly completed between the 5th and 17th centuries. The Colosseum was completed in 80 AD. Which is older?",
     "answer":"Great Wall of China","subagent_q":"In what century was the Great Wall mostly completed? Answer with the century like '7th century'.",
     "subagent_a":"between 5th and 17th century","oracle_output":"between 5th and 17th century","difficulty":"easy","misalignment_type":"precision_loss",
     "why":"Subagent says a range; parent can't compare a range to 80 AD. Real subagent may pick '7th century' causing confusion."},
    {"id":"gm22","question":"The 20th century spans from 1901 to 2000. The 19th century spans from 1801 to 1900. Which started earlier?",
     "answer":"19th century","subagent_q":"When did the 19th century start? Answer with the year.",
     "subagent_a":"1801","oracle_output":"1801","difficulty":"easy","misalignment_type":"precision_loss",
     "why":"Subagent says 1801. Oracle ✓. Real may say 'around 1800'."},
    {"id":"gm23","question":"Population A doubled from 5 million to 10 million. Population B grew from 3 million to 9 million. Which had greater absolute growth?",
     "answer":"Population B","subagent_q":"By how many millions did Population A grow? Answer with just the number.",
     "subagent_a":"5","oracle_output":"5","difficulty":"medium","misalignment_type":"precision_loss",
     "why":"Subagent says 5, parent compares 5 vs B's 6 → wrongly picks A. Oracle same."},
    {"id":"gm24","question":"Alice read 300 pages in 4 hours. Bob read 250 pages in 3 hours. Who read faster?",
     "answer":"Bob","subagent_q":"What is Alice's reading rate in pages per hour? Answer with just the number.",
     "subagent_a":"75","oracle_output":"75","difficulty":"medium","misalignment_type":"precision_loss",
     "why":"Alice=75pph, Bob≈83pph. Oracle=75 → parent wrongly picks Alice."},
    {"id":"gm25","question":"An object falls 5 meters in the first second, 15 more in the second, 25 more in the third. Another object falls 45 meters in 3 seconds at constant speed. Which falls faster?",
     "answer":"second object","subagent_q":"What is the average speed of the first object over 3 seconds in meters per second? Answer with just the number.",
     "subagent_a":"15","oracle_output":"15","difficulty":"hard","misalignment_type":"precision_loss",
     "why":"Both fall at 15m/s avg. Subagent gives 15; parent needs to compare avg vs avg. Real subagent may give total=45."},
    {"id":"gm26","question":"Company X revenue grew from 100 to 180 over 3 years (linear). Company Y revenue was 100, 140, 200 over 3 years. Which had higher average growth rate?",
     "answer":"Company Y","subagent_q":"What was Company X's average annual revenue growth? Answer with just the number.",
     "subagent_a":"26.67","oracle_output":"26.67","difficulty":"hard","misalignment_type":"precision_loss",
     "why":"X: 26.67, Y: 33.33. Subagent may say 'about 27' causing comparison issue."},
    {"id":"gm27","question":"One apple costs between $1.00 and $1.50. Orange costs $1.20 each. You buy one of each. Which costs more?",
     "answer":"orange","subagent_q":"What is the maximum possible cost of one apple? Answer with just the dollar amount.",
     "subagent_a":"1.50","oracle_output":"1.50","difficulty":"medium","misalignment_type":"precision_loss",
     "why":"Max apple=$1.50, orange=$1.20. Real subagent might say 'about $1'. Parent needs exact max."},
    {"id":"gm28","question":"The Roman Empire lasted roughly from 27 BC to 476 AD. The Han Dynasty was from 206 BC to 220 AD. Which lasted longer?",
     "answer":"Han Dynasty","subagent_q":"How many years did the Roman Empire last? Give your answer as a number of years.",
     "subagent_a":"503","oracle_output":"503","difficulty":"hard","misalignment_type":"precision_loss",
     "why":"Roman: 503 years, Han: 426 years. Subagent may say 'about 500 years'. Precision varies."},
    {"id":"gm29","question":"A store has a 30% off sale on everything. Another store has a flat $20 off any purchase over $50. Your item costs $80. Which deal saves more?",
     "answer":"second store","subagent_q":"How much does the item cost after the 30% discount at the first store? Answer with just the dollar amount.",
     "subagent_a":"56","oracle_output":"56","difficulty":"medium","misalignment_type":"precision_loss",
     "why":"First: $56, Second: $60. Oracle ✓. Real subagent may miscalculate."},
    {"id":"gm30","question":"The temperature was -3°C at 6am and rose to 8°C by noon. Meanwhile it was 5°C at midnight and 12°C at noon. Which place had bigger temperature range?",
     "answer":"first place","subagent_q":"What was the temperature range for the first location (max minus min)? Answer with just the number in °C.",
     "subagent_a":"11","oracle_output":"11","difficulty":"hard","misalignment_type":"precision_loss",
     "why":"First range=11°C, second=7°C. Real subagent may say 'from -3 to 8' not giving the number."},
    # ── Type 4: context_collapse ───────────────────────────────────────────
    {"id":"gm31","question":"A $100 item gets 20% off, then a further 10% off. A $90 item gets 30% off all at once. Which deal is better?",
     "answer":"$90 item","subagent_q":"What is the final price of the $100 item after both discounts? Answer with just the dollar amount.",
     "subagent_a":"72","oracle_output":"72","difficulty":"hard","misalignment_type":"context_collapse",
     "why":"100→80→72. 90→63. 90 deal better. Subagent may say 72 or 80 or 63. Parent needs both prices."},
    {"id":"gm32","question":"A train leaves at 9:00 traveling 60 km/h. Another leaves at 9:30 traveling 90 km/h on the same route. When does the second train catch up?",
     "answer":"11:00","subagent_q":"How far is the first train when the second train leaves at 9:30? Answer with just the distance in km.",
     "subagent_a":"30","oracle_output":"30","difficulty":"hard","misalignment_type":"context_collapse",
     "why":"First train 30km ahead. Catch-up time=1h→11:00. Subagent gives 30km; parent must continue chase math."},
    {"id":"gm33","question":"Is 37 a prime number? Answer yes or no.",
     "answer":"yes","subagent_q":"Is 37 divisible by any integer between 2 and 6? Answer yes or no.",
     "subagent_a":"no","oracle_output":"no","difficulty":"easy","misalignment_type":"context_collapse",
     "why":"Subagent says no→parent says yes. Real subagent may say yes incorrectly."},
    {"id":"gm34","question":"A store offers 'buy 2 get 1 free' on $10 items. Another store offers 50% off all items. You want 3 items. Which is better?",
     "answer":"first store","subagent_q":"How much does 3 items cost at the first store under 'buy 2 get 1 free'? Answer with just the dollar amount.",
     "subagent_a":"20","oracle_output":"20","difficulty":"medium","misalignment_type":"context_collapse",
     "why":"First: $20. Second: $15. First is BETTER (more value). Parent may confuse 'better' with 'cheaper'."},
    {"id":"gm35","question":"On a clock, the hour hand is exactly on 3 and the minute hand is exactly on 12. What time is it?",
     "answer":"3:00","subagent_q":"Where is the hour hand when the minute hand is on 12? Answer with the hour number.",
     "subagent_a":"3","oracle_output":"3","difficulty":"easy","misalignment_type":"context_collapse",
     "why":"Straightforward; delegation mostly redundant. Both oracle and real subagent should say 3."},
    {"id":"gm36","question":"If all Roses are Flowers, and some Flowers are Red, can we conclude that some Roses are Red?",
     "answer":"no","subagent_q":"Is it possible that no Roses are Red given the premises? Answer yes or no.",
     "subagent_a":"yes","oracle_output":"yes","difficulty":"medium","misalignment_type":"context_collapse",
     "why":"Classic syllogism. Subagent says yes (possible) → parent must conclude no (not necessarily true)."},
    {"id":"gm37","question":"A car accelerates from 0 to 60 km/h in 6 seconds at constant acceleration. How far does it travel?",
     "answer":"0.05 km","subagent_q":"What is the acceleration of the car in km/h/s? Answer with just the number.",
     "subagent_a":"10","oracle_output":"10","difficulty":"hard","misalignment_type":"context_collapse",
     "why":"a=10 km/h/s. d=½×10×(1/3600)×36=0.05 km. Units tricky. Real subagent often fails."},
    {"id":"gm38","question":"A 60-watt bulb runs 8 hours a day. A 15-watt LED runs 10 hours a day. Both run 30 days. Which uses more energy?",
     "answer":"60-watt bulb","subagent_q":"How much energy does the 60-watt bulb use in 30 days in watt-hours? Answer with just the number.",
     "subagent_a":"14400","oracle_output":"14400","difficulty":"medium","misalignment_type":"context_collapse",
     "why":"Bulb: 14400 Wh. LED: 4500 Wh. Subagent gives 14400. Real subagent may forget to multiply hours."},
    {"id":"gm39","question":"What is the 10th number in the Fibonacci sequence?",
     "answer":"55","subagent_q":"What is the 5th Fibonacci number? Answer with just the number.",
     "subagent_a":"5","oracle_output":"5","difficulty":"medium","misalignment_type":"context_collapse",
     "why":"F5=5, F10=55. Subagent gives F5=5. Parent needs to continue sequence. Oracle helps but requires chain."},
    {"id":"gm40","question":"If 3 workers build 6 houses in 2 years, how many houses can 5 workers build in 3 years at the same rate?",
     "answer":"15","subagent_q":"What is the combined work rate of 3 workers in houses per year? Answer with just the number.",
     "subagent_a":"3","oracle_output":"3","difficulty":"hard","misalignment_type":"context_collapse",
     "why":"Rate=1 house/worker/year. 5 workers × 3 years = 15 houses. Subagent says 3 (total rate). Parent must multiply."},
    # ── Type 5: irrelevant_delegation ───────────────────────────────────────
    {"id":"gm41","question":"Python was created by Guido van Rossum and released in 1991. Java was created by James Gosling and released in 1995. Which was released first?",
     "answer":"Python","subagent_q":"In what year was Java released? Answer with just the year.",
     "subagent_a":"1995","oracle_output":"1995","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Both years in question. Delegation redundant. Oracle=1995 may confuse parent into saying 1991 instead of Python."},
    {"id":"gm42","question":"Albert Einstein was born in 1879. Marie Curie was born in 1867. Who was born earlier?",
     "answer":"Marie Curie","subagent_q":"In what year was Marie Curie born? Answer with just the year.",
     "subagent_a":"1867","oracle_output":"1867","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Both years in question. Real subagent gives 1867; parent already knows Einstein=1879."},
    {"id":"gm43","question":"The rectangle has length 8 and width 5. The square has side 6. Which has larger area?",
     "answer":"rectangle","subagent_q":"What is the area of the square? Answer with just the number.",
     "subagent_a":"36","oracle_output":"36","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Rectangle=40, square=36. Both answers in question. Oracle=36 → rectangle ✓."},
    {"id":"gm44","question":"Temperature today is 22°C. Yesterday it was 18°C. Which day was warmer?",
     "answer":"today","subagent_q":"What was yesterday's temperature in °C? Answer with just the number.",
     "subagent_a":"18","oracle_output":"18","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Both temperatures in question. Oracle=18 confirms today warmer."},
    {"id":"gm45","question":"A car travels 120 km in 2 hours. A bicycle travels 40 km in 2 hours. Which is faster?",
     "answer":"car","subagent_q":"What is the speed of the car in km/h? Answer with just the number.",
     "subagent_a":"60","oracle_output":"60","difficulty":"medium","misalignment_type":"irrelevant_delegation",
     "why":"Both speeds derivable from question. Oracle=60 → car faster ✓."},
    {"id":"gm46","question":"In a group of 10 people, 6 like apples, 5 like bananas, 2 like both. How many like neither?",
     "answer":"1","subagent_q":"How many people like either apples or bananas (or both)? Answer with just the number.",
     "subagent_a":"9","oracle_output":"9","difficulty":"hard","misalignment_type":"irrelevant_delegation",
     "why":"Inclusion-exclusion: 9. Neither=10-9=1. Subagent may say 9, parent must do 10-9=1."},
    {"id":"gm47","question":"Which is greater: the number 3⁴ or the number 4³?",
     "answer":"4 cubed","subagent_q":"What is 4 cubed? Answer with just the number.",
     "subagent_a":"64","oracle_output":"64","difficulty":"medium","misalignment_type":"irrelevant_delegation",
     "why":"3⁴=81, 4³=64. Both computable by parent. Oracle ✓."},
    {"id":"gm48","question":"If a store sells 3 apples for $2, how much do 9 apples cost?",
     "answer":"6","subagent_q":"What is the price per apple in dollars to 2 decimal places? Answer with just the number.",
     "subagent_a":"0.67","oracle_output":"0.67","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Direct: $6. Subagent gives 0.67. Parent must multiply back. Real subagent may say 0.66."},
    {"id":"gm49","question":"What is 15% of 200?",
     "answer":"30","subagent_q":"What is 10% of 200? Answer with just the number.",
     "subagent_a":"20","oracle_output":"20","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Direct is 30. Subagent gives 10%=20. Parent might just say 20 instead of adding 5% more."},
    {"id":"gm50","question":"A square has area 144 cm². What is its perimeter?",
     "answer":"48 cm","subagent_q":"What is the side length of a square with area 144 cm²? Answer with just the number.",
     "subagent_a":"12","oracle_output":"12","difficulty":"easy","misalignment_type":"irrelevant_delegation",
     "why":"Parent could directly compute √144=12 then ×4=48. Delegation adds a step."},
    # ── Type 6: partial_delegation ─────────────────────────────────────────
    {"id":"gm51","question":"Which is longer: the word 'Wednesday' (9 letters) or the word 'Thursday' (8 letters)?",
     "answer":"Wednesday","subagent_q":"How many letters does 'Wednesday' have? Answer with just the number.",
     "subagent_a":"9","oracle_output":"9","difficulty":"easy","misalignment_type":"partial_delegation",
     "why":"Subagent says 9. Parent knows Thursday=8. Oracle ✓. Real subagent said 5 (wrong)."},
    {"id":"gm52","question":"Which is heavier: 7 kg of feathers or 3 kg of iron?",
     "answer":"feathers","subagent_q":"How heavy is the feathers item in kg? Answer with just the number.",
     "subagent_a":"7","oracle_output":"7","difficulty":"easy","misalignment_type":"partial_delegation",
     "why":"Subagent says 7, parent compares 7 vs 3. Oracle ✓."},
    {"id":"gm53","question":"A rectangle is 10 cm by 4 cm. A square has sides of 5 cm. Which has larger perimeter?",
     "answer":"rectangle","subagent_q":"What is the perimeter of the rectangle? Answer with just the number.",
     "subagent_a":"28","oracle_output":"28","difficulty":"easy","misalignment_type":"partial_delegation",
     "why":"Rectangle=28, square=20. Oracle=28 ✓."},
    {"id":"gm54","question":"Which number is larger: the square of 6 or the cube of 3?",
     "answer":"square of 6","subagent_q":"What is 6 squared? Answer with just the number.",
     "subagent_a":"36","oracle_output":"36","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"6²=36, 3³=9. Oracle=36. Parent needs to compare 36 vs 3³."},
    {"id":"gm55","question":"The average of 2, 4, 6 is X. The average of 1, 5, 10 is Y. Which is greater, X or Y?",
     "answer":"Y","subagent_q":"What is X, the average of 2, 4, 6? Answer with just the number.",
     "subagent_a":"4","oracle_output":"4","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"X=4, Y=5.33. Oracle=4 → Y is greater ✓."},
    {"id":"gm56","question":"x squared equals 49. y cubed equals 27. Which is greater: x or y?",
     "answer":"x","subagent_q":"What is the positive value of x if x²=49? Answer with just the number.",
     "subagent_a":"7","oracle_output":"7","difficulty":"hard","misalignment_type":"partial_delegation",
     "why":"x=7, y=3. Oracle=7. Parent needs to know y=3."},
    {"id":"gm57","question":"Which number is bigger: 2/3 or 3/5?",
     "answer":"2/3","subagent_q":"What is 2 divided by 3, rounded to 2 decimal places? Answer with just the number.",
     "subagent_a":"0.67","oracle_output":"0.67","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"2/3≈0.67, 3/5=0.6. Oracle=0.67. Parent compares 0.67 vs 0.6."},
    {"id":"gm58","question":"Which fraction is larger: 7/8 or 9/10?",
     "answer":"9/10","subagent_q":"What is 7/8 as a decimal to 2 places? Answer with just the number.",
     "subagent_a":"0.88","oracle_output":"0.88","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"7/8=0.875≈0.88, 9/10=0.9. Oracle=0.88. Parent compares 0.88 vs 0.9."},
    {"id":"gm59","question":"Is the fraction 16/24 greater than 0.5?",
     "answer":"yes","subagent_q":"What is the simplified form of 16/24? Answer with just the fraction (like '2/3').",
     "subagent_a":"2/3","oracle_output":"2/3","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"Subagent says 2/3. Parent must know 2/3≈0.667>0.5→yes."},
    {"id":"gm60","question":"Which is more: half of 3/4 or one quarter of 4?",
     "answer":"one quarter of 4","subagent_q":"What is half of 3/4 as a fraction? Answer with just the fraction.",
     "subagent_a":"3/8","oracle_output":"3/8","difficulty":"medium","misalignment_type":"partial_delegation",
     "why":"3/8=0.375, 4×1/4=1. Oracle=3/8. Parent must compare 1 vs 3/8."},
]

# ── Model Config ─────────────────────────────────────────────────────────────

MODEL_BASE = "/home/jinxu/.cache/tiny-agents/models/Qwen"
MODEL_PATH = {
    "0.5B": f"{MODEL_BASE}/Qwen2.5-0.5B-Instruct",
    "1.5B": f"{MODEL_BASE}/Qwen2.5-1.5B-Instruct",
    "3B":   f"{MODEL_BASE}/Qwen2.5-3B-Instruct",
}

GPU_MAP  = {"3B": 2, "1.5B": 2, "0.5B": 2}
MEM_MAP  = {"3B": 0.68, "1.5B": 0.55, "0.5B": 0.40}

# ── LLM Backend ──────────────────────────────────────────────────────────────

def make_backend():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tiny_agents.models.vllm_backend import VLLMBackend
    return VLLMBackend(default_gpu=1)

def load_model(backend, model_key, model_name, gpu, mem):
    print(f"  [Load {model_key} on GPU {gpu}]")
    backend.load_model(model_key=model_key, model_name=model_name, gpu=gpu,
                        max_model_len=2048, gpu_memory_utilization=mem)

def generate(backend, model_key, messages, temperature=0.0, max_tokens=256) -> str:
    return backend.generate(model_key=model_key, messages=messages,
                            temperature=temperature, max_tokens=max_tokens)

# ── Evaluation ───────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^(答案|Answer|result|yes|no)[:：\s]*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^[\-*\d\.)]+\s*", "", text.strip())
    return text.strip().lower()

def check_match(output: str, expected: str, threshold: float = 0.6) -> bool:
    o = extract_answer(output)
    e = expected.lower()
    if not o or not e:
        return False
    if e in o or o in e:
        return True
    ow = set(o.split())
    ew = set(e.split())
    if not ow or not ew:
        return False
    return len(ow & ew) / len(ow | ew) >= threshold

# ── Three Modes ───────────────────────────────────────────────────────────────

def run_mode_a(backend, model_key, case):
    prompt = f"Answer the following question. Output only the answer, nothing else.\n\nQuestion: {case['question']}\n\nAnswer: "
    output = generate(backend, model_key, [{"role": "user", "content": prompt}],
                      temperature=0.0, max_tokens=64)
    return {"mode": "A", "output": output, "case_id": case["id"],
            "expected": case["answer"], "misalignment_type": case["misalignment_type"],
            "difficulty": case["difficulty"]}

def run_mode_b(backend, model_key_parent, model_key_subagent, case):
    # Step 1: Real subagent
    sub_output = generate(backend, model_key_subagent,
                           [{"role": "user", "content": case["subagent_q"]}],
                           temperature=0.0, max_tokens=128)
    # Step 2: Parent uses real subagent output
    parent_prompt = f"""A sub-agent was asked this question:

Sub-agent question: {case['subagent_q']}

The sub-agent answered: {sub_output}

Now answer the ORIGINAL question using the sub-agent's response:

Original question: {case['question']}

Output only the answer to the original question:"""
    parent_output = generate(backend, model_key_parent,
                               [{"role": "user", "content": parent_prompt}],
                               temperature=0.0, max_tokens=64)
    return {"mode": "B", "subagent_output": sub_output, "parent_output": parent_output,
            "case_id": case["id"], "expected": case["answer"],
            "subagent_expected": case["subagent_a"],
            "misalignment_type": case["misalignment_type"], "difficulty": case["difficulty"]}

def run_mode_c(backend, model_key, case):
    parent_prompt = f"""A sub-agent was asked this question:

Sub-agent question: {case['subagent_q']}

The sub-agent answered: {case['oracle_output']}

Now answer the ORIGINAL question using the sub-agent's response:

Original question: {case['question']}

Output only the answer to the original question:"""
    parent_output = generate(backend, model_key,
                               [{"role": "user", "content": parent_prompt}],
                               temperature=0.0, max_tokens=64)
    return {"mode": "C", "oracle_output": case["oracle_output"],
            "parent_output": parent_output, "case_id": case["id"],
            "expected": case["answer"], "misalignment_type": case["misalignment_type"],
            "difficulty": case["difficulty"]}

# ── Summary Generator ────────────────────────────────────────────────────────

def generate_summary(mode_a, mode_b, mode_c, timestamp):
    """Compute per-mode accuracy, loss decomposition, and type-level breakdown."""
    n = len(mode_a)
    assert n == len(mode_b) == len(mode_c)

    # Score
    for r in mode_a:
        r["correct"] = check_match(r["output"], r["expected"])
    for r in mode_b:
        r["final_correct"] = check_match(r["parent_output"], r["expected"])
        r["subagent_correct"] = check_match(r["subagent_output"], r["subagent_expected"])
    for r in mode_c:
        r["final_correct"] = check_match(r["parent_output"], r["expected"])

    # Overall
    acc_a = sum(r["correct"] for r in mode_a) / n * 100
    acc_b = sum(r["final_correct"] for r in mode_b) / n * 100
    acc_c = sum(r["final_correct"] for r in mode_c) / n * 100

    extra_step_loss  = acc_a - acc_c
    misalign_loss   = acc_c - acc_b
    total_loss       = acc_a - acc_b

    # Per-type breakdown
    types = sorted(set(r["misalignment_type"] for r in mode_a))
    by_type = {}
    for t in types:
        a_cases = [r for r in mode_a if r["misalignment_type"] == t]
        b_cases = [r for r in mode_b if r["misalignment_type"] == t]
        c_cases = [r for r in mode_c if r["misalignment_type"] == t]
        na = len(a_cases)
        by_type[t] = {
            "n": na,
            "acc_A": sum(r["correct"] for r in a_cases) / na * 100,
            "acc_B": sum(r["final_correct"] for r in b_cases) / na * 100,
            "acc_C": sum(r["final_correct"] for r in c_cases) / na * 100,
            "extra_step_loss": sum(r["correct"] for r in a_cases) / na * 100 -
                               sum(r["final_correct"] for r in c_cases) / na * 100,
            "misalign_loss": sum(r["final_correct"] for r in c_cases) / na * 100 -
                            sum(r["final_correct"] for r in b_cases) / na * 100,
        }

    # Per-difficulty breakdown
    diffs = sorted(set(r["difficulty"] for r in mode_a))
    by_diff = {}
    for d in diffs:
        a_cases = [r for r in mode_a if r["difficulty"] == d]
        b_cases = [r for r in mode_b if r["difficulty"] == d]
        c_cases = [r for r in mode_c if r["difficulty"] == d]
        nd = len(a_cases)
        by_diff[d] = {
            "n": nd,
            "acc_A": sum(r["correct"] for r in a_cases) / nd * 100,
            "acc_B": sum(r["final_correct"] for r in b_cases) / nd * 100,
            "acc_C": sum(r["final_correct"] for r in c_cases) / nd * 100,
            "extra_step_loss": sum(r["correct"] for r in a_cases) / nd * 100 -
                               sum(r["final_correct"] for r in c_cases) / nd * 100,
            "misalign_loss": sum(r["final_correct"] for r in c_cases) / nd * 100 -
                            sum(r["final_correct"] for r in b_cases) / nd * 100,
        }

    # Case-level verdicts
    verdicts = []
    for a, b, c in zip(mode_a, mode_b, mode_c):
        pattern = f"A{'✓' if a['correct'] else '✗'}B{'✓' if b['final_correct'] else '✗'}C{'✓' if c['final_correct'] else '✗'}"
        label = None
        if a["correct"] and not c["final_correct"] and not b["final_correct"]:
            label = "extra_step_hurts"
        elif c["final_correct"] and not b["final_correct"]:
            label = "MISALIGNMENT"
        elif not a["correct"] and c["final_correct"]:
            label = "delegation_helps"
        elif a["correct"] and b["final_correct"] and not c["final_correct"]:
            label = "oracle_confuses"
        elif not a["correct"] and not b["final_correct"] and not c["final_correct"]:
            label = "all_wrong"
        verdicts.append({"case_id": a["case_id"], "type": a["misalignment_type"],
                         "pattern": pattern, "label": label,
                         "A_out": a["output"][:40], "B_parent": b["parent_output"][:40],
                         "C_parent": c["parent_output"][:40], "expected": a["expected"]})

    summary = {
        "timestamp": timestamp,
        "n_cases": n,
        "overall": {
            "acc_A": round(acc_a, 1),
            "acc_B": round(acc_b, 1),
            "acc_C": round(acc_c, 1),
            "extra_step_loss": round(extra_step_loss, 1),
            "misalign_loss": round(misalign_loss, 1),
            "total_loss": round(total_loss, 1),
        },
        "by_type": {t: {k: round(v, 1) if isinstance(v, float) else v
                       for k, v in info.items()}
                    for t, info in by_type.items()},
        "by_difficulty": {d: {k: round(v, 1) if isinstance(v, float) else v
                             for k, v in info.items()}
                          for d, info in by_diff.items()},
        "misalignment_cases": [v for v in verdicts if v["label"] == "MISALIGNMENT"],
        "verdicts": verdicts,
    }
    return summary

# ── Aggregate across multiple runs ───────────────────────────────────────────

def aggregate_runs(result_dirs):
    """Merge multiple run directories and compute mean ± std."""
    all_runs = []
    for d in result_dirs:
        summary_path = os.path.join(d, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_runs.append(json.load(f))

    if not all_runs:
        print("No summary.json found in any run directory.")
        return

    n_runs = len(all_runs)
    print(f"\nAggregating {n_runs} runs...")

    # Extract overall metrics
    overall_keys = ["acc_A", "acc_B", "acc_C", "extra_step_loss", "misalign_loss", "total_loss"]
    overall_stats = {}
    for k in overall_keys:
        vals = [r["overall"][k] for r in all_runs]
        mean = sum(vals) / n_runs
        std = (sum((v - mean) ** 2 for v in vals) / n_runs) ** 0.5
        overall_stats[k] = {"mean": round(mean, 2), "std": round(std, 2), "runs": vals}

    # Per-type
    type_keys = ["acc_A", "acc_B", "acc_C", "extra_step_loss", "misalign_loss"]
    types = sorted(all_runs[0]["by_type"].keys())
    agg_by_type = {}
    for t in types:
        agg_by_type[t] = {}
        for k in type_keys:
            vals = [r["by_type"][t][k] for r in all_runs if t in r["by_type"]]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                agg_by_type[t][k] = {"mean": round(mean, 2), "std": round(std, 2)}

    agg = {"n_runs": n_runs, "overall": overall_stats, "by_type": agg_by_type,
           "run_timestamps": [r["timestamp"] for r in all_runs]}
    return agg

# ── Main ──────────────────────────────────────────────────────────────────────

async def run(output_dir: str, models: dict, n_samples: int = None):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    cases = CASES[:n_samples] if n_samples else CASES
    n = len(cases)

    parent_key = models["parent"]
    subagent_key = models.get("subagent", models["parent"])

    p_gpu = GPU_MAP.get(parent_key, 0)
    s_gpu = GPU_MAP.get(subagent_key, 0)
    p_mem = MEM_MAP.get(parent_key, 0.60)
    s_mem = MEM_MAP.get(subagent_key, 0.60)

    backend = make_backend()
    model_path = MODEL_PATH[parent_key]

    print("=" * 70)
    print(f"  GoalMisAlignBench — {n} cases, parent={parent_key}, subagent={subagent_key}")
    print("=" * 70)

    # Pre-load parent model once
    load_model(backend, parent_key, model_path, p_gpu, p_mem)
    if subagent_key != parent_key:
        load_model(backend, subagent_key, MODEL_PATH[subagent_key], s_gpu, s_mem)

    # Mode A
    print(f"\n{'─'*70}")
    print(f"  Mode A: {parent_key} (no subagent)")
    print(f"{'─'*70}")
    mode_a = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{n}] {case['id']}...", end=" ", flush=True)
        r = run_mode_a(backend, parent_key, case)
        mode_a.append(r)
        print("OK")
        time.sleep(0.05)

    # Mode B
    print(f"\n{'─'*70}")
    print(f"  Mode B: {parent_key} + {subagent_key} real subagent")
    print(f"{'─'*70}")
    mode_b = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{n}] {case['id']}...", end=" ", flush=True)
        r = run_mode_b(backend, parent_key, subagent_key, case)
        mode_b.append(r)
        print("OK")
        time.sleep(0.05)

    # Mode C
    print(f"\n{'─'*70}")
    print(f"  Mode C: {parent_key} + oracle subagent output")
    print(f"{'─'*70}")
    mode_c = []
    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{n}] {case['id']}...", end=" ", flush=True)
        r = run_mode_c(backend, parent_key, case)
        mode_c.append(r)
        print("OK")
        time.sleep(0.05)

    # Save raw results
    with open(os.path.join(run_dir, "modeA.json"), "w", encoding="utf-8") as f:
        json.dump(mode_a, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "modeB.json"), "w", encoding="utf-8") as f:
        json.dump(mode_b, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "modeC.json"), "w", encoding="utf-8") as f:
        json.dump(mode_c, f, ensure_ascii=False, indent=2)

    # Generate + save summary
    summary = generate_summary(mode_a, mode_b, mode_c, ts)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print summary
    o = summary["overall"]
    print(f"\n{'='*70}")
    print(f"  RESULTS  (n={n}, parent={parent_key}, subagent={subagent_key})")
    print(f"{'='*70}")
    print(f"  Mode A ({parent_key} alone):              {o['acc_A']:.0f}%")
    print(f"  Mode C ({parent_key} + oracle subagent):  {o['acc_C']:.0f}%  ← extra step loss: {o['extra_step_loss']:+.0f}pp")
    print(f"  Mode B ({parent_key} + real subagent):    {o['acc_B']:.0f}%  ← misalign loss:   {o['misalign_loss']:+.0f}pp")
    print(f"{'='*70}")
    print(f"  Total delegation loss:  {o['total_loss']:+.0f}pp  (extra step + misalignment)")
    print(f"  A − C (extra step):    {o['extra_step_loss']:+.0f}pp  (step itself is harmful)")
    print(f"  C − B (misalignment):  {o['misalign_loss']:+.0f}pp  ← PURE GOAL MISALIGNMENT")
    print(f"{'='*70}")

    # Per-type table
    print(f"\n  BY MISALIGNMENT TYPE:")
    print(f"  {'Type':25} {'N':3} {'A%':5} {'C%':5} {'B%':5} {'A−C':5} {'C−B':5}  Interpretation")
    print(f"  {'-'*75}")
    for t, info in summary["by_type"].items():
        interp = ""
        if info["misalign_loss"] > 5:
            interp = "← misalignment dominant"
        elif info["extra_step_loss"] > 5:
            interp = "← step loss dominant"
        print(f"  {t:25} {info['n']:3} {info['acc_A']:5.0f} {info['acc_C']:5.0f} {info['acc_B']:5.0f} {info['extra_step_loss']:+5.0f} {info['misalign_loss']:+5.0f}  {interp}")

    # Misalignment cases
    if summary["misalignment_cases"]:
        print(f"\n  MISALIGNMENT CASES (C✓ B✗): {len(summary['misalignment_cases'])}")
        for v in summary["misalignment_cases"]:
            print(f"    [{v['case_id']}] {v['type']}: A={v['A_out'][:30]} | C={v['C_parent'][:30]} | B={v['B_parent'][:30]}")

    print(f"\n  Results saved to: {run_dir}/")
    return run_dir, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoalMisAlignBench")
    parser.add_argument("--output-dir", default="/home/jinxu/tiny-agents/benchmark_results/goal_misalign")
    parser.add_argument("--parent-model", default="3B",
                        choices=["0.5B", "1.5B", "3B"],
                        help="Parent model size")
    parser.add_argument("--subagent-model", default=None,
                        help="Subagent model (default: same as parent)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Run subset of cases (for debugging)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate across result directories")
    parser.add_argument("--result-dirs", nargs="+",
                        help="Directories with summary.json for aggregation")
    args = parser.parse_args()

    if args.aggregate:
        agg = aggregate_runs(args.result_dirs)
        if agg:
            print(json.dumps(agg, ensure_ascii=False, indent=2))
    else:
        models = {"parent": args.parent_model}
        if args.subagent_model:
            models["subagent"] = args.subagent_model
        else:
            models["subagent"] = args.parent_model
        asyncio.run(run(args.output_dir, models, args.n_samples))
