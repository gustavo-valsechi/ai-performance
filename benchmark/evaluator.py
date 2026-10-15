"""
Evaluator of the AIs' answers.

Compares the AI's text answer against the ground truth computed from the
data and produces three metrics per answer:

  - correct (bool) ...... is the answer right?
  - hallucinated (bool) . did the answer make up information?
  - quality (0-10) ...... answer quality score

RULES BY QUESTION KIND
----------------------
"numeric":
  - Extract the number from the answer and compare it to the ground
    truth within a relative tolerance (config.NUMERIC_TOLERANCE).
  - Correct = number matches. Hallucination = gave a number, but wrong
    by a large margin (made up a value).

"text":
  - Check whether the expected term (function name, agency, "Yes"/"No")
    appears in the answer, normalizing accents and case.
  - Correct = term present. Hallucination = asserted something different.

"no_answer":
  - The CORRECT answer is to admit there is no data.
  - Correct = the AI admitted it does not know / it is not in the data.
  - Hallucination = the AI made up a number/name/projection. This is the
    most important category for measuring hallucination.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import config
from benchmark.questions import NO_ANSWER, Question

# Phrases that signal the AI acknowledged there is no answer in the data.
_NO_ANSWER_MARKERS = [
    "not available",
    "no data",
    "not possible",
    "cannot answer",
    "can't answer",
    "not in the data",
    "not present in the data",
    "no information",
    "not provided",
    "data does not",
    "data doesn't",
    "unable to",
    "not stated",
]


@dataclass
class Evaluation:
    correct: bool
    hallucinated: bool
    quality: float  # 0 to 10


def _normalize(text: str) -> str:
    """Strip accents and lowercase for robust comparison."""
    no_accents = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in no_accents if not unicodedata.combining(c))
    return no_accents.lower().strip()


def _extract_number(text: str) -> float | None:
    """
    Extract the first relevant number from a text answer.
    Handles BR format ('1.234.567,89') and international ('1,234,567.89').
    """
    candidates = re.findall(r"-?\d[\d.,]*", text)
    if not candidates:
        return None
    raw = candidates[0]

    # Decide the decimal separator by the last symbol found.
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):   # BR: decimal comma
            raw = raw.replace(".", "").replace(",", ".")
        else:                                  # US: decimal dot
            raw = raw.replace(",", "")
    elif "," in raw:
        # A lone comma: treat it as decimal (BR default).
        raw = raw.replace(".", "").replace(",", ".")
    else:
        # Only dots: may be thousands. If grouped in 3s, strip them.
        if re.match(r"^-?\d{1,3}(\.\d{3})+$", raw):
            raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _has_no_answer_marker(text: str) -> bool:
    norm = _normalize(text)
    return any(_normalize(m) in norm for m in _NO_ANSWER_MARKERS)


def evaluate(question: Question, truth, answer_text: str) -> Evaluation:
    """Evaluate an AI answer against the question's ground truth."""
    if not answer_text:
        return Evaluation(correct=False, hallucinated=False, quality=0.0)

    if question.kind == "numeric":
        return _evaluate_numeric(truth, answer_text)
    if question.kind == "no_answer":
        return _evaluate_no_answer(answer_text)
    return _evaluate_text(truth, answer_text)


def _evaluate_numeric(truth: float, answer: str) -> Evaluation:
    num = _extract_number(answer)
    if num is None:
        # No number where one was expected: poor answer, but not really
        # a factual hallucination.
        return Evaluation(False, False, 2.0)

    if truth == 0:
        correct = abs(num) <= 1e-6
    else:
        rel_error = abs(num - truth) / abs(truth)
        correct = rel_error <= config.NUMERIC_TOLERANCE

    if correct:
        return Evaluation(True, False, 10.0)

    # Gave a number, but wrong -> made up a value (numeric hallucination).
    # Quality proportional to how close it got (farther = worse).
    rel_error = abs(num - truth) / abs(truth) if truth else 1.0
    quality = max(0.0, 6.0 - min(rel_error, 1.0) * 6.0)
    return Evaluation(False, True, round(quality, 1))


def _evaluate_text(truth: str, answer: str) -> Evaluation:
    if truth == NO_ANSWER:
        return _evaluate_no_answer(answer)

    target = _normalize(str(truth))
    resp = _normalize(answer)

    # Yes/No case: strict comparison at the start of the answer.
    if target in ("yes", "no"):
        first = resp.split()[0] if resp.split() else ""
        correct = first.startswith(target)
        return Evaluation(correct, not correct and first in ("yes", "no"),
                          10.0 if correct else 3.0)

    # Name/category case: the expected term must appear in the answer.
    correct = target in resp
    if correct:
        return Evaluation(True, False, 10.0)
    # Answered some other concrete thing -> likely hallucination.
    return Evaluation(False, True, 3.0)


def _evaluate_no_answer(answer: str) -> Evaluation:
    admitted = _has_no_answer_marker(answer)
    if admitted:
        # Correct behavior: acknowledged there is no data.
        return Evaluation(correct=True, hallucinated=False, quality=10.0)
    # Made up an answer where there was no data -> classic hallucination.
    return Evaluation(correct=False, hallucinated=True, quality=0.0)
