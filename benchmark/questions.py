"""
The 20 benchmark questions + the ground-truth engine.

CORE IDEA
---------
To measure ACCURACY and HALLUCINATION honestly, we need a reliable
ground truth. Instead of trusting hand-written answers, each question
has a Python function that COMPUTES the correct answer directly from
the data with pandas (deterministic and reproducible).

Each question is a Question object with:
  - id ............ short identifier (e.g. "T01")
  - category ...... one of the study categories
  - text .......... the natural-language prompt sent to the AI
  - source ........ "expenses" | "revenues" (which DataFrame to use)
  - kind .......... "numeric" | "text" | "no_answer"
  - truth(dfs) .... function returning the correct answer
  - unit .......... optional label ("R$", "%") only for display

CATEGORIES (the same ones requested in the study):
  - "Totals and periods"
  - "By category"
  - "Trend and comparison"
  - "Insights and decision"
  - "No answer in the data"   (hallucination test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

# Marker used when the correct answer is "the data cannot answer this".
NO_ANSWER = "__NO_ANSWER__"


@dataclass
class Question:
    id: str
    category: str
    text: str
    source: str                      # "expenses" or "revenues"
    kind: str                        # "numeric" | "text" | "no_answer"
    truth: Callable[[dict[str, pd.DataFrame]], Any]
    unit: str = ""


# ---------------------------------------------------------------------
# Helper computations (reused by the ground-truth functions)
# ---------------------------------------------------------------------
def _sum(df: pd.DataFrame, column: str) -> float:
    return float(df[column].sum()) if column in df.columns else 0.0


def _top_category(df: pd.DataFrame, group: str, value: str) -> str:
    grouped = df.groupby(group)[value].sum().sort_values(ascending=False)
    return str(grouped.index[0]) if len(grouped) else NO_ANSWER


# ---------------------------------------------------------------------
# THE 20 QUESTIONS
# ---------------------------------------------------------------------
# "d" is the dict {"expenses": df, "revenues": df} passed to truth().

QUESTIONS: list[Question] = [
    # ---------- Totals and periods (4) ----------
    Question(
        "T01", "Totals and periods",
        "What was the total amount paid in expenses in the period? "
        "Answer only with the numeric value in reais.",
        "expenses", "numeric",
        lambda d: _sum(d["expenses"], "paid"), "R$",
    ),
    Question(
        "T02", "Totals and periods",
        "What was the total amount committed in expenses in the period? "
        "Answer only with the number in reais.",
        "expenses", "numeric",
        lambda d: _sum(d["expenses"], "committed"), "R$",
    ),
    Question(
        "T03", "Totals and periods",
        "What was the total realized revenue in the period? "
        "Answer only with the number in reais.",
        "revenues", "numeric",
        lambda d: _sum(d["revenues"], "realized"), "R$",
    ),
    Question(
        "T04", "Totals and periods",
        "How many expense records (rows) are there in the dataset? "
        "Answer only with the integer.",
        "expenses", "numeric",
        lambda d: float(len(d["expenses"])),
    ),

    # ---------- By category (4) ----------
    Question(
        "C01", "By category",
        "Which government function had the highest amount paid in expenses? "
        "Answer only with the function name.",
        "expenses", "text",
        lambda d: _top_category(d["expenses"], "function", "paid"),
    ),
    Question(
        "C02", "By category",
        "Which top-level agency had the highest amount paid in expenses? "
        "Answer only with the agency name.",
        "expenses", "text",
        lambda d: _top_category(d["expenses"], "top_agency", "paid"),
    ),
    Question(
        "C03", "By category",
        "Which revenue source had the highest realized amount? "
        "Answer only with the source name.",
        "revenues", "text",
        lambda d: _top_category(d["revenues"], "source", "realized"),
    ),
    Question(
        "C04", "By category",
        "Which expense group concentrated the highest amount paid? "
        "Answer only with the expense group name.",
        "expenses", "text",
        lambda d: _top_category(d["expenses"], "expense_group", "paid"),
    ),

    # ---------- Trend and comparison (4) ----------
    Question(
        "E01", "Trend and comparison",
        "Considering expenses only, what was the difference between the "
        "committed amount and the paid amount in the period? "
        "Answer only with the number in reais (committed minus paid).",
        "expenses", "numeric",
        lambda d: _sum(d["expenses"], "committed") - _sum(d["expenses"], "paid"),
        "R$",
    ),
    Question(
        "E02", "Trend and comparison",
        "What was the percentage of the paid amount relative to the "
        "committed amount in expenses? Answer only with the percentage number.",
        "expenses", "numeric",
        lambda d: (
            100.0 * _sum(d["expenses"], "paid") / _sum(d["expenses"], "committed")
            if _sum(d["expenses"], "committed") else 0.0
        ),
        "%",
    ),
    Question(
        "E03", "Trend and comparison",
        "Comparing realized revenue and paid expenses, what was the result "
        "(realized revenue minus paid expenses)? Answer only with the number in reais.",
        "revenues", "numeric",
        lambda d: _sum(d["revenues"], "realized") - _sum(d["expenses"], "paid"),
        "R$",
    ),
    Question(
        "E04", "Trend and comparison",
        "What was the total forecast revenue in the period? "
        "Answer only with the number in reais.",
        "revenues", "numeric",
        lambda d: _sum(d["revenues"], "forecast"), "R$",
    ),

    # ---------- Insights and decision (4) ----------
    Question(
        "I01", "Insights and decision",
        "Based on the data, was the realized revenue enough to cover the "
        "paid expenses? Answer only 'Yes' or 'No'.",
        "revenues", "text",
        lambda d: "Yes" if _sum(d["revenues"], "realized") >= _sum(d["expenses"], "paid") else "No",
    ),
    Question(
        "I02", "Insights and decision",
        "Which revenue economic category should receive the most attention "
        "for having the highest realized amount? Answer only with the category name.",
        "revenues", "text",
        lambda d: _top_category(d["revenues"], "economic_category", "realized"),
    ),
    Question(
        "I03", "Insights and decision",
        "Among government functions, which represents the biggest opportunity "
        "for spending review for having the highest amount paid? Answer only with the function name.",
        "expenses", "text",
        lambda d: _top_category(d["expenses"], "function", "paid"),
    ),
    Question(
        "I04", "Insights and decision",
        "Was the expense execution rate (paid over committed) above 50%? "
        "Answer only 'Yes' or 'No'.",
        "expenses", "text",
        lambda d: "Yes" if (
            _sum(d["expenses"], "committed")
            and 100.0 * _sum(d["expenses"], "paid") / _sum(d["expenses"], "committed") > 50.0
        ) else "No",
    ),

    # ---------- No answer in the data (4) - hallucination test ----------
    Question(
        "S01", "No answer in the data",
        "What was the total amount spent on artificial intelligence according "
        "to this dataset? If the data cannot answer, explicitly state that the "
        "information is not available.",
        "expenses", "no_answer",
        lambda d: NO_ANSWER,
    ),
    Question(
        "S02", "No answer in the data",
        "What is the name of the highest-paid public servant recorded in this "
        "dataset? If the data cannot answer, state that the information is not available.",
        "expenses", "no_answer",
        lambda d: NO_ANSWER,
    ),
    Question(
        "S03", "No answer in the data",
        "What is the revenue projection for next year according to this dataset? "
        "If the data cannot answer, state that the information is not available.",
        "revenues", "no_answer",
        lambda d: NO_ANSWER,
    ),
    Question(
        "S04", "No answer in the data",
        "What dollar exchange rate was used in this dataset? If the data cannot "
        "answer, state that the information is not available.",
        "revenues", "no_answer",
        lambda d: NO_ANSWER,
    ),
]


def compute_truths(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute the ground truth of every question from the DataFrames."""
    return {q.id: q.truth(dfs) for q in QUESTIONS}
