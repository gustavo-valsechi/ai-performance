"""
Builds the data context sent to the AIs.

PROBLEM: the Transparency Portal CSVs can have hundreds of thousands of
rows - they do not fit in an AI prompt, and sending them would be very
expensive. SOLUTION: we build a compact, factual SUMMARY of the data
(totals and aggregations) and hand that summary over as context.

Important for the benchmark's honesty: the summary contains ALL the
information needed to answer the questions that DO have an answer, and
NONE about the "no answer" questions (exchange rate, servant names,
projections). So if an AI answers a "no answer" question, it is because
it hallucinated - not because we hinted at it.
"""

from __future__ import annotations

import pandas as pd


def _fmt(value: float) -> str:
    """Format a number in the Brazilian style for readability."""
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _ranking(df: pd.DataFrame, group: str, value: str, n: int = 10) -> str:
    if group not in df.columns or value not in df.columns:
        return "  (data unavailable)"
    agg = df.groupby(group)[value].sum().sort_values(ascending=False).head(n)
    lines = [f"  - {idx}: R$ {_fmt(v)}" for idx, v in agg.items()]
    return "\n".join(lines) if lines else "  (no records)"


def build_context(dfs: dict[str, pd.DataFrame]) -> str:
    """Build the factual context text from the DataFrames."""
    exp = dfs["expenses"]
    rev = dfs["revenues"]

    parts: list[str] = []
    parts.append("=== BUDGET EXECUTION DATA SUMMARY ===\n")

    # --- Expenses ---
    parts.append("## EXPENSE EXECUTION")
    parts.append(f"Total expense records: {len(exp)}")
    parts.append(f"Total committed: R$ {_fmt(exp.get('committed', pd.Series(dtype=float)).sum())}")
    parts.append(f"Total settled: R$ {_fmt(exp.get('settled', pd.Series(dtype=float)).sum())}")
    parts.append(f"Total paid: R$ {_fmt(exp.get('paid', pd.Series(dtype=float)).sum())}")
    parts.append("\nAmount paid by government function (top 10):")
    parts.append(_ranking(exp, "function", "paid"))
    parts.append("\nAmount paid by top-level agency (top 10):")
    parts.append(_ranking(exp, "top_agency", "paid"))
    parts.append("\nAmount paid by expense group (top 10):")
    parts.append(_ranking(exp, "expense_group", "paid"))

    # --- Revenues ---
    parts.append("\n## REVENUE EXECUTION")
    parts.append(f"Total revenue records: {len(rev)}")
    parts.append(f"Total updated forecast: R$ {_fmt(rev.get('forecast', pd.Series(dtype=float)).sum())}")
    parts.append(f"Total posted: R$ {_fmt(rev.get('posted', pd.Series(dtype=float)).sum())}")
    parts.append(f"Total realized: R$ {_fmt(rev.get('realized', pd.Series(dtype=float)).sum())}")
    parts.append("\nRealized amount by revenue source (top 10):")
    parts.append(_ranking(rev, "source", "realized"))
    parts.append("\nRealized amount by economic category (top 10):")
    parts.append(_ranking(rev, "economic_category", "realized"))

    return "\n".join(parts)
