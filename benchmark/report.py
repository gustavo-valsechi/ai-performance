"""
Metric aggregation and comparative report generation.

From the raw records produced by the runner, it computes the study's
five metrics per provider (and per category), points out the best AI
per criterion and issues a final decision with justification.

Metrics:
  - Accuracy (%) ............. % of correct answers
  - Hallucination rate (%) ... % of answers that made up information
  - Average quality score .... mean of the 0-10 scores
  - Average latency (s) ...... mean time per request
  - Average cost per request . mean cost_usd per call
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import config
from benchmark.logging_setup import log


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _metrics(records: list[dict]) -> dict:
    """Compute the 5 metrics for a set of records."""
    n = len(records)
    if n == 0:
        return {"n": 0, "accuracy": 0, "hallucination": 0,
                "quality": 0, "latency": 0, "cost": 0}
    return {
        "n": n,
        "accuracy": 100.0 * sum(r["correct"] for r in records) / n,
        "hallucination": 100.0 * sum(r["hallucinated"] for r in records) / n,
        "quality": _mean([r["quality"] for r in records]),
        "latency": _mean([r["latency_s"] for r in records]),
        "cost": _mean([r["cost_usd"] for r in records]),
    }


def aggregate(package: dict) -> dict:
    """Aggregate metrics per provider and per category."""
    records = package["records"]

    by_provider: dict[str, list[dict]] = defaultdict(list)
    by_prov_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_provider[r["provider"]].append(r)
        by_prov_cat[(r["provider"], r["category"])].append(r)

    summary = {prov: _metrics(recs) for prov, recs in by_provider.items()}

    categories = sorted({r["category"] for r in records})
    by_category = {
        prov: {cat: _metrics(by_prov_cat[(prov, cat)]) for cat in categories}
        for prov in by_provider
    }
    return {"summary": summary, "by_category": by_category, "categories": categories}


def _best(summary: dict, metric: str, higher_is_better: bool) -> str:
    """Name of the provider with the best value for a metric."""
    items = summary.items()
    key = lambda kv: kv[1][metric]
    chosen = (max if higher_is_better else min)(items, key=key)
    return chosen[0]


def _score_decision(summary: dict) -> dict[str, float]:
    """
    Weighted score for the final decision (0-100).
    Weights: accuracy 35%, hallucination 25%, quality 20%, latency 10%, cost 10%.
    Hallucination and cost/latency are "lower is better" (inverted).
    """
    if not summary:
        return {}
    provs = list(summary)

    def norm(metric, higher_is_better):
        vals = {p: summary[p][metric] for p in provs}
        lo, hi = min(vals.values()), max(vals.values())
        span = (hi - lo) or 1.0
        return {p: ((v - lo) / span if higher_is_better else (hi - v) / span)
                for p, v in vals.items()}

    acc = norm("accuracy", True)
    hal = norm("hallucination", False)
    qua = norm("quality", True)
    lat = norm("latency", False)
    cos = norm("cost", False)

    return {
        p: 100.0 * (0.35 * acc[p] + 0.25 * hal[p] + 0.20 * qua[p]
                    + 0.10 * lat[p] + 0.10 * cos[p])
        for p in provs
    }


def generate_report(package: dict) -> str:
    """Build the full comparative text report."""
    agg = aggregate(package)
    summary = agg["summary"]
    L: list[str] = []

    L.append("=" * 70)
    L.append("AI PERFORMANCE REPORT - FINANCIAL ANALYSIS")
    L.append("=" * 70)
    L.append(f"Run window: {package['start']}  ->  {package['end']}")
    L.append(f"Runs per test: {package['runs_per_test']}")
    L.append(f"Providers tested: {', '.join(package['providers'])}")
    L.append("")

    # --- Overall table ---
    L.append("## OVERALL METRICS (per provider)")
    L.append("")
    header = f"{'Provider':<10} {'Accuracy':>10} {'Hallucin.':>11} {'Quality':>10} {'Latency':>10} {'Cost/req':>14}"
    L.append(header)
    L.append("-" * len(header))
    for prov, m in summary.items():
        L.append(
            f"{prov:<10} {m['accuracy']:>9.1f}% {m['hallucination']:>10.1f}% "
            f"{m['quality']:>10.2f} {m['latency']:>9.2f}s "
            f"US$ {m['cost']:>10.6f}"
        )
    L.append("")

    # --- Per category ---
    L.append("## ACCURACY BY CATEGORY (%)")
    L.append("")
    cats = agg["categories"]
    header2 = f"{'Category':<24}" + "".join(f"{p:>12}" for p in summary)
    L.append(header2)
    L.append("-" * len(header2))
    for cat in cats:
        line = f"{cat:<24}"
        for prov in summary:
            line += f"{agg['by_category'][prov][cat]['accuracy']:>11.1f}%"
        L.append(line)
    L.append("")

    # --- Hallucination rate by category (focus on "No answer in the data") ---
    L.append("## HALLUCINATION RATE BY CATEGORY (%)")
    L.append("")
    L.append(header2)
    L.append("-" * len(header2))
    for cat in cats:
        line = f"{cat:<24}"
        for prov in summary:
            line += f"{agg['by_category'][prov][cat]['hallucination']:>11.1f}%"
        L.append(line)
    L.append("")

    # --- Best per criterion ---
    L.append("## BEST PER CRITERION")
    L.append("")
    L.append(f"  Highest accuracy ........ {_best(summary, 'accuracy', True)}")
    L.append(f"  Lowest hallucination .... {_best(summary, 'hallucination', False)}")
    L.append(f"  Best quality ............ {_best(summary, 'quality', True)}")
    L.append(f"  Lowest latency .......... {_best(summary, 'latency', False)}")
    L.append(f"  Lowest cost ............. {_best(summary, 'cost', False)}")
    L.append("")

    # --- Final decision ---
    scores = _score_decision(summary)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    L.append("## INSIGHTS AND DECISION")
    L.append("")
    L.append("Weighted score (accuracy 35%, hallucination 25%, quality 20%,")
    L.append("latency 10%, cost 10%) - higher is better:")
    L.append("")
    for prov, pt in ranking:
        L.append(f"  {prov:<10} {pt:6.1f} / 100")
    L.append("")
    if ranking:
        winner = ranking[0][0]
        mw = summary[winner]
        L.append(f"RECOMMENDED DECISION: {winner}")
        L.append(
            f"  Rationale: best weighted score, with accuracy of "
            f"{mw['accuracy']:.1f}%, hallucination of {mw['hallucination']:.1f}%, "
            f"average quality {mw['quality']:.2f}, latency {mw['latency']:.2f}s "
            f"and average cost of US$ {mw['cost']:.6f} per request."
        )
    L.append("")
    L.append("Note: the 'No answer in the data' category is the main")
    L.append("reliability indicator - high hallucination there disqualifies an")
    L.append("AI for financial analysis, where making up numbers is unacceptable.")
    L.append("=" * 70)

    return "\n".join(L)


def save_report(package: dict) -> Path:
    """Generate the report, print it to the log and save it to a .txt file."""
    text = generate_report(package)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = package["start"].replace(":", "").replace("-", "").replace("T", "_")[:15]
    path = config.RESULTS_DIR / f"report_{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    log.info("Report saved to: %s", path)
    print("\n" + text + "\n")
    return path


def load_package(path: str | Path) -> dict:
    """Load an existing raw-results JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_and_merge(paths: list[str | Path]) -> dict:
    """
    Load one or more raw-results JSONs and merge them into one package.

    Used when providers were collected at different times (separate runs
    or separate days). Records are concatenated; the earliest start and
    latest end are kept; the provider list is the union across files.

    If the same provider appears in more than one file, all its records
    are kept - so avoid passing two files that both contain the same
    provider unless you intend to pool those runs together.
    """
    packages = [load_package(p) for p in paths]
    if len(packages) == 1:
        return packages[0]

    all_records: list[dict] = []
    providers: list[str] = []
    for pkg in packages:
        all_records.extend(pkg.get("records", []))
        for prov in pkg.get("providers", []):
            if prov not in providers:
                providers.append(prov)

    starts = [p["start"] for p in packages if p.get("start")]
    ends = [p["end"] for p in packages if p.get("end")]
    return {
        "start": min(starts) if starts else "",
        "end": max(ends) if ends else "",
        "runs_per_test": packages[0].get("runs_per_test"),
        "providers": providers,
        "records": all_records,
    }
