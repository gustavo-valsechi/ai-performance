"""
Benchmark runner.

Orchestrates the full run:
  1. Loads the data (expenses and revenues).
  2. Computes the ground truth of every question.
  3. Builds the factual context sent to the AIs.
  4. For each active provider, runs the 20 questions x N runs.
  5. Evaluates every answer and writes everything to a raw-results JSON.

The raw results are later consumed by the report module.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import config
from benchmark import ai_clients, context_builder, data_loader, evaluator
from benchmark.logging_setup import log
from benchmark.questions import QUESTIONS, compute_truths


def _prepare_data() -> tuple[dict, dict, str]:
    """Load CSVs, compute ground truths and build the context."""
    dfs = {
        "expenses": data_loader.load_expenses(),
        "revenues": data_loader.load_revenues(),
    }
    truths = compute_truths(dfs)
    context_text = context_builder.build_context(dfs)
    log.info("Context built (%d characters).", len(context_text))
    return dfs, truths, context_text


def _run_provider(name: str, truths: dict, context_text: str) -> list[dict]:
    """Run the 20 questions x N runs for one provider."""
    log.info("=" * 60)
    log.info("PROVIDER: %s (model: %s)", name, config.PROVIDERS[name]["model"])
    log.info("=" * 60)

    try:
        client = ai_clients.create_client(name)
    except Exception as exc:
        log.error("Failed to initialize %s: %s. Skipping provider.", name, exc)
        return []

    records: list[dict] = []
    total = len(QUESTIONS) * config.RUNS_PER_TEST
    done = 0

    for question in QUESTIONS:
        truth = truths[question.id]
        for run in range(1, config.RUNS_PER_TEST + 1):
            resp = client.ask(context_text, question.text)
            done += 1

            if resp.error:
                ev = evaluator.Evaluation(False, False, 0.0)
            else:
                ev = evaluator.evaluate(question, truth, resp.text)

            records.append({
                "provider": name,
                "question_id": question.id,
                "category": question.category,
                "kind": question.kind,
                "run": run,
                "answer": resp.text,
                "error": resp.error,
                "correct": ev.correct,
                "hallucinated": ev.hallucinated,
                "quality": ev.quality,
                "latency_s": round(resp.latency_s, 3),
                "cost_usd": resp.cost_usd,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            })

            flag = "OK " if ev.correct else ("HAL" if ev.hallucinated else "ERR")
            log.info(
                "[%s] %s run %2d/%d  [%s]  %4.2fs  correct=%s quality=%.1f  (%d/%d)",
                name, question.id, run, config.RUNS_PER_TEST,
                flag, resp.latency_s, ev.correct, ev.quality, done, total,
            )
            time.sleep(config.DELAY_BETWEEN_CALLS)

    return records


def run(only: list[str] | None = None) -> dict:
    """
    Run the benchmark and write the raw results to JSON.

    only: optional list of provider names to run (e.g. ["Gemini", "DeepSeek"]).
          When None, runs every provider that has an API key in .env.
          Useful to collect providers at different times - for instance
          when one provider is temporarily rate-limited.
    """
    start = datetime.now()
    log.info("Benchmark start: %s", start.strftime("%Y-%m-%d %H:%M:%S"))

    active = config.active_providers()
    if only:
        requested = [p.strip() for p in only]
        unknown = [p for p in requested if p not in config.PROVIDERS]
        if unknown:
            log.error("Unknown provider(s): %s. Valid: %s",
                      ", ".join(unknown), ", ".join(config.PROVIDERS))
            raise SystemExit(1)
        missing_key = [p for p in requested if p not in active]
        if missing_key:
            log.error("No API key configured for: %s (fill in .env).",
                      ", ".join(missing_key))
            raise SystemExit(1)
        active = requested

    if not active:
        log.error(
            "No provider has an API key configured. "
            "Fill in the .env file (see .env.example)."
        )
        raise SystemExit(1)
    log.info("Providers to run: %s", ", ".join(active))

    _, truths, context_text = _prepare_data()

    all_records: list[dict] = []
    for name in active:
        all_records.extend(_run_provider(name, truths, context_text))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = start.strftime("%Y%m%d_%H%M%S")
    json_path = config.RESULTS_DIR / f"results_{stamp}.json"

    package = {
        "start": start.isoformat(),
        "end": datetime.now().isoformat(),
        "runs_per_test": config.RUNS_PER_TEST,
        "providers": active,
        "records": all_records,
    }
    json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Raw results saved to: %s", json_path)

    duration = (datetime.now() - start).total_seconds()
    log.info("Benchmark finished in %.1f s (%d answers).", duration, len(all_records))
    return package
