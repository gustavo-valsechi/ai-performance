"""
Entry point for the AI benchmark for financial analysis.

Usage:
    python main.py                              # run every configured provider
    python main.py --providers Gemini,DeepSeek  # run only the listed providers
    python main.py --report FILE [FILE ...]     # rebuild the report from one or
                                                # more existing results JSONs

Running providers separately is handy when one of them is temporarily
rate-limited (e.g. a free-tier daily cap): collect each provider when it
is available, then combine the JSONs into a single report with --report.

Before running:
    1. pip install -r requirements.txt
    2. copy .env.example to .env and fill in the API keys
    3. place the CSVs at data/expenses.csv and data/revenues.csv
"""

import argparse

from benchmark import report, runner
from benchmark.logging_setup import log


def main() -> None:
    parser = argparse.ArgumentParser(description="AI benchmark for financial analysis.")
    parser.add_argument(
        "--providers",
        metavar="LIST",
        help="Comma-separated providers to run (e.g. 'Gemini,DeepSeek'). "
             "Default: all providers with a key in .env.",
    )
    parser.add_argument(
        "--report",
        metavar="JSON_FILE",
        nargs="+",
        help="Rebuild the report from one or more existing results JSONs, "
             "without calling the AIs. Multiple files are merged.",
    )
    args = parser.parse_args()

    if args.report:
        log.info("Rebuilding report from: %s", ", ".join(args.report))
        package = report.load_and_merge(args.report)
    else:
        only = args.providers.split(",") if args.providers else None
        package = runner.run(only=only)

    report.save_report(package)


if __name__ == "__main__":
    main()
