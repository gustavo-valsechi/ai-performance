"""
=====================================================================
 CENTRAL BENCHMARK CONFIGURATION
=====================================================================

This is the only file you need to adjust to control the benchmark's
behavior. Every option is commented to explain what it does and the
impact of changing it.

API KEYS do NOT live here - they live in the .env file (see
.env.example for the step-by-step on how to obtain each token).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the variables from the .env file into os.environ.
load_dotenv()


# ---------------------------------------------------------------------
# 1. PROJECT FOLDERS
# ---------------------------------------------------------------------
ROOT = Path(__file__).parent

# Where the CSVs downloaded from the Transparency Portal live.
DATA_DIR = ROOT / "data"

# Where reports and raw results are written.
RESULTS_DIR = ROOT / "results"

# Human-readable log file (also mirrored to the terminal).
LOG_FILE = ROOT / "benchmark.log"


# ---------------------------------------------------------------------
# 2. INPUT DATA FILES
# ---------------------------------------------------------------------
# Put the CSVs exported from the Transparency Portal in the "data/"
# folder. Download them at:
#   https://portaldatransparencia.gov.br/download-de-dados
#   - "Execucao da Despesa" (expense execution)  -> save as data/expenses.csv
#   - "Execucao da Receita" (revenue execution)   -> save as data/revenues.csv
#
# The portal CSVs use a Brazilian decimal comma (e.g. "1.234,56"), but
# the Expense and Revenue exports are NOT consistent with each other:
# they can come with different separators (";" or ",") and different
# encodings (latin-1 or utf-8). The loader AUTO-DETECTS the encoding and
# separator for each file, so you do not need to configure them here.
EXPENSES_FILE = DATA_DIR / "expenses.csv"
REVENUES_FILE = DATA_DIR / "revenues.csv"


# ---------------------------------------------------------------------
# 3. BENCHMARK PARAMETERS
# ---------------------------------------------------------------------
# How many times each of the 20 questions runs per provider.
# The study asks for 20 runs per test. Lower it to 1-3 while debugging
# so you don't burn credits.
RUNS_PER_TEST = 20

# Model temperature. 0 = as deterministic as possible, ideal for a
# reproducible benchmark. Do not change unless you want to measure
# variability on purpose.
TEMPERATURE = 0.0

# Output token cap per answer. The expected answers are short (numbers
# and sentences), so a low cap keeps the cost under control.
MAX_OUTPUT_TOKENS = 800

# Pause (in seconds) between calls to the same provider, to respect the
# APIs' rate limits. Increase it if you hit a 429 error.
#
# Set to 6.0 to respect the Gemini free tier limit of ~10 requests per
# minute (gemini-3.8-flash): 60s / 10 = one call every 6s. This is the
# tightest free-tier RPM among the configured providers, so it keeps the
# whole run within the free limits. If you only test paid/faster tiers,
# you can lower it back toward 0.5 to finish quicker.
DELAY_BETWEEN_CALLS = 6.0

# Relative tolerance for considering a numeric value "correct".
# 0.01 = the answered number may differ by up to 1% from the ground
# truth. This absorbs rounding without rewarding wrong answers.
NUMERIC_TOLERANCE = 0.01

# Automatic retry for TRANSIENT errors (HTTP 503 "model overloaded",
# 429 rate limit, timeouts, connection drops). These are expected during
# a long run and should not be counted as failed answers.
# MAX_RETRIES = how many extra attempts after the first one.
# RETRY_BACKOFF_S = base wait; the wait grows as backoff * 2**attempt
# (e.g. 2s, 4s, 8s...) so the provider has time to recover.
MAX_RETRIES = 4
RETRY_BACKOFF_S = 2.0


# ---------------------------------------------------------------------
# 4. PROVIDERS, MODELS AND PRICES
# ---------------------------------------------------------------------
# "model" ........ the model id in the provider's API.
# "input_price" .. price in USD per 1 MILLION input tokens.
# "output_price" . price in USD per 1 MILLION output tokens.
# "base_url" ..... optional custom endpoint (used by OpenAI-compatible
#                  providers such as OpenRouter). None = provider default.
#
# IMPORTANT: prices change over time. Confirm the current values on the
# official pricing pages before using cost in real decisions:
#   OpenAI     -> https://openai.com/api/pricing/
#   Gemini     -> https://ai.google.dev/gemini-api/docs/pricing
#   OpenRouter -> https://openrouter.ai/models
#
# KEEP THESE PRICES REAL (do not zero them out):
#   The "Average cost per request" metric in the report is computed from
#   these values. OpenAI (gpt-4.1-mini) has NO free tier - it is paid per
#   token - so setting its price to 0 would make the report show a FALSE
#   zero cost. For a trustworthy study (e.g. a PFC paper), the cost
#   column must reflect what each call actually costs.
#   All three providers are now paid per token, so keep every price real.
#   Gemini also runs under a free DAILY QUOTA, but it is still priced per
#   token above that quota, so its real per-token price is kept for an
#   apples-to-apples comparison.
#
# Cost / quota notes (approximate, 2026) - a full 20x20 run is 400
# requests per provider, so plan around these:
#   OpenAI gpt-4.1-mini .... no free tier (prepay credits; ~US$ 5 covers
#                            the whole run, which costs < US$ 1).
#   Gemini 3.8-flash ....... ~250 requests/day free quota -> run spans
#                            ~2 days, or pay per token to run in one go.
#   DeepSeek (OpenRouter) .. paid, but very cheap (~US$ 0.20 for the full
#                            run); add a small OpenRouter credit to run it.
#
# To TURN OFF a provider, just leave its API key blank in .env - it is
# skipped automatically.
#
# MODEL CHOICES (why these models):
#   - OpenAI "gpt-4.1-mini": outperforms gpt-4o-mini on reasoning/math
#     benchmarks, has a 1M-token context window and is still cheap -
#     the better fit for data analysis.
#   - Gemini "gemini-3.8-flash": Google's current Flash model (replaces
#     the retired 2.5-flash), with strong reasoning over data.
#   - DeepSeek via OpenRouter "deepseek/deepseek-chat-v3-0324":
#     a strong, low-cost open model for quantitative reasoning. OpenRouter
#     is OpenAI-compatible, so it reuses the OpenAI client (just a
#     different base_url). Replaces the Claude provider.
PROVIDERS = {
    "OpenAI": {
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "input_price": 0.40,   # USD / 1M tokens
        "output_price": 1.60,  # USD / 1M tokens
        "base_url": None,
    },
    "Gemini": {
        # Served through OpenRouter (same gateway as DeepSeek) because the
        # Google AI Studio billing could not be set up. OpenRouter is
        # OpenAI-compatible, so Gemini reuses the OpenAI client via base_url
        # and the OpenRouter key - no google-genai SDK or Google key needed.
        "model": "google/gemini-3.8-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_price": 0.75,   # USD / 1M tokens (intro price through 2026-12-31)
        "output_price": 3.75,  # USD / 1M tokens (intro price through 2026-12-31)
        "base_url": "https://openrouter.ai/api/v1",
    },
    "DeepSeek": {
        # The ":free" DeepSeek slug was retired on OpenRouter; use the
        # paid (and very cheap) V3 slug. ~400 requests cost well under
        # US$ 0.20 total at these prices.
        "model": "deepseek/deepseek-chat-v3-0324",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_price": 0.29,   # USD / 1M tokens
        "output_price": 1.14,  # USD / 1M tokens
        "base_url": "https://openrouter.ai/api/v1",
    },
}


def api_key(provider_name: str) -> str | None:
    """Return the provider's API key read from .env, or None if blank."""
    env = PROVIDERS[provider_name]["api_key_env"]
    value = os.environ.get(env, "").strip()
    return value or None


def active_providers() -> list[str]:
    """List only the providers that have an API key configured."""
    return [name for name in PROVIDERS if api_key(name)]
