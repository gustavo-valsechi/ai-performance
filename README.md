# AI Benchmark for Financial Analysis

A Python project to **test and compare the performance of three AIs** -
**OpenAI (GPT)**, **Google Gemini** and **DeepSeek** - on the task of
analyzing public financial data. It supports the decision of which AI to
integrate into the analysis chat of a Personal Finance Management platform.

The benchmark runs **20 natural-language tests**, each executed **20 times**
per AI, and measures five indicators:

| Metric                       | Meaning                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| **Accuracy (%)**             | % of correct answers against the ground truth computed from the data |
| **Hallucination rate (%)**   | % of answers that made up non-existent information                   |
| **Average quality score**    | mean of the 0-10 scores given to the answers                         |
| **Average latency (s)**      | mean response time per request                                       |
| **Average cost per request** | USD cost per call, computed from the tokens used                     |

---

## Which models and why

Model choices were reviewed against current (2026) availability and pricing:

- **OpenAI -> `gpt-4.1-mini`**: outperforms the older `gpt-4o-mini` on
  reasoning and math benchmarks, keeps a low price and offers a 1M-token
  context window - the better fit for data analysis.
- **Gemini -> `google/gemini-3.8-flash` via OpenRouter**: Google's current
  Flash model (the older `gemini-2.5-flash` and `gemini-1.5-flash` are
  retired for new users), with strong reasoning over data. It runs through
  OpenRouter rather than the direct Google API, so no Google key is needed.
- **DeepSeek -> `deepseek/deepseek-chat-v3-0324` via OpenRouter**:
  replaces the Claude provider with a strong, **low-cost** open model for
  quantitative reasoning (the former `:free` slug was retired on
  OpenRouter). OpenRouter is OpenAI-compatible, so it reuses the OpenAI
  client (only the base URL changes).

> Prices change over time. Confirm the current values on each provider's
> official pricing page before using cost in your paper's conclusions. The
> links are in `config.py` and `.env.example`.

---

## How the evaluation works (methodology)

The key to an honest benchmark is a **reliable ground truth**. Instead of
trusting hand-written answers, the project **computes the correct answer
directly from the CSVs with pandas** (deterministic and reproducible) and
compares each AI's answer against it.

The 20 questions cover the study categories:

- **Totals and periods** - sums and counts
- **By category** - rankings by function, agency, revenue source, etc.
- **Trend and comparison** - differences and percentages between values
- **Insights and decision** - Yes/No conclusions drawn from the data
- **No answer in the data** - questions whose answer **does not exist** in
  the data (exchange rate, servant names, projections). Here we measure
  **hallucination**: the correct AI admits it does not know; an AI that
  invents a value is penalized.

To let the AI answer, we send a **compact factual summary** of the data
(totals and rankings), not the whole CSV - this controls cost and keeps the
test fair. The summary never contains information about the "no answer"
questions, so making those up is, in fact, a hallucination.

---

## Step-by-step installation and usage

### 1. Install the dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure the API keys

Copy the environment template and fill in the keys:

```powershell
Copy-Item .env.example .env
```

Open `.env` and paste the keys. **You only need the keys of the providers you
want to test** - providers without a key are skipped automatically.

#### How to obtain each token (step by step)

**OpenAI (GPT)**

1. Go to <https://platform.openai.com/>
2. Sign in / create the account.
3. Add credit under **Settings -> Billing** (calls are paid).
4. Open **API keys** (<https://platform.openai.com/api-keys>).
5. **Create new secret key**, name it and copy the value (starts with `sk-...`).
   The key is shown only once.
6. Paste it into `OPENAI_API_KEY` in `.env`.

**Google Gemini** — no separate key needed. Gemini runs through OpenRouter
in this project (not the direct Google API), so the `OPENROUTER_API_KEY`
below covers it. This avoids needing Google AI Studio billing.

**OpenRouter (serves both Gemini and DeepSeek)**

1. Go to <https://openrouter.ai/>
2. Sign in with Google/GitHub.
3. Add a small credit under **Credits** (a few dollars is plenty - the full
   run across both OpenRouter models costs ~US$ 1-2).
4. Open **Keys** (<https://openrouter.ai/settings/keys>).
5. **Create Key**, name it and copy the value (starts with `sk-or-...`).
   It is shown only once.
6. Paste it into `OPENROUTER_API_KEY` in `.env`.

> The `.env` file must **never** be committed. The `.gitignore` already ignores it.

### 3. Download the Transparency Portal data

1. Go to <https://portaldatransparencia.gov.br/download-de-dados>
2. Download **Expense Execution** and **Revenue Execution** (CSV).
3. Save the files as:
   - `data/expenses.csv`
   - `data/revenues.csv`

The portal CSVs use `;` as the separator, `latin-1` encoding and a decimal
comma. The loader handles all of that automatically.

### 4. Run the benchmark

```powershell
python main.py
```

What happens:

- loads the data and computes the ground truth;
- runs the 20 questions x 20 times for each active AI;
- writes the raw results to `results/results_<timestamp>.json`;
- generates and prints the comparative report, saving it to
  `results/report_<timestamp>.txt`.

Logs go to the terminal and to the `benchmark.log` file, in a clean format
(`time | level | message`).

### 5. (Optional) Regenerate the report without spending credit

If you already have a results JSON and only want to recompute the report:

```powershell
python main.py --report results/results_<timestamp>.json
```

---

## Quick tweaks (`config.py`)

Everything that controls the benchmark lives in `config.py`, commented
line by line:

- `RUNS_PER_TEST` - number of runs per question (default **20**).
  Lower it to 1-3 while debugging to save credit.
- `TEMPERATURE` - default **0.0** (more deterministic answers).
- `MAX_OUTPUT_TOKENS` - answer token cap.
- `DELAY_BETWEEN_CALLS` - pause between calls (avoids rate limits / 429).
- `NUMERIC_TOLERANCE` - accepted margin on numeric answers (default 1%).
- `PROVIDERS` - model, **prices per 1 million tokens** and optional
  `base_url` for each AI (OpenRouter uses a custom base URL).

---

## Project structure

```
IA-performance/
├── main.py                   # entry point
├── config.py                 # central configuration (self-explanatory)
├── requirements.txt          # dependencies
├── .env.example              # API key template (step by step)
├── data/                     # Transparency Portal CSVs (you download)
├── results/                  # results JSON + reports (generated)
└── benchmark/
    ├── logging_setup.py      # readable logging (terminal + file)
    ├── data_loader.py        # CSV loading and normalization
    ├── questions.py          # the 20 questions + deterministic ground truth
    ├── context_builder.py    # factual summary sent to the AIs
    ├── ai_clients.py         # OpenAI / Gemini / DeepSeek clients
    ├── evaluator.py          # accuracy, hallucination and quality
    ├── runner.py             # orchestrates the run and saves the results
    └── report.py             # aggregates metrics and generates the report + decision
```

---

## Important notes

- **Real cost**: 20 tests x 20 runs x 3 AIs = **1,200 calls** per full round.
  With the default economical models the total cost is low (roughly US$ 1-2
  across all three providers), but confirm the prices and start with a small
  `RUNS_PER_TEST`.
- **Reliability above all**: in a financial platform, an AI that makes up
  numbers is unacceptable. That is why the _No answer in the data_ category
  weighs heavily in the final decision.
- **Reproducibility**: `temperature=0` and the deterministic ground truth
  make the result comparable across runs.

```

```
