# Reproducing Catch Alpha, step by step

This tutorial goes from "I just cloned the repository" to "I rebuilt every number in the
paper", in six levels (0 to 5). Each level is independent of the ones after it, so stop wherever
you have seen enough. Run times are for a recent laptop; `--workers N` divides the long
ones by roughly N.

| Level | What you reproduce | Needs | Time |
|---|---|---|---|
| 0 | Check all 109 headline numbers against `results/`, redraw the figures | nothing | seconds |
| 1 | The evaluator on synthetic data (Section 2, Figure 1, Table 1) | nothing | minutes to hours |
| 2 | The controlled learning test (Section 3, Figure 2) | PyTorch | seconds (collect) / ~30 min on 4 cores (retrain) |
| 3 | The real-data study (Section 4, Figures 3–4) | the hourly panel | ~10 min |
| 4 | The single final-period evaluation (Section 5) | the panel's test split | ~2 min |
| 5 | The panel itself, from public Binance data | internet access | ~1 h |

## Setup

```bash
git clone https://github.com/JorisViaudQuantAI/Catch-Alpha.git
cd Catch-Alpha
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.11 or later. The core needs only numpy, scipy, pandas and matplotlib.

## Level 0: check the numbers and redraw the figures

```bash
python scripts/verify_paper_numbers.py
```

This reads the result files in `results/` and compares them with 109 numbers printed in
the paper, including every cell of Table 1, rounded to the precision the paper prints.
Every line should say `OK` and the last line `109/109 paper numbers reproduced from
results/`. Then:

```bash
python scripts/make_figures.py      # paper Figures 1-4 + figures S1, S2 (PNG + PDF)
pytest                              # unit tests + the same 109 checks
```

`make_figures.py` draws each figure only from the CSV/JSON files in `results/`, so every
level below works the same way: rerun an experiment, it rewrites its file in `results/`,
redraw the figures. (Font rendering can make the redrawn PNGs differ by a few pixels from
the paper's.)

## Level 1: test the evaluator before testing a strategy (Section 2)

The evaluator lives in `src/catch_alpha/gate.py` and the statistics behind it in
`src/catch_alpha/stats.py`. Section 2 checks it on simulated returns where the truth is
known.

```bash
python scripts/calibrate_evaluator.py true-v               # DSR variance check, seconds
python scripts/calibrate_evaluator.py null --workers 4     # Figure 1, a few minutes
git diff --stat results/calibration                        # empty diff = exact reproduction
```

- `null` draws 200 zero-edge strategies and 6 × 40 strategies with a planted annual Sharpe
  of 0.5 to 3.0, runs the four checks on each, and writes
  `results/calibration/null_calibration.csv`. Expected: `0/200` nulls accepted and
  detection `0, 0, 12, 34, 39, 40` out of 40.
- `true-v` measures the true cross-trial Sharpe variance on the same design and compares
  it with the approximation DSR uses (4.6×10⁻⁴ vs 5.6×10⁻⁴), plus the DSR worked example
  of Section 2 (0.99 with 3 trials, 0.63 with 347).

The two longer experiments:

```bash
python scripts/calibrate_evaluator.py robustness --workers 8   # ~1-2 CPU-hours
python scripts/calibrate_evaluator.py selection --workers 8    # Table 1, ~3 CPU-hours
```

- `robustness` repeats the null test with Student-t, GARCH, AR(1) and Gaussian returns at
  1,800, 5,000 and 50,000 observations (`null_robustness_grid.csv`, 0/2,400 accepted).
- `selection` draws campaigns of 200 correlated candidates, picks the best one and
  compares six decision rules on the same winners (`selection_aware_campaigns.csv`,
  `selection_aware_calibration.csv`).

Add `--quick` to any command for a small smoke run written to
`results/calibration/quick/` (never over the paper files).

Every random draw is seeded, and the seeds and draw order are those of the original
research code, so a full run gives the published files exactly.

## Level 2: can offline RL use a known signal? (Section 3)

```bash
pip install -e ".[learning]"            # adds PyTorch; a CPU build is enough
python scripts/learning_test.py smoke   # tiny end-to-end run, ~10 s
python scripts/learning_test.py collect # report + Figure 2 data from the 90 published runs
```

`collect` reads the 90 published training runs in `results/learning/workers/` (9
configurations × 10 training seeds, each evaluated on the same 100 fresh market paths),
recomputes every per-run summary from the raw path results, aggregates them, applies the
pre-registered decision rules and rewrites `results/learning/report.json` and
`results/learning/capability_common_flat.csv` (the Figure 2 input). It prints the
paper's numbers next to the recomputed ones.

To retrain all 90 policies from scratch:

```bash
python scripts/learning_test.py run --workers 4
```

This writes into `results/learning/rerun/` (the published runs are never overwritten),
then prints the same comparison. Expect the same printed numbers and decisions as the
paper. Individual runs can differ slightly from the published ones because PyTorch
training is not bit-for-bit identical across operating systems and BLAS libraries; see
`results/learning/linux_rerun_vs_published.csv` for one full Linux rerun (79/90 runs equal
to 1e-6, the rest within 0.5%).

Where things are:

- `learning/market.py`: the planted-alpha market (5 assets, AR(1) signal with
  persistence 0.9, information coefficient 0.05, volatility 1%).
- `learning/gp.py`: the Gârleanu–Pedersen reference rule and the profit/turnover
  accounting of Eq. 3.
- `learning/learners.py`: behaviour cloning (BC), implicit Q-learning (IQL) and
  conservative Q-learning (CQL), in plain PyTorch.
- `learning/experiment.py`: the nine configurations, the offline dataset (GP actions
  plus Gaussian noise), training, evaluation, the reference-denominator check and the
  decision rules.

## Level 3: the real-data study (Section 4)

### Get the panel

The study uses an hourly panel of twelve Binance USD-margined perpetual futures (BTC, ETH,
BNB, XRP, ADA, LTC, LINK, DOGE, SOL, AVAX, DOT, ETC), September 2019 to 11 June 2026,
stored as three chronological splits:

```
data/public_v3/
  public_v3_train.npydir/     2019-09-08 to 2023-12-31   (37,807 hours)
  public_v3_valid.npydir/     2024-01-01 to 2024-12-31   ( 8,784 hours)
  public_v3_test.npydir/      2025-01-01 to 2026-06-11   (12,648 hours, the locked final period)
data/lake/binance_spot/1h/symbol=BTCUSDT/date=.../part-000.parquet   spot prices (cash-and-carry only)
```

`data/lake/` is a local parquet cache of the downloaded Binance data; `public_v3` is the
name of the panel.

Market data are not redistributed in this repository; Level 5 explains how to rebuild
them. All studies below use train + valid only (46,590 hourly returns).

### Run the study

```bash
pip install -e ".[data]"        # pyarrow, to read the spot parquet files
python scripts/real_data_signal_screen.py --panel-dir data/public_v3        # Figure 3 (left), ~15 s
python scripts/real_data_strategies.py --panel-dir data/public_v3 \
       --spot-lake data/lake/binance_spot                                    # Figure 4 + S1, ~7 min
python scripts/real_data_weekly_momentum.py --panel-dir data/public_v3      # Section 5 example, ~30 s
git diff --stat results/real_data
```

- **Signal screen.** 32 price signals × 4 horizons = 128 pairs. For each, the hourly
  cross-sectional Spearman correlation with the future return, a Newey–West interval per
  fold, and "stable" when at least 4 of 6 folds exclude zero. Expected: 69 stable pairs;
  24h momentum predicts the next hour with rank IC −0.032 in 6/6 folds (reversal).
- **Strategies.** The four frozen portfolios (raw funding carry, beta-hedged carry, 24h
  long-loser/short-winner reversal, delta-neutral cash-and-carry), each with a 5 bp
  per-leg cost, evaluated on six chronological folds separated by 168-hour gaps, then
  passed to the evaluator against cash, buy-and-hold and their declared baselines.
  Expected: every verdict is `INSUFFICIENT_DATA` (46,590 < 54,160 hours); cash-and-carry
  receives +0.491 in funding but nets −0.263 after −0.732 in fees; break-even fee 3.2 bp.
- **Weekly momentum** is the Section 5 illustration of why abstaining matters (276 weekly
  returns < 323 required; naive mean +0.56% per week) and the market-derived null of
  Section 2 (0/200 accepted).

Where things are: `realdata/panel.py` (loading, returns, folds), `realdata/strategies.py`
(the four portfolios, benchmarks and baselines), `realdata/signals.py` (the screen),
`realdata/study.py` (fold reports, fee sweep, evaluator verdicts, PBO, final period).

The **trial ledger** of the real-data panel is in
`results/real_data/trial_ledger_public_v3.json`: 353 records, one per evaluation ever run
on this panel, including signal screens. The 352 selection (train/valid) records are the
`n_trials` the deflated Sharpe ratio uses; the single final-period run is not counted,
because it tests an already frozen claim. `src/catch_alpha/ledger.py` is the same append-only ledger for your own work.

## Level 4: the final period (Section 5)

```bash
python scripts/real_data_final_period.py --panel-dir data/public_v3 --spot-lake data/lake/binance_spot
```

This is the only script that opens the test split. It rebuilds the four frozen portfolios
over the whole panel, keeps the 12,648 test hours and writes `final_period.json`.
Expected: no portfolio's bootstrap interval lies above zero; cash-and-carry is entirely
below zero (−2.88×10⁻⁵ per hour); the sample is too short for the evaluator to rule.

In the paper this was run exactly once, after every choice was frozen. If you are
extending this work, treat the test split the same way: record each access in a ledger
and decide in advance how many you allow.

## Level 5: rebuild the panel from public Binance data

```bash
pip install -e ".[data]"         # pyarrow, for the parquet files
python scripts/real_data_build_panel.py --lake-dir data/lake --out-dir data/public_v3 --spot
```

This downloads hourly perpetual-futures candles and funding history from the public
Binance API (and spot candles with `--spot`) into a parquet lake, then assembles the
panel. It takes about an hour with polite request throttling and needs 2–3 GB.

Caveat: this downloader could not be run end to end while preparing the release (the
build machine had no access to Binance). The features computed from closing prices were
checked against the published panel; features that need highs, lows and volumes were
not. If your rebuilt panel differs, `git diff results/` after Level 3 will show where.
Binance also revises old data occasionally, so a rebuild months later may not be
bit-identical.

## Using the evaluator on your own strategy

See the [README](../README.md#using-the-evaluator-on-your-own-strategy). The short
version: record every evaluation in a `TrialLedger`, pass `ledger.n_trials(...)` to
`evaluate`, and treat `INSUFFICIENT_DATA` as an answer.
