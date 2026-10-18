# Paper ↔ code map

Where each claim of the paper comes from: the code that computes it, the command that
reruns it and the file in `results/` that holds the number. Section numbers follow the
[ICAIF '26 paper](../paper/catch_alpha_icaif26.pdf).

## Section 2: Evaluation protocol

| Paper element | Code | Command | Result file |
|---|---|---|---|
| Eq. 1, minimum length (54,160 hourly / 1,559 daily) | `stats.minimum_required_observations`, `stats.power_gate` | – | – |
| Stationary-bootstrap intervals (400 resamples; 150 in the calibration sweeps) | `stats.stationary_bootstrap_ci`, `gate.excess_check` | – | – |
| Eq. 2, deflated Sharpe ratio | `stats.deflated_sharpe_ratio`, `gate.dsr_check` | – | – |
| Pairwise SPA, worst p ≤ 0.10 | `stats.spa_pvalue`, `gate.spa_check` | – | – |
| PBO over C(6,3) = 20 splits | `stats.probability_of_backtest_overfitting`, `gate.pbo_check` | – | – |
| The full decision rule | `gate.evaluate` | – | – |
| Append-only trial ledger, 3 final-period accesses | `ledger.TrialLedger` | – (copied from the original study) | `real_data/trial_ledger_public_v3.json` (353 records; N = 352 selection records) |
| Ledger composition (344 screens + 9 portfolio evaluations) | – | – | `real_data/real_data_trial_breakdown.csv` |
| V measured 4.6×10⁻⁴ vs approximation 5.6×10⁻⁴ | `calibration.true_v_check` | `calibrate_evaluator.py true-v` | `calibration/true_v_dsr_check.json` |
| DSR 0.63 (N = 347) vs 0.99 (N = 3) | `calibration.dsr_worked_example` | `calibrate_evaluator.py true-v` | `calibration/true_v_dsr_check.json` |
| Figure 1 left: 0/200 Gaussian nulls, Wilson bound 1.9% | `calibration.null_calibration` | `calibrate_evaluator.py null` | `calibration/null_calibration.csv` |
| Figure 1 right: 0/0/30/85/97.5/100% detection | `calibration.null_calibration` | `calibrate_evaluator.py null` | `calibration/null_calibration.csv` |
| Student-t, GARCH, AR(1) nulls at 1,800/5,000/50,000: 0/2,400 | `calibration.robustness_grid` | `calibrate_evaluator.py robustness` | `calibration/null_robustness_grid.csv` |
| Market-derived null (signal shifted ≥ 26 weeks): 0/200; 276 weekly obs vs 144 required | `realdata.study.market_null_check` | `real_data_weekly_momentum.py` | `real_data/market_null_check.json` |
| Table 1, selection among 200 correlated candidates | `calibration.selection_experiment` | `calibrate_evaluator.py selection` | `calibration/selection_aware_calibration.csv`, `calibration/selection_aware_campaigns.csv` |

## Section 3: Can the learner use a known signal?

| Paper element | Code | Command | Result file |
|---|---|---|---|
| Planted-alpha market (IC 0.05, φ = 0.9, vol 1%) | `learning.market.simulate_market` | – | – |
| Eq. 3 score (profit minus quadratic cost), turnover | `learning.gp.evaluate_weight_path` | – | – |
| Gârleanu–Pedersen reference, Eq. 4 normalization | `learning.gp.gp_weights` | – | – |
| BC, IQL, CQL | `learning.learners` | – | – |
| 9 configurations × 10 seeds × 100 evaluation paths, flat start | `learning.experiment.run_worker`, `ARMS` | `learning_test.py run` | `learning/workers/*.json` |
| Reference mean +0.0461 and its lower bound +0.0395 | `learning.experiment.assess_denominator` | `learning_test.py collect` | `learning/report.json`, `learning/capability_common_flat.csv` |
| Figure 2: 0.900 / 0.731 / 0.335; −0.995 / −4.036; 10/10 losing; 6.03× / 11.97× | `learning.experiment.collect`, `decide` | `learning_test.py collect` | `learning/capability_common_flat.csv` |
| BC + previous position: −0.960, 5.92× | same | same | same |
| Cross-platform rerun of the 90 fits | – | `learning_test.py run` | `learning/linux_rerun_vs_published.csv` |

## Section 4: Real-data study

| Paper element | Code | Command | Result file |
|---|---|---|---|
| Panel: 12 perpetuals, hourly, train/valid/test splits | `realdata.panel.load_panel`; built by `realdata.build_panel` | `real_data_build_panel.py` | `data/public_v3/` (not committed) |
| Six folds with 168h gaps | `realdata.panel.strategy_folds`, `signal_folds` | – | – |
| 46,590 < 54,160: `INSUFFICIENT_DATA` | `stats.power_gate` via `realdata.study` | `real_data_strategies.py` | `real_data/strategy_folds.json`, `real_data/gate_verdicts.json` |
| 128 price pairs, 69 stable; 24h momentum IC −0.032 (6/6, t = −7.79); inverse vol +0.046 at 8h | `realdata.signals.run_signal_screen` | `real_data_signal_screen.py` | `real_data/signal_screen.csv`, `signal_screen_summary.json` |
| Figure 3 left | `realdata.signals.figure_rows` | `real_data_signal_screen.py` | `real_data/signal_atlas_price.csv` |
| Figure 3 right, external and on-chain screen (216 pairs) | not included (third-party data) | – | `real_data/signal_atlas_exog.csv` (published values) |
| Reversal tail −1.45 / −3.02 gross, mirror trend +1.45 / +3.02 | `realdata.strategies.reversal` | `real_data_strategies.py` | `real_data/strategy_folds.json` |
| Raw carry: no fold with positive net lower bound; funding positive 6/6 | `realdata.strategies.raw_carry`, `study.fold_report` | `real_data_strategies.py` | `real_data/strategy_folds.json` |
| Cash-and-carry +0.491 / −0.023 / −0.732 = −0.263 (figure S1) | `realdata.strategies.cash_and_carry`, `study.carry_decomposition` | `real_data_strategies.py` | `real_data/carry_decomposition.csv` |
| Fee sweep +0.468 at 0 bp to −0.995 at 10 bp, break-even 3.2 bp (Figure 4) | `realdata.study.fee_sensitivity` | `real_data_strategies.py` | `real_data/cash_and_carry_fee_sensitivity.csv`, `fee_sensitivity.json` |
| Beta-hedged carry: funding share 0.443 < 0.6 | `realdata.strategies.hedged_carry`, `funding_share` | `real_data_strategies.py` | `real_data/strategy_folds.json` |
| On-chain leads after market-cap removal: hash rate 24h 0.049, active addresses 72h 0.061 (4/6 folds each) | not included (third-party data) | – | `real_data/onchain_mcap_residual.csv` (published values) |
| Figure S2 (supplementary), required length vs target Sharpe | `stats.minimum_required_observations` | `power_gate_table.py` | `real_data/power_gate.csv` |

## Section 5: What the evidence supports

| Paper element | Code | Command | Result file |
|---|---|---|---|
| PBO = 0.00 over 20 splits; winner −1.98 | `realdata.study.pbo_reversal` | `real_data_strategies.py` | `real_data/pbo_reversal.json` |
| Weekly momentum: 276 < 323, +0.56% per week, lower bound 6×10⁻⁵ | `realdata.study.weekly_momentum_benchmark` | `real_data_weekly_momentum.py` | `real_data/weekly_momentum.json` |
| Final period: 12,648 hours, no interval above zero, cash-and-carry −2.88×10⁻⁵ | `realdata.study.final_period` | `real_data_final_period.py` | `real_data/final_period.json` |
| Table 2, evidence summary | – (summarizes the rows above) | `verify_paper_numbers.py` | all of the above |

## Figures

All six figures are drawn by `scripts/make_figures.py` from the files above. F1–F4 are the
paper's Figures 1–4. S1 and S2 are supplementary: they illustrate numbers stated in the
paper's text (the cash-and-carry decomposition of Section 4 and Eq. 1).

| Figure | Input |
|---|---|
| F1 null calibration | `calibration/null_calibration.csv` |
| F2 learning test | `learning/capability_common_flat.csv` |
| F3 signal screen | `real_data/signal_atlas_price.csv`, `real_data/signal_atlas_exog.csv` |
| F4 fee sensitivity | `real_data/cash_and_carry_fee_sensitivity.csv` |
| S1 carry decomposition (supplementary) | `real_data/carry_decomposition.csv` |
| S2 minimum length (supplementary) | `real_data/power_gate.csv` |
