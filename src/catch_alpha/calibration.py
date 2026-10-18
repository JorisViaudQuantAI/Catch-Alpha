"""Test the evaluator on data where the truth is known (Section 2.3 of the paper).

Before trusting any verdict on market data, we measure how the evaluator behaves
when we *know* the answer:

* :func:`null_calibration` -- 200 zero-edge strategies: how many does it accept?
  (paper: 0/200, Wilson upper bound 1.9%). Same function with a planted drift
  gives the detection curve (0/0/30/85/97.5/100% at annualised Sharpe 0.5..3.0).
* :func:`robustness_grid` -- the same zero-edge test with heavy tails (Student-t3),
  volatility clustering (GARCH(1,1)) and serial correlation (AR(1)), at 1,800,
  5,000 and 50,000 observations (paper: 0/2,400).
* :func:`selection_campaign` -- pick the best of 200 correlated candidates first,
  then evaluate the winner; compares six decision rules on the same draws (Table 1).
* :func:`true_v_check` -- the deflated Sharpe with the *measured* cross-trial
  variance V instead of the default approximation.

Every draw is seeded, so the published numbers are reproduced exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from catch_alpha import stats
from catch_alpha.gate import dsr_check, evaluate

PERIODS_PER_YEAR = 252.0  # the synthetic returns are treated as daily
SIGMA = 0.01  # 1% per-period volatility for every simulated series
N_OBS = 1800  # > the 1,559 observations Eq. (1) requires at P = 252, S_min = 1
N_TRIALS = 200  # the trial count the DSR deflates against
BOOTSTRAP_REPS = 150  # fewer resamples than the default 400, for runtime (seeded)


def planted_mu(ann_sharpe: float) -> float:
    """Per-period drift that gives an annualised Sharpe of ``ann_sharpe`` at vol SIGMA."""
    return ann_sharpe / math.sqrt(PERIODS_PER_YEAR) * SIGMA


def _run_gate(candidate, benchmark, baselines, seed: int, reps: int = BOOTSTRAP_REPS) -> dict:
    return evaluate(
        candidate,
        benchmark,
        baselines,
        periods_per_year=PERIODS_PER_YEAR,
        n_trials=N_TRIALS,
        thresholds={"bootstrap_reps": float(reps)},
        seed=seed,
    )


def _pmap(fn, tasks: list, workers: int) -> list:
    if workers <= 1:
        return [fn(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=4)


# ---------------------------------------------------------------------------
# Zero-edge return generators
# ---------------------------------------------------------------------------

T_NU = 3.0
GARCH_A, GARCH_B = 0.08, 0.90
AR1_PHI = 0.2
NULL_FAMILIES = ("gaussian", "student_t", "garch", "ar1")


def draw_null_returns(family: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """One zero-mean return series with unconditional standard deviation SIGMA."""
    if family == "gaussian":
        return rng.normal(0.0, SIGMA, size=n)
    if family == "student_t":  # heavy tails, rescaled to unit variance then to SIGMA
        return rng.standard_t(T_NU, size=n) * (SIGMA / math.sqrt(T_NU / (T_NU - 2.0)))
    if family == "garch":  # volatility clustering, starts at the unconditional variance
        omega = SIGMA * SIGMA * (1.0 - GARCH_A - GARCH_B)
        eps = rng.standard_normal(n)
        r = np.empty(n)
        var = SIGMA * SIGMA
        for t in range(n):
            r[t] = math.sqrt(var) * eps[t]
            var = omega + GARCH_A * r[t] * r[t] + GARCH_B * var
        return r
    if family == "ar1":  # serial correlation, innovations scaled so total sd = SIGMA
        eps = rng.normal(0.0, SIGMA * math.sqrt(1.0 - AR1_PHI * AR1_PHI), size=n)
        r = np.empty(n)
        prev = 0.0
        for t in range(n):
            prev = AR1_PHI * prev + eps[t]
            r[t] = prev
        return r
    raise ValueError(f"unknown null family {family!r}")


# ---------------------------------------------------------------------------
# Experiment 1: null calibration and detection power (Figure 1)
# ---------------------------------------------------------------------------

NULL_SEED_BASE = 0
POWER_SEED_BASE = 1_000_000
EFFECTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def one_strategy(task: tuple[float, int, int]) -> bool:
    """Draw one candidate (drift ``mu``), a benchmark and two baselines; run the gate.

    Candidate, benchmark and baselines are independent, so with mu = 0 there is
    exactly no edge.
    """
    mu, seed, reps = task
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0, SIGMA, size=N_OBS)
    candidate = rng.normal(mu, SIGMA, size=N_OBS)
    baselines = {"b1": rng.normal(0.0, SIGMA, size=N_OBS), "b2": rng.normal(0.0, SIGMA, size=N_OBS)}
    return bool(_run_gate(candidate, benchmark, baselines, seed, reps)["all_pass"])


@dataclass
class PassRate:
    label: str
    effect_ann_sharpe: float
    n: int
    n_pass: int

    @property
    def rate(self) -> float:
        return self.n_pass / self.n if self.n else 0.0

    @property
    def wilson(self) -> tuple[float, float]:
        return stats.wilson_interval(self.n_pass, self.n)


def null_calibration(
    n_null: int = 200,
    n_power: int = 40,
    effects: Iterable[float] = EFFECTS,
    reps: int = BOOTSTRAP_REPS,
    workers: int = 1,
) -> list[PassRate]:
    """Figure 1: acceptance rate of zero-edge strategies, then of planted edges."""
    out = []
    null_tasks = [(0.0, NULL_SEED_BASE + i, reps) for i in range(n_null)]
    out.append(PassRate("null", 0.0, n_null, sum(_pmap(one_strategy, null_tasks, workers))))
    for j, eff in enumerate(effects):
        base = POWER_SEED_BASE + (j + 1) * 100_000
        tasks = [(planted_mu(eff), base + i, reps) for i in range(n_power)]
        out.append(PassRate("power", float(eff), n_power, sum(_pmap(one_strategy, tasks, workers))))
    return out


# ---------------------------------------------------------------------------
# Experiment 2: other null families (heavy tails, GARCH, AR(1))
# ---------------------------------------------------------------------------

ROBUSTNESS_SIZES = (1800, 5000, 50000)


def robustness_seed_base(family: str, n_obs: int) -> int:
    """Seed block of one (family, size) cell, exactly as in the published run.

    The grid was run as parallel shards: one shard for sizes 1,800 and 5,000 and one
    shard per family at 50,000. Each shard numbered its own cells from 10,000,000,
    so the 50,000-observation cells of different families reuse the same seed
    block (the paper therefore claims no pooled bound across cells).
    """
    if n_obs == 50000:
        return 10_000_000
    return (
        10_000_000 + NULL_FAMILIES.index(family) * 1_000_000 + (1800, 5000).index(n_obs) * 100_000
    )


def _one_null_family(task: tuple[str, int, int, int]) -> tuple[bool, float, bool]:
    family, n_obs, seed, reps = task
    rng = np.random.default_rng(seed)
    benchmark = draw_null_returns(family, rng, n_obs)
    candidate = draw_null_returns(family, rng, n_obs)
    baselines = {
        "b1": draw_null_returns(family, rng, n_obs),
        "b2": draw_null_returns(family, rng, n_obs),
    }
    verdict = _run_gate(candidate, benchmark, baselines, seed, reps)
    excess = candidate - benchmark
    sd = float(np.std(excess, ddof=1))
    ann = 0.0 if sd == 0.0 else float(np.mean(excess) / sd * math.sqrt(PERIODS_PER_YEAR))
    return bool(verdict["all_pass"]), ann, verdict["status"] == "ok"


def robustness_grid(
    families: Iterable[str] = NULL_FAMILIES,
    sizes: Iterable[int] = ROBUSTNESS_SIZES,
    n_per_cell: int = 200,
    reps: int = BOOTSTRAP_REPS,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Zero-edge acceptance rate per (family, sample size) cell."""
    rows = []
    for family in families:
        for n_obs in sizes:
            base = robustness_seed_base(family, n_obs)
            results = _pmap(
                _one_null_family,
                [(family, n_obs, base + i, reps) for i in range(n_per_cell)],
                workers,
            )
            n_pass = sum(r[0] for r in results)
            lo, hi = stats.wilson_interval(n_pass, n_per_cell)
            rows.append(
                {
                    "family": family,
                    "n_obs": n_obs,
                    "n_strategies": n_per_cell,
                    "n_pass": n_pass,
                    "pass_rate": n_pass / n_per_cell,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "all_status_ok": all(r[2] for r in results),  # no INSUFFICIENT_DATA
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Experiment 3: selection among correlated candidates (Table 1)
# ---------------------------------------------------------------------------

N_CANDIDATES = 200
RHOS = (0.0, 0.3, 0.7)
SELECTION_EFFECTS = (0.0, 1.5, 2.0, 3.0)  # 0.0 = null cell
N_NULL_CAMPAIGNS = 500
N_PLANTED_CAMPAIGNS = 200
SELECTION_SEED_BASE = 5_000_000
T_ALPHA = 0.05

RULES = {  # labels as in the published selection_aware_calibration.csv
    "V0": "uncorrected t-test, development",
    "V1": "four checks N=200, development",
    "V2": "four checks N=200, holdout",
    "V3": "four checks N=1, development",
    "V4": "DSR only N=200, development",
    "V5": "t-test only, holdout",
}


def selection_cells() -> list[tuple[int, float, float, int]]:
    """(cell index, correlation, planted Sharpe, number of campaigns) in run order."""
    cells = []
    for rho in RHOS:
        for effect in SELECTION_EFFECTS:
            n = N_NULL_CAMPAIGNS if effect == 0.0 else N_PLANTED_CAMPAIGNS
            cells.append((len(cells), rho, effect, n))
    return cells


def _equicorrelated_window(rng: np.random.Generator, mu: np.ndarray, rho: float) -> np.ndarray:
    """[200, N_OBS] returns with pairwise correlation rho (one common factor)."""
    common = rng.standard_normal(N_OBS)
    idio = rng.standard_normal((N_CANDIDATES, N_OBS))
    return mu[:, None] + SIGMA * (math.sqrt(rho) * common[None, :] + math.sqrt(1.0 - rho) * idio)


def _t_test_pass(returns: np.ndarray) -> bool:
    n = returns.size
    t_stat = float(np.mean(returns) / (np.std(returns, ddof=1) / math.sqrt(n)))
    return bool(t_stat > student_t.ppf(1.0 - T_ALPHA, n - 1))


def _checks(verdict: dict) -> dict[str, bool]:
    if verdict["status"] != "ok":
        return {"excess_return": False, "excess_sharpe": False, "dsr": False, "spa": False}
    c = verdict["checks"]
    return {
        "excess_return": bool(c["excess_return_ci_lb_gt_zero"]),
        "excess_sharpe": bool(c["excess_sharpe_ci_lb_gt_zero"]),
        "dsr": bool(c["dsr_ge_min"]),
        "spa": bool(c["spa_worst_p_le_max"]),
    }


def _dsr_with(verdict: dict, n_trials: int) -> bool:
    d = verdict["dsr"]
    return dsr_check(d["sharpe"], n_trials, d["n_obs"], d["dsr_min"], d["skewness"], d["kurtosis"])[
        "all_pass"
    ]


def selection_campaign(task: tuple[int, float, float, int, int]) -> dict[str, Any]:
    """One campaign: draw 200 candidates, select the best Sharpe, evaluate the winner.

    Candidate 0 carries the planted edge (if any). The winner is evaluated on the
    selection ("development") window and on an independent holdout window, each
    against its own benchmark and two baselines.
    """
    cell, rho, effect, campaign, seed = task
    rng = np.random.default_rng(seed)
    mu = np.zeros(N_CANDIDATES)
    mu[0] = planted_mu(effect)
    dev = _equicorrelated_window(rng, mu, rho)
    hold = _equicorrelated_window(rng, mu, rho)

    def reference() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        bench = rng.normal(0.0, SIGMA, size=N_OBS)
        return bench, {
            "b1": rng.normal(0.0, SIGMA, size=N_OBS),
            "b2": rng.normal(0.0, SIGMA, size=N_OBS),
        }

    dev_bench, dev_bases = reference()
    hold_bench, hold_bases = reference()

    dev_sharpe = dev.mean(axis=1) / dev.std(axis=1)
    winner = int(np.argmax(dev_sharpe))
    corr = np.corrcoef(dev)
    mean_corr = float((corr.sum() - N_CANDIDATES) / (N_CANDIDATES * (N_CANDIDATES - 1)))

    v_dev = _run_gate(dev[winner], dev_bench, dev_bases, seed)
    v_hold = _run_gate(hold[winner], hold_bench, hold_bases, seed)
    cd, ch = _checks(v_dev), _checks(v_hold)
    dev_ok = v_dev["status"] == "ok"
    dsr_n1 = _dsr_with(v_dev, 1) if dev_ok else False
    # consistency check: the DSR recomputed from the verdict's own inputs agrees
    recompute_matches = (_dsr_with(v_dev, N_TRIALS) == cd["dsr"]) if dev_ok else False
    three = cd["excess_return"] and cd["excess_sharpe"] and cd["spa"]
    hold_sharpe = float(hold[winner].mean() / hold[winner].std())
    return {
        "cell": cell,
        "rho": str(rho),  # written as in the published file ("0.3", not "0.300000")
        "effect_ann_sharpe": str(effect),
        "campaign": campaign,
        "seed": seed,
        "winner": winner,
        "winner_is_planted": bool(effect > 0.0 and winner == 0),
        "mean_pairwise_corr": mean_corr,
        "dev_sharpe_ann": float(dev_sharpe[winner]) * math.sqrt(PERIODS_PER_YEAR),
        "hold_sharpe_ann": hold_sharpe * math.sqrt(PERIODS_PER_YEAR),
        "dev_status_ok": dev_ok,
        "hold_status_ok": v_hold["status"] == "ok",
        "dev_excess_return": cd["excess_return"],
        "dev_excess_sharpe": cd["excess_sharpe"],
        "dev_dsr": cd["dsr"],
        "dev_spa": cd["spa"],
        "dev_dsr_n1": dsr_n1,
        "hold_excess_return": ch["excess_return"],
        "hold_excess_sharpe": ch["excess_sharpe"],
        "hold_dsr": ch["dsr"],
        "hold_spa": ch["spa"],
        "dsr_recompute_matches": recompute_matches,
        "V0": _t_test_pass(dev[winner]),
        "V1": bool(three and cd["dsr"]),
        "V2": bool(all(ch.values())),
        "V3": bool(three and dsr_n1),
        "V4": cd["dsr"],
        "V5": _t_test_pass(hold[winner]),
    }


def selection_experiment(
    workers: int = 1, campaigns_per_cell: int | None = None, seed_base: int = SELECTION_SEED_BASE
) -> tuple[list[dict], list[dict]]:
    """Run all campaigns; return (per-campaign rows, per-cell-and-rule summary).

    Headline rate: in null cells, false acceptances per campaign; in planted cells,
    detections (accepted AND the winner is the planted candidate).
    """
    tasks = []
    for cell, rho, effect, n in selection_cells():
        n_run = n if campaigns_per_cell is None else campaigns_per_cell
        tasks.extend((cell, rho, effect, i, seed_base + 100_000 * cell + i) for i in range(n_run))
    rows = _pmap(selection_campaign, tasks, workers)

    summary = []
    for cell, rho, effect, _ in selection_cells():
        cell_rows = [r for r in rows if r["cell"] == cell]
        n = len(cell_rows)
        planted_wins = sum(r["winner_is_planted"] for r in cell_rows)
        mean_corr = float(np.mean([float(f"{r['mean_pairwise_corr']:.6f}") for r in cell_rows]))
        insufficient = sum(not (r["dev_status_ok"] and r["hold_status_ok"]) for r in cell_rows)
        for rule, label in RULES.items():
            accepted = sum(r[rule] for r in cell_rows)
            detections = sum(r[rule] and r["winner_is_planted"] for r in cell_rows)
            false_acc = accepted - detections
            k = false_acc if effect == 0.0 else detections
            lo, hi = stats.wilson_interval(k, n)
            summary.append(
                {
                    "cell": cell,
                    "rho": str(rho),
                    "effect_ann_sharpe": str(effect),
                    "variant": rule,
                    "rule": label,
                    "n_campaigns": n,
                    "n_accept": accepted,
                    "n_detect": detections,
                    "n_false_accept": false_acc,
                    "headline_rate": k / n if n else 0.0,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_planted_selected": planted_wins,
                    "n_insufficient_data": insufficient,
                    "mean_pairwise_corr": mean_corr,
                }
            )
    return rows, summary


# ---------------------------------------------------------------------------
# Experiment 4: deflated Sharpe with the measured cross-trial variance
# ---------------------------------------------------------------------------


def true_v_check() -> dict[str, Any]:
    """Re-run the DSR on the 200 Figure-1 null strategies with the measured V.

    The random stream is the same as the null batch of :func:`null_calibration`
    (benchmark, candidate, b1, b2 drawn in that order from seeds 0..199).
    """
    sharpes, moments = [], []
    for i in range(N_TRIALS):
        rng = np.random.default_rng(NULL_SEED_BASE + i)
        rng.normal(0.0, SIGMA, size=N_OBS)  # benchmark (kept to preserve the stream)
        candidate = rng.normal(0.0, SIGMA, size=N_OBS)
        sr = stats.per_period_sharpe(candidate)
        sharpes.append(sr)
        moments.append((sr, *stats.return_moments(candidate)))
    v_true = float(np.var(sharpes, ddof=1))

    pass_proxy = pass_true = 0
    dsr_true_max = 0.0
    proxy_var = []
    for sr, sk, ku in moments:
        pass_proxy += stats.deflated_sharpe_ratio(sr, N_TRIALS, N_OBS, sk, ku) >= 0.95
        d_true = stats.deflated_sharpe_ratio(sr, N_TRIALS, N_OBS, sk, ku, v_true)
        pass_true += d_true >= 0.95
        dsr_true_max = max(dsr_true_max, d_true)
        proxy_var.append(stats.sharpe_standard_error(sr, N_OBS, sk, ku) ** 2)
    z = stats.expected_max_sharpe_z(N_TRIALS)
    return {
        "v_true_cross_trial": v_true,
        "v_proxy_mean_sigma_sr_sq": float(np.mean(proxy_var)),
        "sr0_per_period_true_v": math.sqrt(v_true) * z,
        "sr0_per_period_proxy_mean": math.sqrt(float(np.mean(proxy_var))) * z,
        "dsr_pass_proxy": int(pass_proxy),
        "dsr_pass_true_v": int(pass_true),
        "dsr_true_v_max": float(dsr_true_max),
    }


def dsr_worked_example() -> dict[str, float]:
    """Section 2.2: per-period Sharpe 0.06 on 3,000 observations, N = 3 vs N = 347."""
    return {f"N={n}": stats.deflated_sharpe_ratio(0.06, n, 3000) for n in (3, 347)}
