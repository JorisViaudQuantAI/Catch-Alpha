"""The evaluator: four required checks plus a fold-based overfitting diagnostic.

A candidate strategy is accepted only if ALL of the following hold (Section 2):

0. **Enough data.** The sample is at least as long as Eq. (1) requires. Otherwise the
   verdict is ``INSUFFICIENT_DATA`` and *no* Sharpe-type number is reported.
1. **Uncertainty intervals above the benchmark.** The stationary-bootstrap 95% lower
   bounds of both the mean and the Sharpe ratio of ``candidate - benchmark`` are > 0.
2. **Credible after counting every attempt.** Deflated Sharpe ratio >= 0.95, with
   ``n_trials`` taken from the trial ledger.
3. **Better than every baseline.** The worst pairwise SPA p-value is <= 0.10.
4. **(Diagnostic) no evidence of selection overfitting.** PBO <= 0.50 when a
   [strategies x folds] score matrix exists. In ``decision_mode`` a missing matrix
   blocks acceptance, but a low PBO can never establish profitability on its own.

All thresholds are fixed before a run (:data:`DEFAULT_THRESHOLDS`).

Example
-------
>>> import numpy as np
>>> from catch_alpha.gate import evaluate
>>> rng = np.random.default_rng(0)
>>> verdict = evaluate(
...     candidate=rng.normal(0, 0.01, 1800),
...     benchmark=rng.normal(0, 0.01, 1800),
...     baselines={"b1": rng.normal(0, 0.01, 1800)},
...     periods_per_year=252,
...     n_trials=200,
... )
>>> verdict["all_pass"]
False
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from catch_alpha import stats

DEFAULT_THRESHOLDS: dict[str, float] = {
    "s_min_annual": 1.0,  # smallest annualised Sharpe the test must be powered to detect
    "power_alpha": 0.05,  # significance of the minimum-length calculation
    "power_beta": 0.20,  # 1 - power of the minimum-length calculation
    "excess_ci_alpha": 0.05,  # bootstrap intervals are 95%
    "dsr_min": 0.95,  # deflated Sharpe threshold
    "spa_p_max": 0.10,  # worst baseline SPA p-value allowed
    "pbo_max": 0.5,  # probability of backtest overfitting allowed
    "bootstrap_reps": 400.0,  # resamples for intervals and SPA
    "test_access_budget": 3.0,  # final-period evaluations allowed (informational)
}


def _as_returns(series: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} returns are empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} returns contain non-finite values")
    return arr


def excess_check(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    reps: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Check 1: bootstrap lower bounds of the excess mean and excess Sharpe are > 0."""
    excess = np.asarray(candidate, dtype=np.float64) - np.asarray(benchmark, dtype=np.float64)
    ret_ci = stats.stationary_bootstrap_ci(excess, stats.mean_return, reps, alpha, seed=seed)
    sharpe_ci = stats.stationary_bootstrap_ci(
        excess, stats.per_period_sharpe, reps, alpha, seed=seed
    )
    checks = {
        "excess_return_ci_lb_gt_zero": bool(np.isfinite(ret_ci[0]) and ret_ci[0] > 0.0),
        "excess_sharpe_ci_lb_gt_zero": bool(np.isfinite(sharpe_ci[0]) and sharpe_ci[0] > 0.0),
    }
    return {
        "checks": checks,
        "excess_return_ci": [float(ret_ci[0]), float(ret_ci[1])],
        "excess_sharpe_ci": [float(sharpe_ci[0]), float(sharpe_ci[1])],
        "all_pass": all(checks.values()),
    }


def dsr_check(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    dsr_min: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance_across_trials: float | None = None,
) -> dict[str, Any]:
    """Check 2: deflated Sharpe ratio >= ``dsr_min`` (``sharpe`` is per period)."""
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    effective = max(2, int(n_trials))  # N = 1 is clamped to 2 by the DSR formula
    dsr = stats.deflated_sharpe_ratio(
        sharpe, effective, n_obs, skewness, kurtosis, sr_variance_across_trials
    )
    ok = bool(np.isfinite(dsr) and dsr >= dsr_min)
    return {
        "checks": {"dsr_ge_min": ok},
        "dsr": float(dsr),
        "dsr_min": float(dsr_min),
        "sharpe": float(sharpe),
        "n_trials": int(n_trials),
        "effective_n_trials": effective,
        "n_obs": int(n_obs),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "sr_variance_across_trials": sr_variance_across_trials,
        "all_pass": ok,
    }


def spa_check(
    candidate: np.ndarray,
    baselines: dict[str, np.ndarray],
    p_max: float = 0.10,
    reps: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Check 3: the candidate must beat *every* baseline (worst p-value <= ``p_max``)."""
    per_baseline = {
        str(name): stats.spa_pvalue(candidate, series, reps=reps, seed=seed)
        for name, series in baselines.items()
    }
    worst = max(per_baseline.values()) if per_baseline else 1.0  # no baseline: fail closed
    ok = bool(np.isfinite(worst) and worst <= p_max)
    return {
        "checks": {"spa_worst_p_le_max": ok},
        "p_value": float(worst),
        "p_max": float(p_max),
        "per_baseline_p": per_baseline,
        "all_pass": ok,
    }


def pbo_check(score_matrix: np.ndarray, pbo_max: float = 0.5) -> dict[str, Any]:
    """Diagnostic: PBO of a [strategies x folds] matrix must not exceed ``pbo_max``."""
    pbo = stats.probability_of_backtest_overfitting(score_matrix, seed=42)
    ok = bool(np.isfinite(pbo) and pbo <= pbo_max)
    return {"checks": {"pbo_le_max": ok}, "pbo": pbo, "pbo_max": pbo_max, "all_pass": ok}


def evaluate(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    baselines: dict[str, np.ndarray],
    periods_per_year: float,
    n_trials: int,
    score_matrix: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
    seed: int = 0,
    decision_mode: bool = False,
    horizon_bars: int = 1,
) -> dict[str, Any]:
    """Run the full evaluator on one candidate return series.

    Parameters
    ----------
    candidate, benchmark:
        Per-period returns of equal length (benchmark = the declared reference,
        e.g. cash or equal-weight buy-and-hold).
    baselines:
        Named return series the candidate must beat (SPA check).
    periods_per_year:
        252 for daily, 8760 for hourly returns.
    n_trials:
        Number of evaluations recorded on this dataset, from the trial ledger.
    score_matrix:
        Optional [strategies x folds] matrix for the PBO diagnostic.
    thresholds:
        Overrides of :data:`DEFAULT_THRESHOLDS` (a decision with overrides is
        diagnostic only).
    decision_mode:
        If True, a missing PBO matrix or a threshold override blocks acceptance.
    horizon_bars:
        Overlap horizon of the returns (1 = non-overlapping); a data property that
        inflates the minimum length, not a tunable threshold.

    Returns
    -------
    dict with ``status`` (``"ok"`` or ``"INSUFFICIENT_DATA"``), ``all_pass``, the
    individual ``checks`` and every intermediate statistic.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({str(k): float(v) for k, v in thresholds.items()})
    reps = int(th["bootstrap_reps"])

    cand = _as_returns(candidate, "candidate")
    bench = _as_returns(benchmark, "benchmark")
    if cand.size != bench.size:
        raise ValueError(f"candidate and benchmark length mismatch: {cand.size} != {bench.size}")
    bases = {}
    for name, series in baselines.items():
        arr = _as_returns(series, f"baseline {name!r}")
        if arr.size != cand.size:
            raise ValueError(f"candidate and baseline {name!r} length mismatch")
        bases[str(name)] = arr

    power = stats.power_gate(
        cand.size,
        periods_per_year,
        th["s_min_annual"],
        th["power_alpha"],
        1.0 - th["power_beta"],
        int(horizon_bars),
    )
    decision_checks = {
        "pbo_score_matrix_present": score_matrix is not None,
        "default_thresholds_used": not bool(thresholds),
    }
    if power["status"] != "ok":
        return {
            "status": "INSUFFICIENT_DATA",
            "power_gate": power,
            "all_pass": False,
            "statistical_all_pass": False,
            "decision_mode": bool(decision_mode),
            "decision_eligible": False,
            "decision_checks": decision_checks if decision_mode else {},
            "checks": {},
        }

    excess = excess_check(cand, bench, reps, th["excess_ci_alpha"], seed)
    sharpe = stats.per_period_sharpe(cand)
    skew, kurt = stats.return_moments(cand)
    dsr = dsr_check(sharpe, n_trials, cand.size, th["dsr_min"], skew, kurt)
    dsr["annualized_sharpe"] = float(sharpe * math.sqrt(float(periods_per_year)))
    spa = spa_check(cand, bases, th["spa_p_max"], reps, seed)

    checks: dict[str, bool | None] = {
        **excess["checks"],
        **dsr["checks"],
        **spa["checks"],
    }
    if score_matrix is not None:
        pbo: dict[str, Any] | None = pbo_check(score_matrix, th["pbo_max"])
        checks["pbo_le_max"] = pbo["checks"]["pbo_le_max"]
        pbo_status = "computed"
    elif decision_mode:
        pbo, checks["pbo_le_max"], pbo_status = None, False, "missing_score_matrix_decision_blocker"
    else:
        pbo, checks["pbo_le_max"], pbo_status = None, None, "not_supplied_mechanics"

    statistical_all_pass = all(v for v in checks.values() if v is not None)
    all_pass = statistical_all_pass and (all(decision_checks.values()) if decision_mode else True)
    return {
        "status": "ok",
        "checks": checks,
        "all_pass": bool(all_pass),
        "statistical_all_pass": bool(statistical_all_pass),
        "decision_mode": bool(decision_mode),
        "decision_eligible": bool(decision_mode and all_pass),
        "decision_checks": decision_checks if decision_mode else {},
        "power_gate": power,
        "excess": excess,
        "dsr": dsr,
        "spa": spa,
        "pbo": pbo,
        "pbo_status": pbo_status,
        "n_trials": int(n_trials),
        "periods_per_year": float(periods_per_year),
        "horizon_bars": int(horizon_bars),
    }
