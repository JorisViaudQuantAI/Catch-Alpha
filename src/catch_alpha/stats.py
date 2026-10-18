"""The statistical building blocks of the evaluator.

Everything here is pure numpy/scipy, deterministic given a seed, and small enough
to read in one sitting. Each function maps to one idea in Section 2 of the paper:

* :func:`minimum_required_observations` / :func:`power_gate` -- "abstain when the
  evidence is too weak" (Eq. 1 of the paper).
* :func:`stationary_bootstrap_ci` -- uncertainty intervals that respect serial
  dependence (Politis & Romano, 1994).
* :func:`deflated_sharpe_ratio` -- the Sharpe ratio after counting every attempt
  (Bailey & Lopez de Prado, 2014; Eq. 2 of the paper).
* :func:`spa_pvalue` -- does the candidate beat a baseline? (Hansen, 2005).
* :func:`probability_of_backtest_overfitting` -- does the in-sample winner keep
  winning out of sample? (Bailey et al., 2017).
* :func:`wilson_interval` -- binomial confidence bounds for pass rates.

The implementations are line-for-line equivalent to the code that produced the
paper's numbers (same formulas, same random-number streams), only reorganised.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _scipy_skew

EULER_MASCHERONI = 0.5772156649015329

MetricFn = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def per_period_sharpe(returns: np.ndarray) -> float:
    """Mean over standard deviation, *not* annualised (multiply by sqrt(P) for that)."""
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    vol = float(np.std(arr))
    if not np.isfinite(vol) or vol < 1e-12:
        return 0.0
    return float(np.mean(arr)) / vol


def mean_return(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    return float(np.mean(arr)) if arr.size else 0.0


def return_moments(returns: np.ndarray) -> tuple[float, float]:
    """(skewness, raw kurtosis) with the normal values (0, 3) for degenerate samples.

    Raw kurtosis means a normal distribution has kurtosis 3 (not excess kurtosis 0).
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size < 3 or float(np.std(arr)) < 1e-12:
        return 0.0, 3.0
    skew_val = float(_scipy_skew(arr))
    kurt_val = float(_scipy_kurtosis(arr, fisher=False))
    return (skew_val if np.isfinite(skew_val) else 0.0, kurt_val if np.isfinite(kurt_val) else 3.0)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% interval for a binomial rate k/n (used for every pass rate)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# 1. Minimum sample length ("abstain when the evidence is too weak")
# ---------------------------------------------------------------------------


def overlap_variance_inflation(horizon_bars: int, n_obs: int) -> float:
    """Variance inflation of a sample mean of *overlapping* h-bar returns.

    Two h-bar returns k bars apart share h-k bars, so their correlation is
    max(0, 1 - k/h). Plugging that triangular autocorrelation into the variance of a
    sample mean gives ``1 + 2 * sum_k (1 - k/h)(1 - k/n)``, which tends to h when
    n >> h ("the effective sample shrinks by about h"). Equals 1 for h = 1.
    """
    h, n = int(horizon_bars), int(n_obs)
    if h <= 1 or n <= 1:
        return 1.0
    # A plain loop on purpose: sum() uses compensated summation from Python 3.12, and a
    # one-ulp difference can flip the ceil() in minimum_required_observations.
    total = 0.0
    for k in range(1, min(h - 1, n - 1) + 1):
        total += (1.0 - k / h) * (1.0 - k / n)
    return max(1.0, 1.0 + 2.0 * total)


def minimum_required_observations(
    periods_per_year: float,
    s_min_annual: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    horizon_bars: int = 1,
) -> int:
    """Eq. (1): n >= ((z_{1-alpha} + z_{power}) / S_min)^2 * P  (~ 6.18 P / S_min^2).

    This is the length a simple one-sided mean test needs to detect an annualised
    Sharpe of ``s_min_annual`` with the requested power. With hourly data
    (P = 8760) and S_min = 1 it gives 54,160 observations, about 6.2 years.
    """
    ppy, s_min = float(periods_per_year), float(s_min_annual)
    if ppy <= 0.0 or s_min <= 0.0:
        raise ValueError("periods_per_year and s_min_annual must be positive")
    if int(horizon_bars) < 1:
        raise ValueError("horizon_bars must be >= 1")
    z_alpha = float(norm.ppf(1.0 - float(alpha)))
    z_power = float(norm.ppf(float(power)))
    n_iid = int(math.ceil(((z_alpha + z_power) / s_min) ** 2 * ppy))
    return int(math.ceil(n_iid * overlap_variance_inflation(int(horizon_bars), n_iid)))


def power_gate(
    n_obs: int,
    periods_per_year: float,
    s_min_annual: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    horizon_bars: int = 1,
) -> dict:
    """Return ``status="ok"`` if the sample is long enough, else ``"insufficient_data"``.

    When the status is ``insufficient_data`` the evaluator reports no Sharpe-type
    number at all: a short sample cannot support either success or failure.
    """
    n_required = minimum_required_observations(
        periods_per_year, s_min_annual, alpha, power, horizon_bars
    )
    n_iid = minimum_required_observations(periods_per_year, s_min_annual, alpha, power, 1)
    return {
        "status": "ok" if int(n_obs) >= n_required else "insufficient_data",
        "n_obs": int(n_obs),
        "n_required": int(n_required),
        "periods_per_year": float(periods_per_year),
        "s_min_annual": float(s_min_annual),
        "horizon_bars": int(horizon_bars),
        "variance_inflation_factor": float(overlap_variance_inflation(int(horizon_bars), n_iid)),
    }


# ---------------------------------------------------------------------------
# 2. Stationary bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------


def _block_size(n_samples: int, block_size: int | None) -> int:
    if n_samples <= 1:
        return 1
    if block_size is None:
        return max(2, min(int(np.sqrt(float(n_samples))), n_samples))
    return max(1, min(int(block_size), n_samples))


def stationary_bootstrap_indices(
    n_samples: int, block_size: int | None = None, seed: int = 0
) -> np.ndarray:
    """One resample of time indices made of blocks with random (geometric) lengths.

    At each step the block continues with probability 1 - 1/block_size, otherwise a
    new block starts at a uniformly random position. Blocks wrap around the end.
    The default mean block length is sqrt(n).
    """
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.int64)
    p_restart = 1.0 / float(_block_size(n_samples, block_size))
    rng = np.random.default_rng(seed)
    out = np.zeros((n_samples,), dtype=np.int64)
    out[0] = int(rng.integers(0, n_samples))
    # Written as a loop on purpose: the exact sequence of random draws defines the
    # published numbers, and the loop makes that sequence obvious.
    for t in range(1, n_samples):
        if float(rng.random()) < p_restart:
            out[t] = int(rng.integers(0, n_samples))
        else:
            out[t] = (out[t - 1] + 1) % n_samples
    return out


def stationary_bootstrap_ci(
    series: np.ndarray,
    metric: MetricFn,
    reps: int = 500,
    alpha: float = 0.05,
    block_size: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of ``metric`` over ``reps`` stationary-bootstrap resamples.

    Resample i uses seed ``seed + i + 1``, so results are reproducible.
    """
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    if reps <= 0:
        point = float(metric(arr))
        return point, point
    a = float(np.clip(alpha, 1e-6, 0.5))
    draws = np.array(
        [
            float(metric(arr[stationary_bootstrap_indices(arr.size, block_size, seed + i + 1)]))
            for i in range(int(reps))
        ]
    )
    return float(np.quantile(draws, a / 2.0)), float(np.quantile(draws, 1.0 - a / 2.0))


# ---------------------------------------------------------------------------
# 3. Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------


def expected_max_sharpe_z(n_trials: int) -> float:
    """Expected maximum of N standard normals (the bracket in Eq. 2 of the paper)."""
    n = max(int(n_trials), 2)
    g = EULER_MASCHERONI
    return float((1.0 - g) * norm.ppf(1.0 - 1.0 / n) + g * norm.ppf(1.0 - 1.0 / (n * np.e)))


def sharpe_standard_error(sharpe: float, n_obs: int, skewness: float, kurtosis: float) -> float:
    """Standard error of an estimated per-period Sharpe ratio (Mertens / Lo).

    sigma_SR = sqrt((1 - skew * SR + (kurt - 1)/4 * SR^2) / (n - 1)), raw kurtosis.
    """
    var = (1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2) / float(n_obs - 1)
    return max(float(np.sqrt(max(var, 0.0))), 1e-12)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance_across_trials: float | None = None,
) -> float:
    """Probability that the true Sharpe beats the best of ``n_trials`` lucky draws.

    ``sharpe`` is per period. The null reference is the expected maximum Sharpe of
    N zero-skill trials, ``SR0 = sqrt(V) * E[max of N normals]`` (Eq. 2), and the
    DSR is ``Phi((SR - SR0) / sigma_SR)``. A strategy passes at DSR >= 0.95.

    ``V`` is the variance of Sharpe estimates across the trials. The trial ledger
    records *which* evaluations ran, not a Sharpe for each, so by default V is
    approximated by the candidate's own sigma_SR^2 (on the Gaussian calibration
    cell: 5.6e-4 approximation vs 4.6e-4 measured; see ``calibration.true_v_check``).
    ``N`` is clamped to at least 2.
    """
    if n_obs <= 1 or not (np.isfinite(sharpe) and np.isfinite(skewness) and np.isfinite(kurtosis)):
        return 0.0
    if sr_variance_across_trials is not None and not np.isfinite(sr_variance_across_trials):
        return 0.0
    sigma_sr = sharpe_standard_error(float(sharpe), int(n_obs), skewness, kurtosis)
    variance = (
        sigma_sr**2 if sr_variance_across_trials is None else max(sr_variance_across_trials, 0.0)
    )
    sr0 = np.sqrt(variance) * expected_max_sharpe_z(n_trials)
    return float(norm.cdf((float(sharpe) - sr0) / sigma_sr))


# ---------------------------------------------------------------------------
# 4. Superior predictive ability, pairwise (Hansen, 2005)
# ---------------------------------------------------------------------------


def spa_pvalue(
    candidate: np.ndarray,
    baseline: np.ndarray,
    reps: int = 500,
    block_size: int | None = None,
    seed: int = 0,
) -> float:
    """p-value of "the candidate does not beat this baseline".

    Studentised mean of the return difference, with Hansen's lower recentring
    (only positive mean differences are removed) and stationary-bootstrap resamples.
    The evaluator runs this against every declared baseline and keeps the worst
    (largest) p-value. Unlike Hansen's original statistic, the studentised mean is
    not truncated at zero, so a candidate that underperforms gets p close to 0.5
    rather than 1; at the 0.10 threshold the decision is the same.
    """
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    t = int(min(cand.size, base.size))
    if t <= 1:
        return 1.0
    cand, base = cand[:t], base[:t]
    finite = np.isfinite(base) & np.isfinite(cand)
    if not np.any(finite):
        return 1.0
    diff = (cand - base)[finite]
    n = int(diff.size)
    mu = float(np.mean(diff))
    sigma = float(np.std(diff)) + 1e-12
    stat_obs = np.sqrt(float(n)) * mu / sigma
    recentered = diff - max(mu, 0.0)
    block = _block_size(n, block_size)
    exceed = 0
    for i in range(int(max(1, reps))):
        idx = stationary_bootstrap_indices(n, block, seed + i + 7)
        if np.sqrt(float(n)) * float(np.mean(recentered[idx])) / sigma >= stat_obs:
            exceed += 1
    return exceed / float(max(1, reps))


# ---------------------------------------------------------------------------
# 5. Probability of backtest overfitting (CSCV; Bailey et al., 2017)
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    score_matrix: np.ndarray, max_combinations: int = 4096, seed: int = 42
) -> float:
    """Share of fold splits where the in-sample winner lands below the OOS median.

    ``score_matrix`` has shape [n_strategies, n_folds]. Every way of splitting the
    folds into two halves (C(6,3) = 20 splits for six folds) picks the best strategy
    on one half and records its relative rank w on the other half; PBO is the share
    of splits with w <= 1/2. Returns NaN for an under-identified matrix (fewer than
    two strategies or folds) so that downstream checks fail closed.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 2:
        return float("nan")
    n_strat, n_folds = scores.shape
    k = max(1, min(n_folds // 2, n_folds - 1))
    total = math.comb(n_folds, k)
    if total <= max_combinations and n_folds <= 32:
        combos = list(itertools.combinations(range(n_folds), k))
    else:  # sample unique splits when the full enumeration would explode
        rng = np.random.default_rng(seed)
        seen: set[tuple[int, ...]] = set()
        combos = []
        while len(combos) < min(total, max_combinations):
            c = tuple(sorted(int(x) for x in rng.choice(n_folds, size=k, replace=False)))
            if c not in seen:
                seen.add(c)
                combos.append(c)

    below, valid = 0, 0
    for combo in combos:
        in_idx = np.asarray(combo)
        out_idx = np.asarray(sorted(set(range(n_folds)) - set(combo)))
        ins = scores[:, in_idx].mean(axis=1)
        oos = scores[:, out_idx].mean(axis=1)
        if not (np.isfinite(ins).all() and np.isfinite(oos).all()):
            continue
        best = int(np.argmax(ins))
        rank = int(np.sum(oos <= oos[best]))  # 1 = worst, n_strat = best
        below += int(rank / float(n_strat + 1) <= 0.5)
        valid += 1
    return float("nan") if valid == 0 else float(np.clip(below / valid, 0.0, 1.0))
