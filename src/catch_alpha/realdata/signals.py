"""Price-signal screen: does any of 32 pre-registered signals rank future returns?

For each (signal, horizon) pair -- 32 signals x {1, 8, 24, 72} hours = 128 pairs --
we measure, every hour, the cross-sectional Spearman rank correlation between the
signal and the subsequent ``h``-hour return across the (up to) twelve contracts.
This per-hour number is the *information coefficient* (IC). A positive IC means
assets ranked high by the signal tend to outperform; a negative IC means the
ranking reverses.

The hourly IC series is averaged within each of six chronological folds. Because
``h``-hour forward returns overlap, consecutive ICs are correlated, so the standard
error of each fold mean uses the Newey-West (Bartlett kernel) estimator with lag
``h``. A fold is *stable* when ``mean +/- 1.96 * se`` excludes zero, and a pair is
*stable* when this holds in at least four of the six folds.

Only train+valid data is used. A (time, asset) cell participates only if the
contract is tradable both at the signal time ``t`` and at the horizon end ``t+h``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

from catch_alpha.realdata.panel import Panel, signal_folds

HORIZONS: tuple[int, ...] = (1, 8, 24, 72)
MOMENTUM_LOOKBACKS: tuple[int, ...] = (24, 72, 168, 336)
MEANREV_LOOKBACK = 168
INV_VOL_LOOKBACK = 168
MIN_ASSETS = 3  # an hour needs at least three valid contracts to define a rank correlation
STABLE_FOLDS = 4  # stable in >= 4 of 6 folds
Z_95 = 1.96


# ---------------------------------------------------------------------------
# Signals (all strictly causal: the value at t uses data up to t only)
# ---------------------------------------------------------------------------


def trailing_returns(prices: np.ndarray, lookback: int) -> np.ndarray:
    """``[T, N]`` of ``P[t]/P[t-lookback] - 1`` (NaN for the first rows / unlisted)."""
    out = np.full(prices.shape, np.nan)
    if prices.shape[0] > lookback:
        prev, cur = prices[:-lookback], prices[lookback:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lookback:] = np.where(prev > 0.0, cur / prev - 1.0, np.nan)
    return out


def inverse_volatility(prices: np.ndarray, lookback: int = INV_VOL_LOOKBACK) -> np.ndarray:
    """``1 / std`` of the last ``lookback`` hourly returns (population std, NaN-aware)."""
    n_t, n_a = prices.shape
    rets = np.full((n_t, n_a), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets[1:] = np.where(prices[:-1] > 0.0, prices[1:] / prices[:-1] - 1.0, np.nan)
    out = np.full((n_t, n_a), np.nan)
    windows = sliding_window_view(rets, lookback, axis=0)  # [T-lookback+1, N, lookback]
    chunk = 2048
    for lo in range(0, windows.shape[0], chunk):  # row i of `windows` ends at bar i+lookback-1
        block = windows[lo : lo + chunk]
        with warnings.catch_warnings():  # all-NaN windows (before listing) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            vol = np.nanstd(block, axis=2)
        ok = np.isfinite(vol) & (vol > 1e-12)
        inv = np.full(vol.shape, np.nan)
        inv[ok] = 1.0 / vol[ok]
        out[lo + lookback - 1 : lo + lookback - 1 + block.shape[0]] = inv
    out[:lookback] = np.nan  # the original screen starts at t = lookback
    return out


@dataclass(frozen=True)
class Signal:
    name: str
    family: str
    values: np.ndarray  # [T, N]


def build_signals(panel: Panel) -> list[Signal]:
    """The 32 frozen signals: 25 raw features plus 7 price-derived signals.

    * ``feat_<name>``: each raw panel feature (unnormalised; rank statistics are
      unaffected by monotone scaling anyway). ``feat_funding_rate_last`` is the raw
      funding rate and ``feat_ret_1`` the last one-hour log return.
    * ``momentum_{24,72,168,336}h``: trailing simple return.
    * ``meanrev_168h``: minus the 168-hour trailing return.
    * ``inv_vol_168h``: inverse realised volatility over 168 hours.
    * ``carry_funding``: minus the funding rate (a long position earns ``-rate``).
    """
    signals = [
        Signal(f"feat_{name}", "raw_feature", panel.features[:, :, j])
        for j, name in enumerate(panel.feature_names)
    ]
    signals += [
        Signal(f"momentum_{h}h", "momentum", trailing_returns(panel.prices, h))
        for h in MOMENTUM_LOOKBACKS
    ]
    signals.append(
        Signal(
            f"meanrev_{MEANREV_LOOKBACK}h",
            "mean_reversion",
            -trailing_returns(panel.prices, MEANREV_LOOKBACK),
        )
    )
    signals.append(Signal("inv_vol_168h", "inverse_vol", inverse_volatility(panel.prices)))
    signals.append(Signal("carry_funding", "carry", -panel.funding))
    return signals


def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """``[T, N]`` of ``P[t+h]/P[t] - 1``; the last ``h`` rows are NaN (no peeking past the panel)."""
    n_t = prices.shape[0]
    out = np.full(prices.shape, np.nan)
    if n_t > horizon:
        cur, fut = prices[: n_t - horizon], prices[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: n_t - horizon] = np.where(cur > 0.0, fut / cur - 1.0, np.nan)
    return out


def horizon_tradable(tradable: np.ndarray, horizon: int) -> np.ndarray:
    """True where the contract is tradable at both ``t`` and ``t + horizon``."""
    out = np.zeros(tradable.shape, dtype=bool)
    n = tradable.shape[0]
    if n > horizon:
        out[: n - horizon] = tradable[: n - horizon] & tradable[horizon:]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def rank_ic_series(signal: np.ndarray, future: np.ndarray) -> np.ndarray:
    """``[T]`` cross-sectional Spearman correlation per hour (NaN if < 3 valid assets).

    Spearman's rho is the Pearson correlation of average ranks, computed over the
    assets where both the signal and the forward return are finite.
    """
    valid = np.isfinite(signal) & np.isfinite(future)
    s = np.where(valid, signal, np.nan)
    f = np.where(valid, future, np.nan)
    rs = rankdata(s, axis=1, nan_policy="omit")
    rf = rankdata(f, axis=1, nan_policy="omit")
    n_valid = valid.sum(axis=1)
    count = np.maximum(n_valid, 1)[:, None]
    rs, rf = np.where(valid, rs, 0.0), np.where(valid, rf, 0.0)
    rs = np.where(valid, rs - rs.sum(axis=1, keepdims=True) / count, 0.0)
    rf = np.where(valid, rf - rf.sum(axis=1, keepdims=True) / count, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt((rs * rs).sum(axis=1) * (rf * rf).sum(axis=1))
        ic = (rs * rf).sum(axis=1) / denom
    ic[(n_valid < MIN_ASSETS) | ~(denom >= 1e-14)] = np.nan
    return ic


def newey_west_mean(series: np.ndarray, lags: int) -> dict[str, float]:
    """Mean of a series and its Newey-West t-statistic (Bartlett weights ``1 - k/(L+1)``).

    NaNs are dropped first. The lag is capped at ``n - 1``.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {"mean": math.nan, "tstat": math.nan, "n_obs": 0, "lags": 0}
    mean = float(x.mean())
    lags = min(int(lags), n - 1)
    e = x - mean
    long_run_var = float(np.dot(e, e)) / n
    for k in range(1, lags + 1):
        long_run_var += 2.0 * (1.0 - k / (lags + 1.0)) * float(np.dot(e[k:], e[:-k])) / n
    se = math.sqrt(max(long_run_var, 1e-30) / n)
    return {"mean": mean, "tstat": mean / se if se > 0 else math.nan, "n_obs": n, "lags": lags}


@dataclass
class PairResult:
    """Screen result for one (signal, horizon) pair."""

    signal: str
    family: str
    horizon: int
    folds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def folds_stable(self) -> int:
        return sum(1 for f in self.folds if f["stable"])

    @property
    def mean_ic(self) -> float:
        means = [f["mean_ic"] for f in self.folds if np.isfinite(f["mean_ic"])]
        return float(np.mean(means)) if means else math.nan

    @property
    def median_fold_t(self) -> float:
        ts = [f["tstat"] for f in self.folds if np.isfinite(f["tstat"])]
        return float(np.median(ts)) if ts else math.nan

    @property
    def stable(self) -> bool:
        return self.folds_stable >= STABLE_FOLDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "family": self.family,
            "horizon_h": self.horizon,
            "mean_rank_ic": self.mean_ic,
            "median_fold_t": self.median_fold_t,
            "folds_stable": self.folds_stable,
            "stable": self.stable,
            "folds": self.folds,
        }


def screen_pair(
    signal: Signal, future: np.ndarray, mask: np.ndarray, folds: list[tuple[int, int]], horizon: int
) -> PairResult:
    """Evaluate one (signal, horizon) pair on every fold."""
    ic = rank_ic_series(np.where(mask, signal.values, np.nan), np.where(mask, future, np.nan))
    result = PairResult(signal.name, signal.family, horizon)
    for i, (lo, hi) in enumerate(folds):
        seg = ic[lo:hi]
        nw = newey_west_mean(seg, lags=horizon)
        mean, t = nw["mean"], nw["tstat"]
        if np.isfinite(mean) and np.isfinite(t) and abs(t) > 1e-12:
            se = abs(mean / t)
            lo_ci, hi_ci = mean - Z_95 * se, mean + Z_95 * se
            stable = lo_ci > 0.0 or hi_ci < 0.0
        else:
            se = lo_ci = hi_ci = math.nan
            stable = False
        result.folds.append(
            {
                "fold": i,
                "start": lo,
                "end": hi,
                "mean_ic": mean,
                "tstat": t,
                "se": se,
                "ci_low": lo_ci,
                "ci_high": hi_ci,
                "n_hours": int(nw["n_obs"]),
                "coverage": float(np.isfinite(seg).mean()) if seg.size else 0.0,
                "stable": bool(stable),
            }
        )
    return result


def run_signal_screen(panel: Panel) -> list[PairResult]:
    """Screen all 128 pre-registered (signal, horizon) pairs on a train+valid panel."""
    folds = signal_folds(panel.n_rows)
    results = []
    future = {h: forward_returns(panel.prices, h) for h in HORIZONS}
    masks = {h: horizon_tradable(panel.tradable, h) for h in HORIZONS}
    for sig in build_signals(panel):
        for h in HORIZONS:
            results.append(screen_pair(sig, future[h], masks[h], folds, h))
    return results


def summarize_screen(results: list[PairResult]) -> dict[str, Any]:
    """Headline counts: number of pairs, number stable, and the stable pairs' signs."""
    stable = [r for r in results if r.stable]
    return {
        "n_pairs": len(results),
        "n_stable": len(stable),
        "n_stable_negative": sum(1 for r in stable if r.mean_ic < 0),
        "n_stable_positive": sum(1 for r in stable if r.mean_ic > 0),
        "stable_momentum_pairs": [
            {"signal": r.signal, "horizon_h": r.horizon, "mean_rank_ic": r.mean_ic}
            for r in stable
            if r.family == "momentum"
        ],
    }


# Pairs plotted in Figure 3 (left panel), in the order of the paper's signal_atlas_price.csv.
FIGURE_PAIRS: tuple[tuple[str, int], ...] = (
    ("feat_ret_1", 1),
    ("feat_compound_return_8", 1),
    ("feat_rsi", 1),
    ("momentum_24h", 1),
    ("momentum_72h", 1),
    ("feat_realized_vol", 24),
    ("feat_spread_proxy", 24),
    ("feat_ema_fast", 24),
    ("inv_vol_168h", 8),
    ("feat_illiq", 8),
    ("carry_funding", 24),
    ("meanrev_168h", 1),
)

FIGURE_HEADER: tuple[str, ...] = (
    "# figure: F3 panel A (price-family rank-IC forest plot)",
    "# source: catch_alpha.realdata.signals.figure_rows (scripts/real_data_signal_screen.py),",
    "#   public_v3 train+valid. rank-IC = per-period cross-sectional Spearman; nw_t = median",
    "#   per-fold Newey-West t (lag = horizon); folds_stable out of 6 (NW CI excludes 0).",
    "# Bars in F3 are mean +/- 1.96*(|mean|/|nw_t|): a heuristic, not a 95% CI of the mean.",
)


def figure_rows(results: list[PairResult]) -> list[dict[str, Any]]:
    """Rows of ``signal_atlas_price.csv`` (rounded exactly as in the paper file)."""
    by_key = {(r.signal, r.horizon): r for r in results}
    rows = []
    for name, h in FIGURE_PAIRS:
        r = by_key[(name, h)]
        rows.append(
            {
                "signal": r.signal,
                "family": r.family,
                "horizon_h": h,
                "mean_rank_ic": f"{r.mean_ic:.4f}",
                "nw_t": f"{r.median_fold_t:.2f}",
                "folds_stable": r.folds_stable,
                "folds_total": len(r.folds),
            }
        )
    return rows
