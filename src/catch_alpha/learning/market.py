"""Planted-alpha synthetic market used by the controlled learning test.

The market plants a *known* predictive signal so that a learner can be graded against
ground truth. Per asset ``i`` (assets are independent)::

    s[t+1] = phi * s[t] + eta[t],      eta ~ N(0, 1 - phi^2)      (stationary std 1)
    r[t+1] = b * s[t] + eps[t+1],      eps ~ N(0, sigma_eps^2)

The loading and residual noise are calibrated in closed form so that the information
coefficient ``corr(s[t], r[t+1])`` equals ``ic`` and the return volatility equals ``vol``::

    b = ic * vol,     sigma_eps = vol * sqrt(1 - ic^2)     =>  Var(r) = vol^2.

Timing convention: ``signal[t]`` and ``features[t]`` are observable at ``t`` and predict
``returns[t+1]``; ``returns[0] = 0``. A weight chosen at ``t`` earns ``returns[t+1]``.

The observed features are ``[signal, trailing_vol_16, trailing_mean_16]``; the two trailing
statistics are causal (left-expanding window) and carry no information about the future.

Random draws come from a single ``np.random.default_rng(seed)`` in a fixed order (initial
signal, signal innovations, return noise), so a seed fully determines a path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_WINDOW = 16
FEATURE_NAMES = ("signal", "trailing_vol_16", "trailing_mean_16")


@dataclass(frozen=True)
class MarketPath:
    """One simulated path. Arrays are float64 and time-major."""

    returns: np.ndarray  # [T, N]; returns[t] is realised at t; returns[0] = 0
    signal: np.ndarray  # [T, N]; signal[t] predicts returns[t+1]
    features: np.ndarray  # [T, N, 3]; features[t] is observable at t
    seed: int
    ic: float
    phi: float
    vol: float

    @property
    def n_periods(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[1])

    def head(self, n_periods: int) -> MarketPath:
        """First ``n_periods`` rows. Every feature is causal, so slicing loses nothing."""
        if not 2 <= n_periods <= self.n_periods:
            raise ValueError(f"cannot slice {self.n_periods} periods to {n_periods}")
        return MarketPath(
            returns=self.returns[:n_periods].copy(),
            signal=self.signal[:n_periods].copy(),
            features=self.features[:n_periods].copy(),
            seed=self.seed,
            ic=self.ic,
            phi=self.phi,
            vol=self.vol,
        )


def signal_loading(ic: float, vol: float) -> float:
    """Return loading ``b = ic * vol`` (the signal has unit stationary std)."""
    return float(ic) * float(vol)


def residual_std(ic: float, vol: float) -> float:
    """Residual return std ``sigma_eps = vol * sqrt(1 - ic^2)``."""
    if abs(ic) > 1.0:
        raise ValueError(f"|ic| must be <= 1, got {ic}")
    return float(vol) * float(np.sqrt(1.0 - float(ic) * float(ic)))


def realized_ic(signal: np.ndarray, returns: np.ndarray) -> float:
    """Pooled Pearson correlation of ``signal[t]`` with ``returns[t+1]``."""
    x = np.asarray(signal, dtype=np.float64)[:-1].reshape(-1)
    y = np.asarray(returns, dtype=np.float64)[1:].reshape(-1)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _trailing(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    """Causal trailing std/mean over ``values[max(0, t-window+1) : t+1]`` (std of 1 row = 0)."""
    out = np.zeros_like(values, dtype=np.float64)
    for t in range(values.shape[0]):
        block = values[max(0, t - window + 1) : t + 1]
        if reducer == "std":
            out[t] = np.std(block, axis=0) if block.shape[0] > 1 else 0.0
        else:
            out[t] = np.mean(block, axis=0)
    return out


def simulate_market(
    seed: int,
    *,
    n_periods: int,
    n_assets: int = 5,
    ic: float = 0.05,
    phi: float = 0.9,
    vol: float = 0.01,
) -> MarketPath:
    """Simulate one planted-alpha path (oracle observation: feature 0 is the signal)."""
    if n_periods < 2 or n_assets < 1:
        raise ValueError("need n_periods >= 2 and n_assets >= 1")
    rng = np.random.default_rng(seed)
    b = signal_loading(ic, vol)
    sigma_eps = residual_std(ic, vol)
    eta_std = float(np.sqrt(max(0.0, 1.0 - phi * phi)))

    # Draw order matters for reproducibility: s[0], then eta, then eps.
    signal = np.empty((n_periods, n_assets), dtype=np.float64)
    signal[0] = rng.standard_normal((1, n_assets))[0]
    eta = rng.standard_normal((n_periods - 1, n_assets)) * eta_std
    for t in range(1, n_periods):
        signal[t] = phi * signal[t - 1] + eta[t - 1]
    eps = rng.standard_normal((n_periods, n_assets)) * sigma_eps

    returns = np.zeros((n_periods, n_assets), dtype=np.float64)
    returns[1:] = b * signal[:-1] + eps[1:]

    features = np.stack(
        [
            signal.copy(),
            _trailing(returns, FEATURE_WINDOW, "std"),
            _trailing(returns, FEATURE_WINDOW, "mean"),
        ],
        axis=-1,
    )
    return MarketPath(
        returns=returns,
        signal=signal,
        features=features,
        seed=int(seed),
        ic=float(ic),
        phi=float(phi),
        vol=float(vol),
    )
