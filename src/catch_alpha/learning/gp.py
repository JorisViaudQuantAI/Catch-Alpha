r"""Gârleanu–Pedersen (2013) reference rule and weight-path accounting.

The learning test grades each learner against a *named reference*: the closed-form
Gârleanu–Pedersen (GP) trading rule for the planted market. GP is optimal for a
risk-adjusted objective; the score below is raw net PnL, so GP is a reference, not the
literal optimum of the score.

Derivation (per asset, signal ``s``, known ``b``, ``phi``, ``sigma_eps``)
-------------------------------------------------------------------------
The trader maximises, with ``x_t`` the weight and ``g = gamma * sigma_eps^2``::

    J = sum_t [ x_t b s_t - (g/2) x_t^2 - (lam/2) (x_t - x_{t-1})^2 ].

Guess ``V(x_{t-1}, s) = -(c_xx/2) x_{t-1}^2 + c_xs x_{t-1} s + f(s)``. With
``E[s_{t+1}] = phi s`` the first-order condition of the Bellman equation is::

    b s - g x - lam (x - x_{t-1}) - c_xx x + c_xs phi s = 0.

Write ``P = g + c_xx`` and ``D = P + lam``. Then ``x_t = x_{t-1} + A (aim_t - x_{t-1})``: a
partial adjustment with trade rate ``A = P / D``. Matching the ``x_{t-1}^2`` term gives
``c_xx = lam P / D``, i.e. the Riccati quadratic::

    c_xx^2 + g c_xx - lam g = 0   =>   c_xx = (-g + sqrt(g^2 + 4 lam g)) / 2.

Matching the ``x_{t-1} s`` term gives ``c_xs = b alpha / (1 - phi alpha)`` with
``alpha = lam / D``; using ``P^2 = g D`` the aim is the Markowitz weight shrunk by::

    aim_t = m(phi) * b s_t / g,      m(phi) = P / (D - phi lam).

Limits: ``lam -> 0`` gives ``A = 1, m = 1`` (frictionless Markowitz); ``lam -> inf`` gives
``A -> 0``; ``phi -> 1`` gives ``m -> 1``; ``m(0) = A``.

Weights are long-short, capped at ``|w| <= weight_cap``; cash ``1 - sum(w)`` is implicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catch_alpha.learning.market import MarketPath, residual_std, signal_loading

_EPS = 1e-12


def _risk_curvature(gamma: float, sigma_eps: float) -> float:
    """Per-period risk curvature ``g = gamma * sigma_eps^2``."""
    return max(float(gamma) * float(sigma_eps) * float(sigma_eps), _EPS)


def _riccati_root(g: float, lam: float) -> float:
    """Non-negative root of ``c^2 + g c - lam g = 0`` (zero when frictionless)."""
    if lam <= _EPS:
        return 0.0
    return float((-g + np.sqrt(g * g + 4.0 * lam * g)) / 2.0)


def gp_trading_rate(*, gamma: float, sigma_eps: float, lam: float) -> float:
    """Trade rate ``A = P / D`` in ``(0, 1]``."""
    g = _risk_curvature(gamma, sigma_eps)
    lam = max(float(lam), 0.0)
    p = g + _riccati_root(g, lam)
    return float(p / (p + lam))


def gp_aim_multiplier(*, gamma: float, sigma_eps: float, lam: float, phi: float) -> float:
    """Aim shrinkage ``m(phi) = P / (D - phi lam)`` (1 when frictionless)."""
    g = _risk_curvature(gamma, sigma_eps)
    lam = max(float(lam), 0.0)
    if lam <= _EPS:
        return 1.0
    p = g + _riccati_root(g, lam)
    return float(p / (p + lam - float(phi) * lam))


def gp_weights(
    path: MarketPath,
    *,
    lam_quad: float,
    gamma: float = 10.0,
    weight_cap: float = 2.0,
) -> np.ndarray:
    """Roll the GP rule through a path from a flat start; returns ``[T, N]`` weights.

    The rule reads the oracle signal (feature 0). As in the original implementation, the
    lagged position is carried in float64 while each emitted weight is rounded to float32
    (the baseline-policy interface returns float32 actions); this keeps results bit-exact.
    """
    b = signal_loading(path.ic, path.vol)
    sigma_eps = residual_std(path.ic, path.vol)
    g = _risk_curvature(gamma, sigma_eps)
    rate = gp_trading_rate(gamma=gamma, sigma_eps=sigma_eps, lam=lam_quad)
    aim_mult = gp_aim_multiplier(gamma=gamma, sigma_eps=sigma_eps, lam=lam_quad, phi=path.phi)
    cap = abs(float(weight_cap))

    out = np.zeros((path.n_periods, path.n_assets), dtype=np.float64)
    prev = np.zeros((path.n_assets,), dtype=np.float64)
    for t in range(path.n_periods):
        s = path.features[t, :, 0]
        s = np.where(np.isfinite(s), s, 0.0)
        markowitz = (float(b) * s) / g
        new = np.clip(prev + rate * (aim_mult * markowitz - prev), -cap, cap)
        prev = new
        out[t] = new.astype(np.float32)
    return out


@dataclass(frozen=True)
class WeightPathScore:
    """Accounting of one weight path: ``pnl = gross_pnl - quadratic_cost``."""

    pnl: float
    gross_pnl: float
    quadratic_cost: float
    turnover_l1: float
    turnover_l2_squared: float
    n_scored_actions: int


def evaluate_weight_path(
    weights: np.ndarray,
    returns: np.ndarray,
    lam_quad: float,
    *,
    initial_weights: np.ndarray | None = None,
) -> WeightPathScore:
    """Score a causal weight path in weight space (simple returns, no compounding)::

        pnl = sum_{t=0}^{T-2} [ w[t] . r[t+1] - (lam/2) ||w[t] - w[t-1]||^2 ],

    with ``w[-1] = initial_weights`` (flat by default). The last row earns nothing.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if w.shape != r.shape:
        raise ValueError(f"weights {w.shape} and returns {r.shape} must match")
    if w.shape[0] < 2:
        return WeightPathScore(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    active = w[:-1]
    gross = float(np.sum(active * r[1:]))
    prev0 = (
        np.zeros(w.shape[1])
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64).reshape(-1)
    )
    deltas = np.empty_like(active)
    deltas[0] = active[0] - prev0
    deltas[1:] = active[1:] - active[:-1]
    l2 = float(np.sum(deltas * deltas))
    cost = 0.5 * float(lam_quad) * l2
    return WeightPathScore(
        pnl=gross - cost,
        gross_pnl=gross,
        quadratic_cost=cost,
        turnover_l1=float(np.sum(np.abs(deltas))),
        turnover_l2_squared=l2,
        n_scored_actions=int(active.shape[0]),
    )
