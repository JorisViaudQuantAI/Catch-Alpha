"""The four frozen portfolio rules of the real-data study, plus benchmarks and baselines.

All rules were fixed before evaluation and are not tuned here. Each one produces a
per-hour net return series together with its components, so that a reader can see
where the money comes from (price moves, funding, the perpetual-spot basis) and
where it goes (trading fees).

Accounting conventions (shared by every book)
---------------------------------------------
Let ``w[t]`` be the signed portfolio weights chosen at the close of bar ``t``
(positive = long, as a fraction of capital) and ``r[t] = P[t]/P[t-1] - 1`` the
return from bar ``t-1`` to bar ``t``. Then, for every bar ``t >= 1``::

    price[t]    = sum_i w_i[t-1] * r_i[t]          weights earn the NEXT bar's return
    funding[t]  = sum_i -w_i[t-1] * rate_i[t]      only at 00/08/16 UTC funding bars
    cost[t]     = cost_bps * 1e-4 * sum_i |w_i[t] - w_i[t-1]|
    total[t]    = price[t] + funding[t] - cost[t]

* **Weights vs returns.** A weight decided with information at ``t`` is only
  exposed to ``r[t+1]``; it can never earn the return it was computed from.
* **Funding sign.** Binance's rule is that longs pay shorts when the rate is
  positive, so a position with signed weight ``w`` receives ``-w * rate``. To
  *collect* funding you short contracts with positive rates and buy contracts with
  negative rates. The payment at a funding hour goes to the position held *into*
  that hour, ``w[t-1]``.
* **Costs.** Every unit of weight traded pays the frozen taker fee of 5 bp
  (0.05%). A two-legged trade (perpetual and spot) pays on both legs. If the book
  is built at the very first bar, that build cost is charged on the first return.

All series returned below have length ``T - 1`` and are aligned with returns
``r[1], ..., r[T-1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from catch_alpha.realdata.panel import FUNDING_HOURS, Panel, simple_returns

COST_BPS = 5.0  # frozen taker fee per leg, in basis points
K_PER_SIDE = 3  # three of twelve assets per side (the extreme quarter)
CARRY_REBALANCE_HOURS = 8
REVERSAL_LOOKBACK_HOURS = 24
REVERSAL_REBALANCE_VARIANTS: tuple[int, ...] = (24, 8)
MOMENTUM_BASELINE_LOOKBACK_HOURS = 168

TargetFn = Callable[[int, np.ndarray], "np.ndarray | None"]


@dataclass
class Book:
    """Per-hour returns of one portfolio and their decomposition (all length ``T-1``).

    ``price`` is the PnL from price moves; for the delta-neutral cash-and-carry book
    it is the *basis* term (perpetual vs spot drift). ``cost`` is a positive number
    that is subtracted: ``total = price + funding - cost``.
    """

    total: np.ndarray
    price: np.ndarray
    funding: np.ndarray
    cost: np.ndarray
    turnover: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def sums(self) -> dict[str, float]:
        """Component totals over the whole series (costs reported as a negative)."""
        return {
            "price": float(np.sum(self.price)),
            "funding": float(np.sum(self.funding)),
            "cost": -float(np.sum(self.cost)),
            "total": float(np.sum(self.total)),
        }


# ---------------------------------------------------------------------------
# Generic book engine
# ---------------------------------------------------------------------------


def run_book(
    asset_returns: np.ndarray,
    target: TargetFn,
    is_rebalance: np.ndarray,
    cost_bps: float = COST_BPS,
    funding_rates: np.ndarray | None = None,
    is_funding_bar: np.ndarray | None = None,
) -> Book:
    """Simulate a rebalanced book bar by bar (see module docstring for the rules).

    ``target(t, w_prev)`` is called at rebalance bars and returns the new weights,
    or ``None`` to keep the current ones. It must only use information available at
    bar ``t``. ``funding_rates`` (same shape as ``asset_returns``) is paid at funding
    bars; columns without funding (e.g. spot legs) should hold 0.
    """
    rets = np.asarray(asset_returns, dtype=np.float64)
    n_t, n_a = rets.shape
    cost_rate = float(cost_bps) * 1e-4
    out = {k: np.zeros(n_t - 1) for k in ("total", "price", "funding", "cost", "turnover")}

    # Build the book at bar 0 if it is a rebalance bar; its cost lands on bar 1.
    w_prev = np.zeros(n_a)
    if n_t >= 1 and is_rebalance[0]:
        w0 = target(0, w_prev)
        w_prev = np.zeros(n_a) if w0 is None else np.asarray(w0, dtype=np.float64)
    initial_turnover = float(np.abs(w_prev).sum())

    for t in range(1, n_t):
        r_t = np.where(np.isfinite(rets[t]), rets[t], 0.0)
        p_term = float(np.dot(w_prev, r_t))  # w[t-1] earns r[t]
        f_term = 0.0
        if funding_rates is not None and is_funding_bar is not None and is_funding_bar[t]:
            rate = np.where(np.isfinite(funding_rates[t]), funding_rates[t], 0.0)
            f_term = float(np.dot(-w_prev, rate))  # longs pay a positive rate
        w_new = w_prev
        if is_rebalance[t]:
            proposal = target(t, w_prev)
            if proposal is not None:
                w_new = np.asarray(proposal, dtype=np.float64)
        turnover = float(np.abs(w_new - w_prev).sum())
        cost = cost_rate * turnover
        if t == 1:
            cost += cost_rate * initial_turnover
            turnover += initial_turnover
        out["total"][t - 1] = p_term + f_term - cost
        out["price"][t - 1] = p_term
        out["funding"][t - 1] = f_term
        out["cost"][t - 1] = cost
        out["turnover"][t - 1] = turnover
        w_prev = w_new
    return Book(**out)


def _rebalance_hours_mask(panel: Panel, every_hours: int) -> np.ndarray:
    """True at bars whose UTC hour is a multiple of ``every_hours`` (24 -> 00:00 only)."""
    return (panel.hours % int(every_hours)) == 0


# ---------------------------------------------------------------------------
# 1. Raw funding carry (H104)
# ---------------------------------------------------------------------------


def carry_weights(funding: np.ndarray, tradable: np.ndarray, k: int = K_PER_SIDE) -> np.ndarray:
    """Short the ``k`` most positive funding rates, buy the ``k`` most negative.

    Each selected name gets weight ``1/(2k)`` (gross 1.0 when both sides are full).
    If fewer than ``k`` contracts have a given sign, that side is left partly empty
    rather than filled with contracts that would *pay* funding, so the book can be
    net long or net short.
    """
    rate = np.asarray(funding, dtype=np.float64).reshape(-1)
    ok = np.flatnonzero(np.asarray(tradable, dtype=bool).reshape(-1) & np.isfinite(rate))
    w = np.zeros(rate.size)
    per_name = 1.0 / (2.0 * k)
    pos, neg = ok[rate[ok] > 0.0], ok[rate[ok] < 0.0]
    w[pos[np.argsort(rate[pos])[::-1]][:k]] = -per_name  # most positive first -> short
    w[neg[np.argsort(rate[neg])][:k]] = per_name  # most negative first -> long
    return w


def raw_carry(
    panel: Panel,
    k: int = K_PER_SIDE,
    rebalance_hours: int = CARRY_REBALANCE_HOURS,
    cost_bps: float = COST_BPS,
) -> Book:
    """H104: hold the funding-collecting book, rebalanced at every funding hour.

    The book collects funding but stays exposed to the price moves of the names it
    holds (it is not market neutral).
    """
    reb_hours = [h for h in FUNDING_HOURS if h % int(rebalance_hours) == 0]
    is_rebalance = np.isin(panel.hours, reb_hours)
    return run_book(
        simple_returns(panel.prices),
        lambda t, _w: carry_weights(panel.funding[t], panel.tradable[t], k),
        is_rebalance,
        cost_bps,
        funding_rates=panel.funding,
        is_funding_bar=panel.is_funding_bar,
    )


# ---------------------------------------------------------------------------
# 2. Beta-hedged carry (H105)
# ---------------------------------------------------------------------------


def equal_weight_market_returns(panel: Panel) -> np.ndarray:
    """``[T]`` average return of the tradable contracts at each bar (0 if none)."""
    rets = simple_returns(panel.prices)
    rets = np.where(np.isfinite(rets), rets, 0.0)
    count = panel.tradable.sum(axis=1)
    summed = np.where(panel.tradable, rets, 0.0).sum(axis=1)
    out = np.where(count > 0, summed / np.maximum(count, 1), 0.0)
    out[0] = 0.0
    return out


def hedged_carry(
    panel: Panel,
    k: int = K_PER_SIDE,
    rebalance_hours: int = CARRY_REBALANCE_HOURS,
    cost_bps: float = COST_BPS,
    beta_fit_rows: int | None = None,
) -> Book:
    """H105: raw carry plus a market hedge whose size is estimated on training data only.

    The carry book's price PnL is regressed on the equal-weight market return
    ``m_t`` over the training returns (``beta = cov / var``), and the overlay
    ``-beta * m_t`` is added to every bar, including validation and test bars.
    Fitting on training data only keeps the hedge free of look-ahead: later data
    never influences the coefficient that is applied to it.

    The overlay is modelled as a spot basket, so it adds no funding and the funding
    term equals that of the raw carry book; its one-off build cost ``|beta|`` is
    charged on the first bar. ``beta_fit_rows`` defaults to ``panel.n_train``.
    """
    raw = raw_carry(panel, k, rebalance_hours, cost_bps)
    market = equal_weight_market_returns(panel)[1:]  # aligned with returns r[1..T-1]
    fit_rows = panel.n_train if beta_fit_rows is None else int(beta_fit_rows)
    fit_end = max(0, min(fit_rows - 1, raw.price.size))  # train rows -> train returns
    x, y = market[:fit_end], raw.price[:fit_end]
    var_m = float(np.var(x)) if x.size else 0.0
    beta = 0.0
    if x.size >= 2 and var_m > 1e-18:
        beta = float(np.cov(y, x, bias=True)[0, 1] / var_m)
        beta = beta if np.isfinite(beta) else 0.0
    hedge = -beta * market
    hedge_turnover = np.zeros_like(raw.cost)
    if hedge_turnover.size:
        hedge_turnover[0] = abs(beta)
    cost = raw.cost + float(cost_bps) * 1e-4 * hedge_turnover
    price = raw.price + hedge
    return Book(
        total=price + raw.funding - cost,
        price=price,
        funding=raw.funding,
        cost=cost,
        turnover=raw.turnover + hedge_turnover,
        info={"hedge_beta": beta, "beta_fit_returns": int(fit_end), "raw_price": raw.price},
    )


def funding_share(book: Book) -> float:
    """``|funding| / (|price| + |funding|)`` over the whole series.

    The pre-registered requirement for the hedged carry was a share above 0.6:
    a carry result should come mostly from funding, not from price moves.
    """
    p, f = abs(float(np.sum(book.price))), abs(float(np.sum(book.funding)))
    return f / (p + f) if p + f > 1e-18 else 0.0


# ---------------------------------------------------------------------------
# 3. Short-horizon reversal (H106) and its mirror trend book
# ---------------------------------------------------------------------------


def trailing_return(prices: np.ndarray, t: int, lookback: int) -> np.ndarray:
    """``P[t]/P[t-lookback] - 1`` per asset (NaN if unavailable). Uses only data up to t."""
    n_a = prices.shape[1]
    out = np.full(n_a, np.nan)
    if t < lookback:
        return out
    p0, p1 = prices[t - lookback], prices[t]
    ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
    out[ok] = p1[ok] / p0[ok] - 1.0
    return out


def reversal_weights(
    signal: np.ndarray, tradable: np.ndarray, k: int = K_PER_SIDE, trend: bool = False
) -> np.ndarray:
    """Long the ``k`` lowest past returns and short the ``k`` highest (``trend`` flips it).

    Weight ``1/(2k)`` per name, dollar neutral. With fewer than ``2k`` rankable
    contracts both sides shrink to ``n // 2`` names so the book stays balanced.
    """
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    ok = np.flatnonzero(np.asarray(tradable, dtype=bool).reshape(-1) & np.isfinite(sig))
    w = np.zeros(sig.size)
    if ok.size < 2:
        return w
    order = ok[np.argsort(sig[ok], kind="stable")]  # ascending: biggest losers first
    leg = min(int(k), order.size // 2)
    losers, winners = order[:leg], order[order.size - leg :]
    longs, shorts = (winners, losers) if trend else (losers, winners)
    per_name = 1.0 / (2.0 * k)
    w[longs] += per_name
    w[shorts] -= per_name
    return w


def reversal(
    panel: Panel,
    rebalance_hours: int = 24,
    k: int = K_PER_SIDE,
    lookback_hours: int = REVERSAL_LOOKBACK_HOURS,
    cost_bps: float = COST_BPS,
    trend: bool = False,
) -> Book:
    """H106: rank by 24-hour return; buy recent losers and short recent winners.

    Two frozen rebalancing variants are reported side by side: every 24 hours
    (00:00 UTC) and every 8 hours. ``trend=True`` gives the mirror portfolio (long
    winners, short losers), whose gross price PnL is exactly the negative.
    Perpetual funding is ignored here: this is a pure price strategy.
    """
    return run_book(
        simple_returns(panel.prices),
        lambda t, _w: reversal_weights(
            trailing_return(panel.prices, t, lookback_hours), panel.tradable[t], k, trend
        ),
        _rebalance_hours_mask(panel, rebalance_hours),
        cost_bps,
    )


# ---------------------------------------------------------------------------
# 4. Delta-neutral cash-and-carry (H107)
# ---------------------------------------------------------------------------


def cash_and_carry_weights(
    funding: np.ndarray, perp_tradable: np.ndarray, spot_listed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per asset with positive funding: short the perpetual and buy the same notional spot.

    Assets with zero or negative funding are left flat (harvesting them would
    require borrowing spot). With ``m`` active assets each leg gets ``1/(2m)``, so
    gross exposure is 1.0 and the net price exposure is zero by construction.
    """
    rate = np.asarray(funding, dtype=np.float64).reshape(-1)
    active = (rate > 0.0) & np.isfinite(rate) & perp_tradable & spot_listed
    wp, ws = np.zeros(rate.size), np.zeros(rate.size)
    m = int(active.sum())
    if m:
        wp[active], ws[active] = -1.0 / (2.0 * m), 1.0 / (2.0 * m)
    return wp, ws


def cash_and_carry(
    panel: Panel,
    spot_prices: np.ndarray,
    spot_observed: np.ndarray,
    cost_bps: float = COST_BPS,
) -> Book:
    """H107: the textbook funding harvest, rebalanced at every funding hour.

    The book is run on ``2N`` columns (``N`` perpetual legs then ``N`` spot legs).
    Only the perpetual legs pay or receive funding. Because the two legs offset,
    the price term is the *basis* drift ``sum (1/2m) * (r_spot - r_perp)``, and the
    fee is paid on both legs.

    ``spot_prices`` / ``spot_observed`` are ``[T, N]`` spot closes aligned to the
    perpetual timestamps (see :func:`catch_alpha.realdata.build_panel.load_spot_panel`).
    An asset is spot-eligible from its first observed spot bar onward.
    """
    n = panel.prices.shape[1]
    spot_listed = np.cumsum(np.asarray(spot_observed, dtype=np.int64), axis=0) > 0
    rets = np.concatenate([simple_returns(panel.prices), simple_returns(spot_prices)], axis=1)
    rates = np.concatenate([panel.funding, np.zeros_like(panel.funding)], axis=1)

    def target(t: int, _w: np.ndarray) -> np.ndarray:
        wp, ws = cash_and_carry_weights(panel.funding[t], panel.tradable[t], spot_listed[t])
        return np.concatenate([wp, ws])

    book = run_book(rets, target, panel.is_funding_bar, cost_bps, rates, panel.is_funding_bar)
    book.info["n_assets"] = n
    return book


# ---------------------------------------------------------------------------
# Benchmarks and baselines
# ---------------------------------------------------------------------------


def cash(panel: Panel) -> np.ndarray:
    """The cash benchmark: zero return every hour."""
    return np.zeros(panel.n_returns)


def buy_and_hold(panel: Panel, cost_bps: float = COST_BPS) -> np.ndarray:
    """Equal-weight buy-and-hold of the contracts tradable at the first bar.

    Bought once, then left to drift with prices; names that later become
    untradable are sold (and pay the fee). Contracts that list later are not
    added. Funding is ignored, as in the original benchmark.
    """
    rets = simple_returns(panel.prices)
    n_t, n_a = rets.shape
    cost_rate = float(cost_bps) * 1e-4
    out = np.zeros(n_t - 1)
    w = np.zeros(n_a)
    started, initial_turnover = False, 0.0
    if panel.tradable[0].any():
        w[panel.tradable[0]] = 1.0 / panel.tradable[0].sum()
        started, initial_turnover = True, float(np.abs(w).sum())
    for t in range(1, n_t):
        r_t = np.where(np.isfinite(rets[t]), rets[t], 0.0)
        p_term = float(np.dot(w, r_t))
        value = 1.0 + p_term
        turnover = 0.0
        if started:
            w = w * (1.0 + r_t) / value if np.isfinite(value) and value > 1e-12 else np.zeros(n_a)
            gone = ~panel.tradable[t]
            if gone.any():
                turnover += float(np.abs(w[gone]).sum())
                w[gone] = 0.0
        elif panel.tradable[t].any():  # nothing tradable at bar 0: enter at first chance
            w = np.zeros(n_a)
            w[panel.tradable[t]] = 1.0 / panel.tradable[t].sum()
            turnover += float(np.abs(w).sum())
            started = True
        if t == 1:
            turnover += initial_turnover
        out[t - 1] = p_term - cost_rate * turnover
    return out


def momentum_top3(
    panel: Panel,
    lookback_hours: int = MOMENTUM_BASELINE_LOOKBACK_HOURS,
    rebalance_hours: int = 24,
    top_k: int = 3,
    cost_bps: float = COST_BPS,
) -> np.ndarray:
    """Baseline for raw carry: long the three best 168-hour performers, daily at 00:00."""

    def target(t: int, _w: np.ndarray) -> np.ndarray | None:
        if t < lookback_hours:
            return None  # not enough history yet: keep the current (empty) book
        mom = np.full(panel.prices.shape[1], -np.inf)
        past = trailing_return(panel.prices, t, lookback_hours)
        ok = panel.tradable[t] & np.isfinite(past)
        mom[ok] = past[ok]
        chosen = [int(a) for a in np.argsort(mom)[::-1] if np.isfinite(mom[a])][:top_k]
        w = np.zeros(mom.size)
        if chosen:
            w[chosen] = 1.0 / len(chosen)
        return w

    is_rebalance = _rebalance_hours_mask(panel, rebalance_hours)
    return run_book(simple_returns(panel.prices), target, is_rebalance, cost_bps).total


# ---------------------------------------------------------------------------
# Strategy registry: candidate, benchmark and declared baselines
# ---------------------------------------------------------------------------

STRATEGY_IDS: tuple[str, ...] = (
    "raw_carry",
    "hedged_carry",
    "reversal_24h",
    "reversal_8h",
    "cash_and_carry",
)


def build_strategy(
    name: str,
    panel: Panel,
    spot_prices: np.ndarray | None = None,
    spot_observed: np.ndarray | None = None,
    cost_bps: float = COST_BPS,
) -> dict[str, Any]:
    """Return ``{"book", "candidate", "benchmark", "baselines"}`` for one frozen rule.

    Every strategy is compared with cash (the benchmark) and with equal-weight
    buy-and-hold. In addition: raw carry vs 168-hour momentum; hedged carry and
    cash-and-carry vs raw carry; reversal vs its mirror trend book.
    """
    bh = buy_and_hold(panel, cost_bps)
    if name == "raw_carry":
        book = raw_carry(panel, cost_bps=cost_bps)
        baselines = {"buy_hold_equal_weight": bh, "momentum_168h_top3": momentum_top3(panel)}
    elif name == "hedged_carry":
        book = hedged_carry(panel, cost_bps=cost_bps)
        baselines = {"raw_carry": raw_carry(panel, cost_bps=cost_bps).total}
        baselines["buy_hold_equal_weight"] = bh
    elif name in ("reversal_24h", "reversal_8h"):
        every = 24 if name == "reversal_24h" else 8
        book = reversal(panel, every, cost_bps=cost_bps)
        trend = reversal(panel, every, cost_bps=cost_bps, trend=True)
        book.info["trend"] = trend
        baselines = {"momentum_24h_trend": trend.total, "buy_hold_equal_weight": bh}
    elif name == "cash_and_carry":
        if spot_prices is None or spot_observed is None:
            raise ValueError("cash_and_carry needs the spot panel (spot_prices, spot_observed)")
        book = cash_and_carry(panel, spot_prices, spot_observed, cost_bps)
        baselines = {"raw_carry": raw_carry(panel, cost_bps=cost_bps).total}
        baselines["buy_hold_equal_weight"] = bh
    else:
        raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGY_IDS}")
    return {"book": book, "candidate": book.total, "benchmark": cash(panel), "baselines": baselines}
