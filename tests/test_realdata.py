"""Fast synthetic checks of the real-data accounting (no market data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from catch_alpha.realdata import panel as P
from catch_alpha.realdata import signals as G
from catch_alpha.realdata import strategies as S
from catch_alpha.realdata.build_panel import funding_features


def make_panel(
    prices: np.ndarray,
    funding: np.ndarray | None = None,
    start: str = "2020-01-01T00:00:00Z",
    n_train: int | None = None,
) -> P.Panel:
    prices = np.asarray(prices, dtype=float)
    n_t, n_a = prices.shape
    stamps = pd.date_range(start, periods=n_t, freq="1h", tz="UTC")
    funding = np.zeros((n_t, n_a)) if funding is None else np.asarray(funding, dtype=float)
    features = np.zeros((n_t, n_a, 1))
    features[:, :, 0] = funding
    return P.Panel(
        timestamps=np.array([t.isoformat() for t in stamps]),
        symbols=tuple(f"A{i}" for i in range(n_a)),
        prices=prices,
        funding=funding,
        features=features,
        feature_names=("funding_rate_last",),
        tradable=prices > 0,
        n_train=n_t if n_train is None else n_train,
        splits=("train",),
    )


# ---------------------------------------------------------------------------
# Funding sign and timing
# ---------------------------------------------------------------------------


def test_short_position_collects_positive_funding_on_position_held_into_the_bar() -> None:
    n_t = 17
    rets = np.zeros((n_t, 1))
    rates = np.full((n_t, 1), 0.001)
    is_funding = np.zeros(n_t, dtype=bool)
    is_funding[[8, 16]] = True
    is_rebalance = np.zeros(n_t, dtype=bool)
    is_rebalance[[0, 8]] = True
    # short 0.5 from bar 0, flat from bar 8 on
    book = S.run_book(
        rets,
        lambda t, _w: np.array([-0.5 if t == 0 else 0.0]),
        is_rebalance,
        0.0,
        rates,
        is_funding,
    )
    # the bar-8 payment goes to the short held INTO bar 8 (output index 7) ...
    assert book.funding[7] == pytest.approx(0.0005)
    # ... and the book is flat into bar 16, so nothing is paid there
    assert book.funding[15] == 0.0
    assert book.funding.sum() == pytest.approx(0.0005)


def test_carry_weights_short_positive_long_negative_and_partial_sides() -> None:
    rates = np.array([0.003, 0.002, 0.001, 0.0005, -0.001, 0.0])
    w = S.carry_weights(rates, np.ones(6, dtype=bool), k=3)
    assert np.allclose(w, [-1 / 6, -1 / 6, -1 / 6, 0.0, 1 / 6, 0.0])  # one negative name only
    w = S.carry_weights(rates, np.array([False, True, True, True, True, True]), k=3)
    assert w[0] == 0.0 and w[3] == pytest.approx(-1 / 6)  # untradable names are skipped


def test_raw_carry_funding_term_equals_minus_weight_times_rate() -> None:
    n_t = 25
    prices = np.full((n_t, 2), 100.0)
    funding = np.tile([0.001, -0.002], (n_t, 1))
    book = S.raw_carry(make_panel(prices, funding), k=1, cost_bps=0.0)
    # short asset 0 (+0.001), long asset 1 (-0.002): each funding bar pays 0.5*0.001 + 0.5*0.002
    assert book.funding[[7, 15, 23]] == pytest.approx([0.0015] * 3)
    assert np.count_nonzero(book.funding) == 3


# ---------------------------------------------------------------------------
# Weight / return alignment and costs
# ---------------------------------------------------------------------------


def test_weights_earn_the_next_return_not_the_current_one() -> None:
    rets = np.zeros((4, 1))
    rets[1, 0], rets[2, 0] = 0.10, 0.05
    reb = np.array([False, True, False, False])
    book = S.run_book(rets, lambda t, _w: np.array([1.0]), reb, 0.0)
    # bought at bar 1 -> misses r[1] = 10%, earns r[2] = 5%
    assert book.price.tolist() == pytest.approx([0.0, 0.05, 0.0])


def test_costs_are_charged_on_turnover_and_initial_build_lands_on_first_bar() -> None:
    rets = np.zeros((5, 2))
    reb = np.array([True, False, True, False, False])
    targets = {0: np.array([0.5, -0.5]), 2: np.array([-0.5, 0.5])}
    book = S.run_book(rets, lambda t, _w: targets[t], reb, cost_bps=5.0)
    assert book.cost[0] == pytest.approx(1.0 * 5e-4)  # build at bar 0, charged on bar 1
    assert book.cost[1] == pytest.approx(2.0 * 5e-4)  # full flip at bar 2
    assert book.turnover.sum() == pytest.approx(3.0)
    assert book.total.sum() == pytest.approx(-3.0 * 5e-4)


def test_reversal_is_causal_and_mirror_trend_has_opposite_gross_pnl() -> None:
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(200, 8)), axis=0))
    pan = make_panel(prices)
    rev = S.reversal(pan, rebalance_hours=8)
    trend = S.reversal(pan, rebalance_hours=8, trend=True)
    assert np.allclose(rev.price, -trend.price)
    bumped = prices.copy()
    bumped[150:, 0] *= 3.0  # change the future only
    rev2 = S.reversal(make_panel(bumped), rebalance_hours=8)
    assert np.array_equal(rev.price[:149], rev2.price[:149])


def test_reversal_weights_long_losers_short_winners_dollar_neutral() -> None:
    w = S.reversal_weights(np.array([0.05, -0.02, 0.01, -0.04]), np.ones(4, dtype=bool), k=1)
    assert np.allclose(w, [-0.5, 0.0, 0.0, 0.5]) and w.sum() == 0.0


# ---------------------------------------------------------------------------
# Cash-and-carry and the hedge
# ---------------------------------------------------------------------------


def test_cash_and_carry_is_delta_neutral_and_pays_fees_on_both_legs() -> None:
    n_t = 9
    perp = np.tile(np.linspace(100, 108, n_t)[:, None], (1, 2))
    spot = perp * 0.999
    funding = np.tile([0.001, -0.001], (n_t, 1))
    pan = make_panel(perp, funding)
    book = S.cash_and_carry(pan, spot, np.ones_like(spot, dtype=bool), cost_bps=5.0)
    # only asset 0 is active: legs of 1/2 each -> perp and spot moves cancel exactly
    assert np.allclose(book.price, 0.0)
    assert book.funding[7] == pytest.approx(0.5 * 0.001)
    assert book.cost[0] == pytest.approx(1.0 * 5e-4)  # 0.5 perp + 0.5 spot traded


def test_hedge_beta_uses_training_returns_only() -> None:
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(400, 6)), axis=0))
    funding = rng.normal(0, 1e-4, size=(400, 6))
    a = S.hedged_carry(make_panel(prices, funding, n_train=200))
    altered = prices.copy()
    altered[250:] *= np.exp(np.cumsum(rng.normal(0, 0.05, size=(150, 6)), axis=0))
    b = S.hedged_carry(make_panel(altered, funding, n_train=200))
    assert a.info["hedge_beta"] == b.info["hedge_beta"]
    assert np.array_equal(a.funding, b.funding)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_strategy_folds_have_168_hour_gaps_on_the_paper_window() -> None:
    folds = P.strategy_folds(46_590)
    assert len(folds) == 6
    assert folds[0][0] == 168 and all(f.size == 7_597 for f in folds)
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert nxt[0] - prev[-1] - 1 == 168
    assert folds[-1][-1] == 46_589


def test_signal_folds_drop_the_first_168_rows_of_each_block() -> None:
    folds = P.signal_folds(46_591)
    edges = np.linspace(0, 46_591, 7, dtype=np.int64)
    assert [f[0] for f in folds] == [int(e) + 168 for e in edges[:-1]]
    assert [f[1] for f in folds] == [int(e) for e in edges[1:]]


# ---------------------------------------------------------------------------
# Signal statistics and funding features
# ---------------------------------------------------------------------------


def test_rank_ic_matches_scipy_and_skips_thin_cross_sections() -> None:
    rng = np.random.default_rng(3)
    sig, fut = rng.normal(size=(50, 7)), rng.normal(size=(50, 7))
    sig[5, :5] = np.nan  # only two valid assets -> NaN
    ic = G.rank_ic_series(sig, fut)
    assert np.isnan(ic[5])
    for t in (0, 10, 49):
        assert ic[t] == pytest.approx(spearmanr(sig[t], fut[t]).statistic)


def test_newey_west_with_zero_lags_is_the_iid_t_statistic() -> None:
    x = np.random.default_rng(4).normal(0.1, 1.0, size=500)
    nw = G.newey_west_mean(x, lags=0)
    assert nw["tstat"] == pytest.approx(x.mean() / (x.std() / np.sqrt(x.size)))


def test_funding_rate_becomes_visible_at_its_event_bar_not_before() -> None:
    cal = pd.date_range("2024-01-01T06:00:00Z", periods=4, freq="1h", tz="UTC")
    events = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-01T08:00:00Z"]), "funding_rate": [0.0003]}
    )
    out = funding_features(cal, events)
    assert out["funding_rate_last"].tolist() == [0.0, 0.0, 0.0003, 0.0003]
    assert out["hours_to_next_funding"].tolist() == [2.0, 1.0, 8.0, 7.0]


def test_buy_and_hold_drifts_without_rebalancing() -> None:
    prices = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]])
    out = S.buy_and_hold(make_panel(prices), cost_bps=0.0)
    # 50/50 -> asset 0 doubles (+50%), weights drift to 2/3, 1/3 -> asset 1 doubles (+1/3)
    assert out.tolist() == pytest.approx([0.5, 1.0 / 3.0])
