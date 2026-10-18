"""Regenerate every paper-facing real-data artifact from the hourly panel.

Each public function returns plain Python data (dicts / lists of dicts) that the
CLI scripts in ``scripts/real_data_*.py`` write to ``results/real_data/``.

* :func:`fold_report`            -- six-fold mechanics of one book (CIs, components)
* :func:`strategy_study`         -- the four frozen strategies on train+valid
* :func:`carry_decomposition`    -- the "carry economics" table / figure S1 data
* :func:`fee_sensitivity`        -- cash-and-carry net return vs fee (Figure 4)
* :func:`gate_verdicts`          -- the full evaluator on each strategy ("gate outcomes")
* :func:`pbo_reversal`           -- PBO on the 2 x 6 reversal score matrix
* :func:`final_period`           -- the single look at the locked 2025-2026 period
* :func:`weekly_momentum_benchmark`, :func:`market_null_check` -- the weekly
  momentum example of Section 5 and the market-derived null of Section 3

Statistical helpers (power gate, stationary bootstrap, PBO) and the evaluator
come from :mod:`catch_alpha.stats` and :mod:`catch_alpha.gate`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from catch_alpha import gate, stats
from catch_alpha.realdata import strategies as S
from catch_alpha.realdata.panel import PERIODS_PER_YEAR, Panel, strategy_folds

BOOTSTRAP_REPS = 400  # resamples for every fold-level mean-return interval
AUDITED_N_TRIALS = 347  # real-data evaluations on record before the gate re-emissions
FEE_GRID_BPS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
# Fee grid labels exactly as in the paper's cash_and_carry_fee_sensitivity.csv.
FEE_LABELS: dict[float, tuple[str, str]] = {
    0.0: ("cash_and_carry_v3_cost0bps", "none"),
    1.0: ("cash_and_carry_v3_cost1bps", "taker"),
    2.0: ("cash_and_carry_v3_maker2bps", "maker"),
    3.0: ("cash_and_carry_v3_cost3bps", "taker"),
    5.0: ("cash_and_carry_v3", "taker"),
    10.0: ("cash_and_carry_v3_cost10bps", "taker"),
}
FINAL_PERIOD_BOOKS: tuple[str, ...] = (
    "raw_carry",
    "hedged_carry",
    "reversal_24h",
    "cash_and_carry",
)


def annualized_sharpe(returns: np.ndarray, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Mean / population std, times sqrt(periods per year); 0 for a flat series."""
    arr = np.asarray(returns, dtype=np.float64)
    sd = float(np.std(arr)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(np.mean(arr)) / sd * math.sqrt(periods_per_year)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


# ---------------------------------------------------------------------------
# Fold mechanics
# ---------------------------------------------------------------------------


def fold_report(
    book: S.Book,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Six-fold report of one book: bootstrap CI of the mean return and its components.

    For fold ``i`` the 95% interval of the mean hourly net return comes from a
    stationary bootstrap with ``reps`` resamples and seed ``seed + i``. A Sharpe
    ratio is reported only when the fold passes the power gate (it never does on
    7,597 hours); ``annualized_sharpe_ungated`` is the plain number, used only to
    fill the PBO score matrix.
    """
    folds = []
    for i, idx in enumerate(strategy_folds(book.total.size)):
        seg = book.total[idx]
        lo, hi = stats.stationary_bootstrap_ci(seg, _mean, reps=reps, seed=seed + i)
        power = stats.power_gate(seg.size, periods_per_year)
        p, f = float(np.sum(book.price[idx])), float(np.sum(book.funding[idx]))
        sharpe = annualized_sharpe(seg, periods_per_year)
        folds.append(
            {
                "fold": i,
                "n_obs": int(seg.size),
                "start": int(idx[0]),
                "end": int(idx[-1] + 1),
                "mean_return_ci_95": [lo, hi],
                "ci_lb_gt_zero": bool(lo > 0.0),
                "ci_ub_lt_zero": bool(hi < 0.0),
                "sum_price": p,
                "sum_funding": f,
                "sum_cost": -float(np.sum(book.cost[idx])),
                "sum_total": float(np.sum(seg)),
                "funding_share": abs(f) / (abs(p) + abs(f)) if abs(p) + abs(f) > 1e-18 else 0.0,
                "turnover_mean": float(np.mean(book.turnover[idx])),
                "power_gate": power["status"],
                "annualized_sharpe": sharpe if power["status"] == "ok" else None,
                "annualized_sharpe_ungated": sharpe,
            }
        )
    return {
        "n_returns": int(book.total.size),
        "full_window": book.sums(),
        "full_window_power_gate": stats.power_gate(book.total.size, periods_per_year),
        "n_folds_ci_lb_gt_zero": sum(f["ci_lb_gt_zero"] for f in folds),
        "n_folds_funding_positive": sum(f["sum_funding"] > 0 for f in folds),
        "folds": folds,
    }


def strategy_study(
    panel: Panel,
    spot_prices: np.ndarray | None = None,
    spot_observed: np.ndarray | None = None,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Fold reports and headline numbers for the frozen strategies on train+valid.

    Cash-and-carry is skipped (and reported as such) when no spot panel is given.
    """
    out: dict[str, Any] = {
        "window": {"start": str(panel.timestamps[1]), "end": str(panel.timestamps[-1])},
        "power_gate": stats.power_gate(panel.n_returns, PERIODS_PER_YEAR),
        "strategies": {},
    }
    for name in S.STRATEGY_IDS:
        if name == "cash_and_carry" and spot_prices is None:
            out["strategies"][name] = {"skipped": "no spot panel supplied"}
            continue
        built = S.build_strategy(name, panel, spot_prices, spot_observed)
        book: S.Book = built["book"]
        report = fold_report(book, reps=reps)
        if name == "hedged_carry":
            report["hedge_beta"] = book.info["hedge_beta"]
            report["beta_fit_returns"] = book.info["beta_fit_returns"]
            report["funding_share"] = S.funding_share(book)
            report["funding_share_requirement"] = 0.6
        if name.startswith("reversal"):
            report["gross_price_pnl"] = float(np.sum(book.price))
            report["mirror_trend_gross_price_pnl"] = float(np.sum(book.info["trend"].price))
        out["strategies"][name] = report
    return out


# ---------------------------------------------------------------------------
# Carry economics and fee sweep
# ---------------------------------------------------------------------------


def carry_decomposition(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the carry-economics figure (same schema as ``carry_decomposition.csv``).

    ``panel_a``: raw-carry funding received in each fold. ``panel_b``: cash-and-carry
    funding, basis, cost (negative) and net over the whole train+valid window.
    """
    rows = [
        {
            "panel": "panel_a",
            "fold": f["fold"],
            "component_kind": "funding",
            "label": f"fold {f['fold']}",
            "value": f"{f['sum_funding']:.6f}",
        }
        for f in study["strategies"]["raw_carry"]["folds"]
    ]
    cc = study["strategies"].get("cash_and_carry", {})
    if "full_window" in cc:
        fw = cc["full_window"]
        for kind, label, key in (
            ("funding", "funding income", "funding"),
            ("basis", "basis drift", "price"),
            ("cost", "double-leg cost (5 bps/leg)", "cost"),
            ("net", "net", "total"),
        ):
            rows.append(
                {
                    "panel": "panel_b",
                    "fold": "all",
                    "component_kind": kind,
                    "label": label,
                    "value": f"{fw[key]:.6f}",
                }
            )
    return rows


CARRY_DECOMPOSITION_HEADER = (
    "# figure: F4 (carry economics decomposition)",
    "# source: catch_alpha.realdata.study.carry_decomposition (scripts/real_data_strategies.py)",
    "# panel_a: raw funding carry, per-fold funding-leg sum (positive every fold).",
    "# panel_b: cash-and-carry three-way decomposition; net = funding + basis + cost.",
    "# component_kind: funding|basis|cost|net ; value in book PnL units over train+valid.",
)


def fee_sensitivity(
    panel: Panel,
    spot_prices: np.ndarray,
    spot_observed: np.ndarray,
    fees_bps: tuple[float, ...] = FEE_GRID_BPS,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Cash-and-carry over train+valid at several fees per leg, with trades unchanged.

    The fee does not change which positions are held, so funding and basis are
    identical across rows and the net return is linear in the fee. The break-even
    fee is ``(funding + basis) / (cost per basis point)``.
    """
    rows = []
    for fee in fees_bps:
        book = S.cash_and_carry(panel, spot_prices, spot_observed, cost_bps=fee)
        rep = fold_report(book, reps=reps)
        fw = rep["full_window"]
        rows.append(
            {
                "fee_bps": fee,
                "funding": fw["funding"],
                "basis": fw["price"],
                "cost": fw["cost"],
                "net": fw["total"],
                "n_folds_ci_lb_gt_zero": rep["n_folds_ci_lb_gt_zero"],
            }
        )
    turnover = float(np.sum(S.cash_and_carry(panel, spot_prices, spot_observed, 1.0).turnover))
    paper_rows = [
        {
            "fee_bps": r["fee_bps"],
            "label": FEE_LABELS.get(r["fee_bps"], ("", "taker"))[1],
            "run_dir": FEE_LABELS.get(r["fee_bps"], (f"cost{r['fee_bps']:g}bps", ""))[0],
            "full_window_sum_funding_term": repr(float(r["funding"])),
            "full_window_sum_basis_term": repr(float(r["basis"])),
            "full_window_sum_cost": repr(float(-r["cost"] + 0.0)),
            "full_window_sum_total": repr(float(r["net"])),
            "n_folds_mean_ci_lb_gt_zero": r["n_folds_ci_lb_gt_zero"],
        }
        for r in rows
    ]
    gross = rows[0]["funding"] + rows[0]["basis"]
    return {
        "rows": rows,
        "paper_rows": paper_rows,
        "cost_per_bp": turnover * 1e-4,
        "break_even_fee_bps": gross / (turnover * 1e-4),
    }


# ---------------------------------------------------------------------------
# Evaluator verdicts and PBO
# ---------------------------------------------------------------------------


def gate_verdicts(
    panel: Panel,
    spot_prices: np.ndarray | None = None,
    spot_observed: np.ndarray | None = None,
    n_trials: int = AUDITED_N_TRIALS,
) -> dict[str, Any]:
    """Run the four-check evaluator on each strategy's return series.

    ``default`` uses the frozen thresholds in decision mode with the audited trial
    count; on 46,590 hourly returns it stops at the power gate
    (``INSUFFICIENT_DATA``, since 54,160 are required for S_min = 1). For raw carry
    and cash-and-carry a documented diagnostic with ``s_min_annual = 1.5`` (and the
    small trial counts used at the time, 2 and 3) lets the remaining checks run;
    diagnostic verdicts can never accept a strategy.
    """
    diagnostics = {"raw_carry": 2, "cash_and_carry": 3}
    out: dict[str, Any] = {}
    for name in S.STRATEGY_IDS:
        if name == "cash_and_carry" and spot_prices is None:
            out[name] = {"skipped": "no spot panel supplied"}
            continue
        built = S.build_strategy(name, panel, spot_prices, spot_observed)
        args = (built["candidate"], built["benchmark"], built["baselines"], PERIODS_PER_YEAR)
        entry = {"default": gate.evaluate(*args, n_trials=n_trials, decision_mode=True)}
        if name in diagnostics:
            entry["s_min_1.5_diagnostic"] = gate.evaluate(
                *args, n_trials=diagnostics[name], thresholds={"s_min_annual": 1.5}
            )
        out[name] = entry
    return out


def pbo_reversal(panel: Panel, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    """PBO on the only complete score matrix: two reversal variants x six folds.

    Score = per-fold annualised Sharpe of the net returns. With two strategies and
    six folds there are C(6,3) = 20 train/test splits. A PBO of 0 here only shows
    that the calculation runs; the matrix is far too small to be informative.
    """
    labels = ["reversal_24h", "reversal_8h"]
    scores = []
    for name in labels:
        book = S.reversal(panel, 24 if name == "reversal_24h" else 8)
        scores.append(
            [annualized_sharpe(book.total[idx]) for idx in strategy_folds(book.total.size)]
        )
    matrix = np.array(scores)
    means = matrix.mean(axis=1)
    best = int(np.argmax(means))
    return {
        "strategies": labels,
        "score_matrix": matrix.tolist(),
        "pbo": stats.probability_of_backtest_overfitting(matrix),
        "n_splits": math.comb(matrix.shape[1], matrix.shape[1] // 2),
        "in_sample_best": labels[best],
        "in_sample_best_mean_fold_sharpe": float(means[best]),
    }


# ---------------------------------------------------------------------------
# Final period (opened once)
# ---------------------------------------------------------------------------


def _descriptive(series: np.ndarray, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    lo, hi = stats.stationary_bootstrap_ci(series, _mean, reps=reps, seed=0)
    return {
        "n": int(series.size),
        "mean_per_hour": float(np.mean(series)),
        "mean_ci_95": [lo, hi],
        "ci_above_zero": bool(lo > 0.0),
        "ci_below_zero": bool(hi < 0.0),
        "sum": float(np.sum(series)),
    }


def final_period(
    full_panel: Panel,
    n_train_valid_rows: int,
    spot_prices: np.ndarray | None = None,
    spot_observed: np.ndarray | None = None,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Rebuild the four frozen books on train+valid+test and keep only the test hours.

    The books run over the contiguous panel so that positions entering 2025 are the
    ones the rules would really hold; the output is then cut to returns earned on
    test bars (1 Jan 2025 to 11 Jun 2026, 12,648 hours). The hedge beta is still fit
    on training rows only. The equal-weight average of the four books is passed once
    through the evaluator; each book also gets a descriptive bootstrap interval.
    """
    cut = slice(n_train_valid_rows - 1, None)  # output t-1 is the return into bar t
    books: dict[str, np.ndarray] = {
        "raw_carry": S.raw_carry(full_panel).total[cut],
        "hedged_carry": S.hedged_carry(full_panel).total[cut],
        "reversal_24h": S.reversal(full_panel, 24).total[cut],
    }
    if spot_prices is not None and spot_observed is not None:
        cc = S.cash_and_carry(full_panel, spot_prices, spot_observed)
        books["cash_and_carry"] = cc.total[cut]
    out: dict[str, Any] = {
        "window": {
            "start": str(full_panel.timestamps[n_train_valid_rows]),
            "end": str(full_panel.timestamps[-1]),
            "n_returns": int(full_panel.n_rows - n_train_valid_rows),
        },
        "power_gate": stats.power_gate(full_panel.n_rows - n_train_valid_rows, PERIODS_PER_YEAR),
        "books": {name: _descriptive(series, reps) for name, series in books.items()},
    }
    if len(books) == len(FINAL_PERIOD_BOOKS):
        portfolio = 0.25 * (
            books["raw_carry"]
            + books["hedged_carry"]
            + books["reversal_24h"]
            + books["cash_and_carry"]
        )
        out["equal_weight_portfolio"] = _descriptive(portfolio, reps)
        out["gate"] = gate.evaluate(
            portfolio,
            np.zeros_like(portfolio),
            books,
            PERIODS_PER_YEAR,
            n_trials=AUDITED_N_TRIALS,
            decision_mode=True,
        )
    else:
        out["equal_weight_portfolio"] = {"skipped": "cash-and-carry needs the spot panel"}
    out["no_book_ci_above_zero"] = not any(b["ci_above_zero"] for b in out["books"].values())
    return out


def funding_timestamp_diagnostic(panel: Panel) -> dict[str, Any]:
    """How much the one-hour-late funding rates (see :mod:`panel`) move raw-carry funding.

    At each funding hour, the rate the exchange actually paid is the one visible one
    bar later whenever the event was stamped a few milliseconds after the hour. This
    keeps the raw-carry positions unchanged and recomputes only the funding received
    with those event-time rates. It is a diagnostic, not part of the frozen study.
    """
    fb = panel.is_funding_bar
    idx = np.flatnonzero(fb[:-1])
    event_rate = panel.funding.copy()
    event_rate[idx] = np.where(panel.tradable[idx + 1], panel.funding[idx + 1], panel.funding[idx])
    stale = (event_rate[idx] != panel.funding[idx]) & panel.tradable[idx]
    reb = np.isin(panel.hours, [0, 8, 16])
    as_coded = S.raw_carry(panel)
    weights_only = S.run_book(
        np.zeros_like(panel.prices),
        lambda t, _w: S.carry_weights(panel.funding[t], panel.tradable[t]),
        reb,
        0.0,
        event_rate,
        fb,
    )
    folds = strategy_folds(as_coded.funding.size)
    return {
        "funding_cells": int(panel.tradable[idx].sum()),
        "cells_with_previous_event_rate": int(stale.sum()),
        "raw_carry_funding_as_published": float(as_coded.funding.sum()),
        "raw_carry_funding_event_time": float(weights_only.funding.sum()),
        "folds_funding_positive_event_time": int(
            sum(float(weights_only.funding[f].sum()) > 0 for f in folds)
        ),
    }


# ---------------------------------------------------------------------------
# Weekly momentum example (Section 5) and market-derived null (Section 3)
# ---------------------------------------------------------------------------

WEEK_HOURS = 168
WEEKLY_TAIL = 4  # tercile of twelve
WEEKS_PER_YEAR = 365.25 / 7.0


def _weekly_panel(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Weekly simple returns ``[W-1, N]`` (sampled every 168 h from the start) and tradability."""
    idx = np.arange(0, panel.n_rows, WEEK_HOURS)
    wp, wt = panel.prices[idx], panel.tradable[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = wp[1:] / wp[:-1] - 1.0
    wr[~np.isfinite(wr)] = 0.0
    return wr, wt


def _weekly_book(wr: np.ndarray, ok_mask: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Long the top-4 and short the bottom-4 by ``signal[w-1]``, held over week ``w``."""
    rets, prev = [], np.zeros(wr.shape[1])
    for w in range(1, wr.shape[0]):
        sig = np.where(ok_mask[w], signal[w - 1], np.nan)
        valid = np.array([i for i in np.argsort(sig) if np.isfinite(sig[i])])
        wgt = np.zeros_like(prev)
        if valid.size >= 2 * WEEKLY_TAIL:
            wgt[valid[-WEEKLY_TAIL:]] = 1.0 / (2 * WEEKLY_TAIL)
            wgt[valid[:WEEKLY_TAIL]] = -1.0 / (2 * WEEKLY_TAIL)
        rets.append(float(np.dot(wgt, wr[w])) - S.COST_BPS * 1e-4 * np.abs(wgt - prev).sum())
        prev = wgt
    return np.asarray(rets)


def _equal_weight_weekly(wr: np.ndarray, wt: np.ndarray) -> np.ndarray:
    return np.array(
        [float(wr[w][wt[w]].mean()) if wt[w].any() else 0.0 for w in range(1, wr.shape[0])]
    )


def weekly_momentum_benchmark(panel: Panel, n_trials: int = 411) -> dict[str, Any]:
    """A conventional weekly momentum book: long last week's top tercile, short the bottom.

    276 weekly returns on train+valid are fewer than the 323 required by the power
    gate, so the evaluator abstains even though the naive bootstrap interval of the
    mean is (barely) above zero.
    """
    wr, wt = _weekly_panel(panel)
    n_w = wt.shape[0]
    ok = np.zeros_like(wt)
    for w in range(1, wr.shape[0]):
        ok[w] = wt[w] & wt[w - 1] & wt[w + 1] if w + 1 < n_w else wt[w] & wt[w - 1]
    rets = _weekly_book(wr, ok, wr)
    lo, hi = stats.stationary_bootstrap_ci(rets, _mean, reps=BOOTSTRAP_REPS, seed=0)
    verdict = gate.evaluate(
        rets,
        np.zeros_like(rets),
        {"market_ew": _equal_weight_weekly(wr, wt)},
        WEEKS_PER_YEAR,
        n_trials=n_trials,
    )
    return {
        "n_weekly_returns": int(rets.size),
        "n_required": stats.minimum_required_observations(WEEKS_PER_YEAR),
        "mean_weekly_net": float(rets.mean()),
        "mean_ci_95": [lo, hi],
        "gate_status": verdict["status"],
        "gate_all_pass": verdict["all_pass"],
    }


def market_null_check(panel: Panel, n_replicates: int = 200, min_shift: int = 26) -> dict[str, Any]:
    """Run the evaluator on weekly momentum books whose signal is circularly shifted.

    Shifting the signal by at least 26 weeks keeps every real feature of the returns
    (fat tails, volatility clustering, cross-correlation) but destroys any genuine
    link between signal and future return. Run at the diagnostic S_min = 1.5 so that
    all checks downstream of the power gate are exercised; none should pass.
    """
    wr, wt = _weekly_panel(panel)
    ok = np.zeros_like(wt)
    for w in range(1, wr.shape[0]):
        ok[w] = wt[w] & wt[w - 1]
    ew = _equal_weight_weekly(wr, wt)
    rng = np.random.default_rng(0)
    n_pass = n_ok = 0
    means = []
    for i in range(n_replicates):
        shift = int(rng.integers(min_shift, wr.shape[0] - min_shift))
        rets = _weekly_book(wr, ok, np.roll(wr, shift, axis=0))
        verdict = gate.evaluate(
            rets,
            np.zeros_like(rets),
            {"market_ew": ew},
            WEEKS_PER_YEAR,
            n_trials=n_replicates,
            thresholds={"s_min_annual": 1.5, "bootstrap_reps": 150.0},
            seed=1_000 + i,
        )
        n_pass += int(bool(verdict["all_pass"]))
        n_ok += int(verdict["status"] == "ok")
        means.append(float(rets.mean()))
    return {  # same keys as the paper's market_null_check.json
        "construction": "H111 weekly tercile book on REAL panel returns, signal "
        "circularly shifted by >=26 weeks; 5 bps/leg; S_min=1.5 "
        "diagnostic so the full downstream stack runs",
        "n_replicates": n_replicates,
        "n_full_gate_pass": n_pass,
        "n_status_ok": n_ok,
        "mean_of_mean_weekly_net": float(np.mean(means)),
        "n_weekly_obs": int(wr.shape[0] - 1),
    }


# ---------------------------------------------------------------------------
# Output helpers used by the CLI scripts
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as indented JSON (numpy types converted, NaN -> null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n", encoding="utf-8")
    return path


def save_csv(rows: list[dict[str, Any]], path: str | Path, header: tuple[str, ...] = ()) -> Path:
    """Write a list of flat dicts as CSV (columns from the first row).

    ``header`` lines (each starting with ``#``) are written first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        for line in header:
            fh.write(line + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    return path
