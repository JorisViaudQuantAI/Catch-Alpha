"""Load the hourly Binance perpetual-futures panel used in the real-data study.

The panel (called ``public_v3`` in the original research code) holds twelve
USD-margined perpetual futures at hourly frequency, stored as three chronological
splits, each a directory of ``.npy`` arrays:

* ``public_v3_train.npydir``  -- 2019-09-08 17:00 to 2023-12-31 23:00 UTC (37,807 rows)
* ``public_v3_valid.npydir``  -- 2024-01-01 00:00 to 2024-12-31 23:00 UTC (8,784 rows)
* ``public_v3_test.npydir``   -- 2025-01-01 00:00 to 2026-06-11 23:00 UTC (12,648 rows)

Every study in the paper uses train and valid concatenated (46,591 rows, i.e.
46,590 hour-to-hour returns). The test split is the locked "final period" and is
opened exactly once (:func:`catch_alpha.realdata.study.final_period`).

Conventions a reader needs:

* Timestamps are bar-open times in UTC. ``prices[t]`` is the close of the bar that
  opened at ``timestamps[t]``; the simple return earned between rows ``t-1`` and
  ``t`` is ``prices[t] / prices[t-1] - 1``.
* Before a contract lists, its price is 0 and it is not tradable. Missing bars after
  listing are forward-filled and marked not tradable.
* ``funding[t]`` is the most recent funding rate whose event time is at or before
  ``t`` (Binance pays funding every eight hours at 00:00, 08:00 and 16:00 UTC). A
  positive rate means holders of a long position pay holders of a short position.
  The strategies treat ``funding[t]`` at a funding hour as the rate paid at that
  hour. Caveat found while preparing this release: Binance stamps many events a
  few milliseconds after the hour (e.g. ``16:00:00.015``), and the panel builder
  compares event times with exact bar-open times, so for those events the new rate
  only appears at the next bar (01/09/17 UTC). At about 29% of (funding hour,
  asset) cells the rate used is therefore the previous event's rate. This is kept
  as published; :func:`catch_alpha.realdata.study.funding_timestamp_diagnostic`
  measures the effect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SPLITS: tuple[str, ...] = ("train", "valid", "test")
FUNDING_HOURS: tuple[int, ...] = (0, 8, 16)  # UTC hours at which Binance pays funding
PERIODS_PER_YEAR = 8760.0  # hourly bars, 24 x 365
FOLD_GAP_HOURS = 168  # one week between folds so overlapping information cannot leak
N_FOLDS = 6


@dataclass(frozen=True)
class Panel:
    """A contiguous block of the hourly panel (one split, or several concatenated).

    Attributes
    ----------
    timestamps:
        ``[T]`` ISO-8601 bar-open strings, e.g. ``"2019-09-08T17:00:00+00:00"``.
    symbols:
        The twelve contract names (``"BTCUSDT"``, ...).
    prices:
        ``[T, N]`` close prices (0 before listing).
    funding:
        ``[T, N]`` latest funding rate known at ``t`` (raw, per eight-hour event).
    features:
        ``[T, N, F]`` the 25 raw per-asset features (unnormalised).
    feature_names:
        Names of the ``F`` feature columns.
    tradable:
        ``[T, N]`` True where a real bar exists and the contract can be traded.
    n_train:
        Number of leading rows that come from the train split (used to fit the
        beta hedge on training data only). Zero if the panel has no train rows.
    splits:
        Which splits were concatenated, in order.
    """

    timestamps: np.ndarray
    symbols: tuple[str, ...]
    prices: np.ndarray
    funding: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    tradable: np.ndarray
    n_train: int
    splits: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.prices.shape[0])

    @property
    def n_returns(self) -> int:
        """Number of hour-to-hour returns (rows minus one)."""
        return max(0, self.n_rows - 1)

    @property
    def hours(self) -> np.ndarray:
        """``[T]`` UTC hour of each bar."""
        return np.array([int(str(ts)[11:13]) for ts in self.timestamps], dtype=np.int64)

    @property
    def is_funding_bar(self) -> np.ndarray:
        """``[T]`` True at 00:00, 08:00 and 16:00 UTC, when funding changes hands."""
        return np.isin(self.hours, np.array(FUNDING_HOURS, dtype=np.int64))

    def feature(self, name: str) -> np.ndarray:
        """Return one raw feature column ``[T, N]``."""
        return self.features[:, :, self.feature_names.index(name)]


def _load_split_arrays(split_dir: Path) -> dict[str, np.ndarray]:
    names = ("prices", "features", "tradable_mask_ts", "timestamps", "feature_names", "symbols")
    return {name: np.load(split_dir / f"{name}.npy", allow_pickle=False) for name in names}


def load_panel(panel_dir: str | Path, splits: tuple[str, ...] = ("train", "valid")) -> Panel:
    """Load and concatenate the requested splits chronologically.

    ``load_panel(d)`` gives the train+valid window used for every selection-stage
    result; ``load_panel(d, ("train", "valid", "test"))`` gives the contiguous panel
    used once for the final-period evaluation.
    """
    panel_dir = Path(panel_dir)
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; expected a subset of {SPLITS}")
    order = [s for s in SPLITS if s in splits]  # always chronological
    parts = [_load_split_arrays(panel_dir / f"public_v3_{s}.npydir") for s in order]

    feature_names = tuple(str(x) for x in parts[0]["feature_names"])
    symbols = tuple(str(x) for x in parts[0]["symbols"])
    for part in parts[1:]:
        if tuple(str(x) for x in part["feature_names"]) != feature_names:
            raise ValueError("feature_names differ between splits")
        if tuple(str(x) for x in part["symbols"]) != symbols:
            raise ValueError("symbols differ between splits")

    features = np.concatenate([p["features"] for p in parts]).astype(np.float64)
    n_train = int(parts[0]["prices"].shape[0]) if order[0] == "train" else 0
    return Panel(
        timestamps=np.concatenate([p["timestamps"] for p in parts]),
        symbols=symbols,
        prices=np.concatenate([p["prices"] for p in parts]).astype(np.float64),
        funding=features[:, :, feature_names.index("funding_rate_last")].copy(),
        features=features,
        feature_names=feature_names,
        tradable=np.concatenate([p["tradable_mask_ts"] for p in parts]).astype(np.float64) > 0.5,
        n_train=n_train,
        splits=tuple(order),
    )


def slice_rows(panel: Panel, start: int, stop: int) -> Panel:
    """Rows ``[start, stop)`` of a panel (used by tests and for sub-windows)."""
    return replace(
        panel,
        timestamps=panel.timestamps[start:stop],
        prices=panel.prices[start:stop],
        funding=panel.funding[start:stop],
        features=panel.features[start:stop],
        tradable=panel.tradable[start:stop],
        n_train=max(0, min(panel.n_train, stop) - start),
    )


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """``[T, N]`` simple returns with ``r[t] = P[t] / P[t-1] - 1`` and ``r[0] = 0``.

    A return is 0 whenever either price is missing or non-positive (e.g. before
    listing), so an unlisted asset never contributes PnL.
    """
    prices = np.asarray(prices, dtype=np.float64)
    out = np.zeros_like(prices)
    prev, cur = prices[:-1], prices[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0.0) & (cur > 0.0)
    ret = np.zeros_like(cur)
    ret[ok] = cur[ok] / prev[ok] - 1.0
    out[1:] = ret
    return out


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def strategy_folds(
    n_returns: int, n_folds: int = N_FOLDS, gap: int = FOLD_GAP_HOURS
) -> list[np.ndarray]:
    """Six chronological evaluation windows over a return series, each after a gap.

    Used for every portfolio result. The layout is ``gap | fold 0 | gap | fold 1 |
    ... | gap | fold 5`` with equal fold lengths ``(n - n_folds * gap) // n_folds``;
    the frozen strategies have nothing to fit, so only the evaluation windows are
    used. On the 46,590 train+valid returns each fold has 7,597 hours and the
    first fold starts at index 168.
    """
    n_folds = max(1, int(n_folds))
    length = max(1, (int(n_returns) - gap * n_folds) // n_folds)
    folds, start = [], 0
    for _ in range(n_folds):
        lo, hi = start + gap, start + gap + length
        if hi > n_returns:
            break
        folds.append(np.arange(lo, hi, dtype=np.int64))
        start = hi
    return folds


def signal_folds(
    n_rows: int, n_folds: int = N_FOLDS, gap: int = FOLD_GAP_HOURS
) -> list[tuple[int, int]]:
    """Six equal chronological row blocks; the first ``gap`` rows of each are dropped.

    Used by the signal screen (it works on rows, not returns). Returns
    ``[(start, end), ...]`` half-open row ranges after the gap. This is the same idea
    as :func:`strategy_folds` with a slightly different rounding, kept exactly as in
    the original research code so the published numbers reproduce.
    """
    edges = np.linspace(0, int(n_rows), int(n_folds) + 1, dtype=np.int64)
    return [
        (min(int(edges[i]) + gap, int(edges[i + 1])), int(edges[i + 1])) for i in range(n_folds)
    ]
