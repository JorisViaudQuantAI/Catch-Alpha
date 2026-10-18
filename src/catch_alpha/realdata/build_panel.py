"""Turn the downloaded lake into the hourly panel splits read by :mod:`panel`.

UNVERIFIED END TO END IN THIS REPOSITORY'S CI (no Binance access); the feature
code was checked against the published panel on the columns that depend on close
prices only (see ``tests/test_realdata.py`` and the release notes). This is a
simplified port of ``build_public_v3_dataset.py --stage materialize``; dataset
registry bookkeeping and the train-only z-scored copy of the features (unused by
the paper) are omitted.

Steps:

1. Put every symbol on one hourly UTC calendar from the first listing to the end.
   Before a contract lists its price is 0 and it is not tradable; after listing,
   missing bars are forward-filled and flagged as not observed (not tradable).
2. Compute 23 causal bar features per symbol (:func:`bar_features`) and two
   funding columns (:func:`funding_features`). Features are set to 0 on bars that
   are not tradable.
3. Cut the calendar into train (to 2023-12-31 23:00), valid (2024) and test
   (2025-01-01 onward) and save each as a directory of ``.npy`` arrays.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catch_alpha.realdata.download import PANEL_END, UNIVERSE, read_funding, read_klines
from catch_alpha.realdata.panel import Panel

logger = logging.getLogger(__name__)

TRAIN_END = pd.Timestamp("2023-12-31T23:00:00Z")
VALID_START = pd.Timestamp("2024-01-01T00:00:00Z")
VALID_END = pd.Timestamp("2024-12-31T23:00:00Z")
TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")
FUNDING_INTERVAL_HOURS = 8


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def bar_features(bars: pd.DataFrame, observed: np.ndarray) -> pd.DataFrame:
    """The 23 per-bar features of one symbol (port of the original ``BarFeaturePipeline``).

    ``bars`` has columns open/high/low/close/volume on the hourly calendar
    (forward-filled after listing, 0 before); ``observed`` flags real bars. Every
    feature at ``t`` uses data up to ``t`` only. NaNs (warm-up windows) become 0.
    Note: ``breakout_lookback_high_distance`` and ``drawdown_from_lookback_high`` are
    the same statistic in the original code; both are kept so the 32-signal screen
    is reproduced exactly.
    """
    obs = pd.Series(np.asarray(observed, dtype=bool), index=bars.index)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    obs_close = close.where(obs)
    close_ff = obs_close.ffill()
    obs_volume = volume.where(obs)

    log_close = np.log(close_ff + 1e-12)
    ret1 = log_close.diff(1).where(obs, 0.0)
    ret5 = log_close.diff(5).where(obs, 0.0)
    rolling_mean = obs_close.rolling(20).mean()
    ema_fast = close_ff.ewm(span=12, adjust=False).mean().where(obs)
    ema_slow = close_ff.ewm(span=26, adjust=False).mean().where(obs)
    delta = close_ff.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    rsi = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))).where(obs)
    boll_std = obs_close.rolling(20).std()
    lookback_high = bars["high"].astype(float).where(obs).rolling(32).max()
    distance_from_high = close / (lookback_high + 1e-12) - 1.0
    compound = {
        f"compound_return_{w}": (close_ff / (close_ff.shift(w) + 1e-12) - 1.0).where(obs, 0.0)
        for w in (8, 16, 32, 64)
    }
    feats = pd.DataFrame(
        {
            "ret_1": ret1,
            "ret_5": ret5,
            "realized_vol": ret1.where(obs).rolling(20).std(),
            "rolling_mean": rolling_mean,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
            "macd": ema_fast - ema_slow,
            "ema_fast_slow_spread": ema_fast / (ema_slow + 1e-12) - 1.0,
            "ema_slope": ema_fast.pct_change(12),
            "boll_width": 2.0 * boll_std / (rolling_mean + 1e-12),
            "volume_z": (obs_volume - obs_volume.rolling(20).mean())
            / (obs_volume.rolling(20).std() + 1e-12),
            "illiq": (ret1.abs() / (obs_volume + 1e-8)).replace([np.inf, -np.inf], 0.0),
            "spread_proxy": ((bars["high"] - bars["low"]) / (close.abs() + 1e-12)).where(obs),
            "breakout_lookback_high_distance": distance_from_high,
            "drawdown_from_lookback_high": distance_from_high,
            "positive_momentum_flag_32": (compound["compound_return_32"] > 0.0).astype(float),
            "bar_observed_mask": obs.astype(float),
            "bar_stale_mask": 1.0 - obs.astype(float),
            **compound,
        },
        index=bars.index,
    )
    return feats.fillna(0.0).astype("float32")


def funding_features(calendar: pd.DatetimeIndex, funding: pd.DataFrame) -> pd.DataFrame:
    """``funding_rate_last`` and ``hours_to_next_funding`` on the hourly calendar.

    ``funding_rate_last[t]`` is the rate of the latest funding event with event time
    ``<= t``: an 08:00 payment is never visible before the 08:00 bar. Because event
    times can carry a few milliseconds past the hour, such an event first shows up at
    the 09:00 bar; this matches the published panel and is kept on purpose (see the
    caveat in :mod:`catch_alpha.realdata.panel`).
    """
    rate = np.zeros(len(calendar))
    if len(funding):
        ev = funding.sort_values("timestamp")
        idx = np.searchsorted(ev["timestamp"].to_numpy(), calendar.to_numpy(), side="right") - 1
        rate[idx >= 0] = ev["funding_rate"].to_numpy(dtype=np.float64)[idx[idx >= 0]]
    hours = calendar.hour.to_numpy()
    to_next = (FUNDING_INTERVAL_HOURS - hours % FUNDING_INTERVAL_HOURS).astype(np.float64)
    return pd.DataFrame(
        {"funding_rate_last": rate, "hours_to_next_funding": to_next}, index=calendar
    )


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def assemble_panel(
    lake_dir: str | Path, symbols: Sequence[str] = UNIVERSE, end: pd.Timestamp = PANEL_END
) -> dict[str, Any]:
    """Build the full aligned panel (all splits) from ``<lake>/binance_perps``."""
    root = Path(lake_dir) / "binance_perps"
    bars_by_symbol = {}
    for s in symbols:
        frame = read_klines(root / "1h", s)
        frame = frame[frame["timestamp"] <= end]
        if frame.empty:
            raise ValueError(f"no bars for {s}; run download_futures first")
        bars_by_symbol[s] = frame.set_index("timestamp")
    first_bar = {s: f.index.min() for s, f in bars_by_symbol.items()}
    calendar = pd.date_range(
        min(first_bar.values()),
        max(f.index.max() for f in bars_by_symbol.values()),
        freq="1h",
        tz="UTC",
    )
    prices, tradable, features, names = [], [], [], None
    for s in symbols:
        raw = bars_by_symbol[s].reindex(calendar)
        listed = np.asarray(calendar >= first_bar[s])
        observed = listed & raw["close"].notna().to_numpy()
        ohlc = raw[["open", "high", "low", "close"]].copy()
        ohlc.loc[~listed] = np.nan
        ohlc = ohlc.ffill().fillna(0.0)  # forward-fill after listing, 0 before
        vol = raw["volume"].where(listed).fillna(0.0)
        bars = ohlc.assign(volume=vol)
        merged = pd.concat(
            [
                bar_features(bars, observed),
                funding_features(calendar, read_funding(root / "funding", s)),
            ],
            axis=1,
        )
        merged.loc[~observed] = 0.0  # no features on bars that cannot be traded
        names = list(merged.columns)
        prices.append(bars["close"].to_numpy(dtype=np.float64))
        tradable.append(observed)
        features.append(merged.to_numpy(dtype=np.float64))
    return {
        "timestamps": calendar,
        "symbols": list(symbols),
        "prices": np.stack(prices, axis=1),
        "tradable": np.stack(tradable, axis=1),
        "features": np.nan_to_num(np.stack(features, axis=1), nan=0.0, posinf=0.0, neginf=0.0),
        "feature_names": names,
        "first_bar": {s: t.isoformat() for s, t in first_bar.items()},
    }


def split_bounds(calendar: pd.DatetimeIndex) -> dict[str, tuple[int, int]]:
    """Half-open row ranges of the train / valid / test splits."""
    masks = {
        "train": calendar <= TRAIN_END,
        "valid": (calendar >= VALID_START) & (calendar <= VALID_END),
        "test": calendar >= TEST_START,
    }
    out = {}
    for name, m in masks.items():
        rows = np.flatnonzero(m)
        out[name] = (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, 0)
    return out


def write_splits(panel: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Save ``public_v3_{train,valid,test}.npydir`` plus a small manifest."""
    out_dir = Path(out_dir)
    stamps = np.asarray([ts.isoformat() for ts in panel["timestamps"]])
    manifest: dict[str, Any] = {
        "symbols": panel["symbols"],
        "feature_names": panel["feature_names"],
        "first_bar_by_symbol": panel["first_bar"],
        "splits": {},
    }
    for name, (lo, hi) in split_bounds(panel["timestamps"]).items():
        mask = panel["tradable"][lo:hi].astype(np.float32)
        arrays = {
            "prices": panel["prices"][lo:hi].astype(np.float32),
            "features": panel["features"][lo:hi].astype(np.float32),
            "tradable_mask_ts": mask,
            "tradable_execution_mask": mask,
            "observed_bar_mask": mask,
            "timestamps": stamps[lo:hi],
            "symbols": np.asarray(panel["symbols"]),
            "feature_names": np.asarray(panel["feature_names"]),
        }
        split_dir = out_dir / f"public_v3_{name}.npydir"
        split_dir.mkdir(parents=True, exist_ok=True)
        spec = {}
        for key, arr in arrays.items():
            np.save(split_dir / f"{key}.npy", arr, allow_pickle=False)
            spec[key] = {"path": f"{key}.npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}
        (split_dir / "arrays.json").write_text(json.dumps({"arrays": spec}, indent=2))
        manifest["splits"][name] = {
            "rows": hi - lo,
            "start": str(stamps[lo]) if hi > lo else None,
            "end": str(stamps[hi - 1]) if hi > lo else None,
        }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_panel(lake_dir: str | Path, out_dir: str | Path, end: pd.Timestamp = PANEL_END) -> dict:
    """Lake -> panel splits, in one call."""
    return write_splits(assemble_panel(lake_dir, end=end), out_dir)


# ---------------------------------------------------------------------------
# Spot panel for cash-and-carry
# ---------------------------------------------------------------------------


def load_spot_panel(spot_lake_dir: str | Path, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Spot closes ``[T, N]`` aligned to the perpetual timestamps, and an observed mask.

    ``spot_lake_dir`` is the ``binance_spot`` directory (containing ``1h/``). Missing
    spot bars after the first one are forward-filled (and flagged unobserved);
    before the first spot bar the price is 0.
    """
    root = Path(spot_lake_dir) / "1h"
    if not root.is_dir():
        raise FileNotFoundError(
            f"no spot lake at {root}; pass the binance_spot directory that contains 1h/ "
            "(scripts/real_data_build_panel.py --spot writes it to <lake-dir>/binance_spot)"
        )
    index = pd.DatetimeIndex(pd.to_datetime([str(t) for t in panel.timestamps], utc=True))
    n_t, n_a = len(index), len(panel.symbols)
    close, observed = np.zeros((n_t, n_a)), np.zeros((n_t, n_a), dtype=bool)
    for j, symbol in enumerate(panel.symbols):
        frame = read_klines(root, symbol)
        if frame.empty:
            logger.warning("no spot bars for %s in %s", symbol, root)
            continue
        series = pd.Series(frame["close"].to_numpy(dtype=np.float64), index=frame["timestamp"])
        series = series[~series.index.duplicated(keep="last")].sort_index().reindex(index)
        seen = series.notna().to_numpy()
        listed = np.cumsum(seen) > 0
        vals = series.ffill().to_numpy(dtype=np.float64)
        close[:, j] = np.where(listed & np.isfinite(vals), vals, 0.0)
        observed[:, j] = seen
    if not observed.any():
        raise ValueError(f"the spot lake {root} has no bars for any panel symbol")
    return close, observed
