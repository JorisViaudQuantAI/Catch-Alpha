"""Download the public Binance data behind the panel into a local parquet "lake".

UNVERIFIED IN THIS REPOSITORY'S CI: this module needs network access to Binance's
public REST API, which the environment used to prepare the release could not
reach. It is a faithful, simplified port of the original collectors
(``build_public_v3_dataset.py --stage fetch`` and ``build_public_v3_spot.py``):
same endpoints, same pagination, same storage layout. No API key is required.

What is downloaded, per symbol:

* USD-M perpetual 1h klines from ``https://fapi.binance.com/fapi/v1/klines``
  (from listing onward), stored as ``<lake>/binance_perps/1h/symbol=S/date=D/part-000.parquet``;
* the full funding-rate history from ``/fapi/v1/fundingRate``, stored as
  ``<lake>/binance_perps/funding/symbol=S/funding.parquet``;
* spot 1h klines from ``https://api.binance.com/api/v3/klines`` (only needed for the
  cash-and-carry strategy), stored as ``<lake>/binance_spot/1h/symbol=S/date=D/...``.

Binance timestamps are bar-open times in milliseconds (UTC). Downloads are
resumable: days already on disk are skipped.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

UNIVERSE: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "ETCUSDT",
)
PANEL_START = pd.Timestamp("2019-09-08T17:00:00Z")  # first bar of the published panel
PANEL_END = pd.Timestamp("2026-06-11T23:00:00Z")  # last bar of the published panel

FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"
FUTURES_KLINES = "/fapi/v1/klines"
FUNDING_RATES = "/fapi/v1/fundingRate"
SPOT_KLINES = "/api/v3/klines"
PAGE_LIMIT = 1000  # Binance maximum rows per request for both endpoints
HOUR_MS = 3_600_000
FUTURES_DISCOVERY_START = pd.Timestamp("2019-01-01T00:00:00Z")  # before any USD-M listing
SPOT_DISCOVERY_START = pd.Timestamp("2017-01-01T00:00:00Z")

RequestFn = Callable[[str], bytes]


def http_get(url: str, timeout: float = 30.0) -> bytes:
    """Plain HTTPS GET (the only network call in this module)."""
    req = urllib.request.Request(url, headers={"User-Agent": "catch-alpha/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return bytes(response.read())


def get_json(
    url: str, request: RequestFn = http_get, retries: int = 5, backoff: float = 0.5
) -> Any:
    """GET + JSON decode with exponential backoff; HTTP 400 (bad symbol/range) is not retried."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(request(url).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - transient transport errors
            last = exc
        LOGGER.warning("retrying %s (attempt %d): %s", url, attempt + 1, last)
        time.sleep(backoff * 2**attempt)
    assert last is not None
    raise last


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Paginated fetchers
# ---------------------------------------------------------------------------


def fetch_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
    base: str = FUTURES_BASE,
    path: str = FUTURES_KLINES,
    request: RequestFn = http_get,
    sleep: float = 0.15,
) -> pd.DataFrame:
    """Hourly OHLCV bars with open time in ``[start_ms, end_ms)``.

    Pages forward by setting ``startTime`` to the last open time plus one hour;
    duplicates at page edges are dropped and the (possibly incomplete) bar at
    ``end_ms`` is excluded.
    """
    rows: dict[int, dict[str, Any]] = {}
    cursor = int(start_ms)
    while cursor < end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": PAGE_LIMIT,
            }
        )
        page = get_json(f"{base}{path}?{query}", request)
        if not isinstance(page, list) or not page:
            break
        for k in page:  # [open_ms, open, high, low, close, volume, ...]
            if len(k) >= 6 and int(k[0]) < end_ms:
                rows[int(k[0])] = {
                    "timestamp": pd.Timestamp(int(k[0]), unit="ms", tz="UTC"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
        next_cursor = int(page[-1][0]) + HOUR_MS
        if next_cursor <= cursor or len(page) < PAGE_LIMIT:
            break
        cursor = next_cursor
        time.sleep(sleep)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return pd.DataFrame([rows[k] for k in sorted(rows)], columns=cols)


def fetch_funding(
    symbol: str, start_ms: int, end_ms: int, request: RequestFn = http_get, sleep: float = 0.15
) -> pd.DataFrame:
    """Every funding event (time, rate) in ``[start_ms, end_ms]``.

    ``fundingRate`` is the fraction of notional paid at ``fundingTime``; positive
    means longs pay shorts.
    """
    rows: dict[int, float] = {}
    cursor = int(start_ms)
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": PAGE_LIMIT}
        )
        page = get_json(f"{FUTURES_BASE}{FUNDING_RATES}?{query}", request)
        if not isinstance(page, list) or not page:
            break
        for ev in page:
            if ev.get("fundingTime") is not None and ev.get("fundingRate") is not None:
                rows[int(ev["fundingTime"])] = float(ev["fundingRate"])
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor or len(page) < PAGE_LIMIT:
            break
        cursor = next_cursor
        time.sleep(sleep)
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(k, unit="ms", tz="UTC") for k in sorted(rows)],
            "funding_rate": [rows[k] for k in sorted(rows)],
        }
    )


# ---------------------------------------------------------------------------
# Lake layout
# ---------------------------------------------------------------------------


def write_daily_partitions(frame: pd.DataFrame, klines_root: Path, symbol: str) -> int:
    """Write one parquet file per UTC day; existing days are left untouched."""
    if frame.empty:
        return 0
    written = 0
    days = frame["timestamp"].dt.strftime("%Y-%m-%d")
    for day, group in frame.groupby(days, sort=True):
        target = klines_root / f"symbol={symbol}" / f"date={day}" / "part-000.parquet"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        out = group[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        out.insert(1, "symbol", symbol)
        out.to_parquet(target, index=False)
        written += 1
    return written


def read_klines(klines_root: Path, symbol: str) -> pd.DataFrame:
    """All stored hourly bars of one symbol, sorted and de-duplicated."""
    parts = sorted((klines_root / f"symbol={symbol}").glob("date=*/part-000.parquet"))
    if not parts:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def read_funding(funding_root: Path, symbol: str) -> pd.DataFrame:
    """Stored funding events of one symbol (empty frame if none)."""
    target = funding_root / f"symbol={symbol}" / "funding.parquet"
    if not target.exists():
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    frame = pd.read_parquet(target)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def _resume_start(klines_root: Path, symbol: str, discovery: pd.Timestamp) -> int:
    existing = read_klines(klines_root, symbol)
    if len(existing):
        return _ms(existing["timestamp"].max()) + HOUR_MS
    return _ms(discovery)


def download_futures(
    lake_dir: str | Path,
    symbols: Sequence[str] = UNIVERSE,
    end: pd.Timestamp = PANEL_END,
    request: RequestFn = http_get,
) -> dict[str, Any]:
    """Download perpetual klines and funding for every symbol into ``<lake>/binance_perps``.

    ``end`` is the last bar-open time to include. The universe is fixed: a symbol
    that returns no data raises instead of being silently dropped.
    """
    root = Path(lake_dir) / "binance_perps"
    klines_root, funding_root = root / "1h", root / "funding"
    end_ms = _ms(end) + HOUR_MS  # include the bar that opens at `end`
    summary: dict[str, Any] = {}
    for symbol in symbols:
        start_ms = _resume_start(klines_root, symbol, FUTURES_DISCOVERY_START)
        bars = (
            fetch_klines(symbol, start_ms, end_ms, request=request) if start_ms < end_ms else None
        )
        if bars is not None:
            write_daily_partitions(bars, klines_root, symbol)
        funding = fetch_funding(symbol, _ms(FUTURES_DISCOVERY_START), _ms(end), request)
        target = funding_root / f"symbol={symbol}" / "funding.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        funding.assign(symbol=symbol)[["timestamp", "symbol", "funding_rate"]].to_parquet(
            target, index=False
        )
        stored = read_klines(klines_root, symbol)
        if stored.empty:
            raise RuntimeError(f"no klines returned for {symbol}; the universe is fixed")
        summary[symbol] = {
            "first_bar": stored["timestamp"].min().isoformat(),
            "last_bar": stored["timestamp"].max().isoformat(),
            "n_bars": int(len(stored)),
            "n_funding_events": int(len(funding)),
        }
        LOGGER.info("futures %s: %s", symbol, summary[symbol])
    return summary


def download_spot(
    lake_dir: str | Path,
    symbols: Sequence[str] = UNIVERSE,
    start: pd.Timestamp = PANEL_START,
    end: pd.Timestamp = PANEL_END,
    request: RequestFn = http_get,
) -> dict[str, Any]:
    """Download spot hourly klines into ``<lake>/binance_spot`` (for cash-and-carry).

    The window starts at the first perpetual bar: spot history before it is not
    needed. Note: the original collector stopped at the end of the validation
    period; the final-period evaluation needs spot data through ``PANEL_END``.
    """
    klines_root = Path(lake_dir) / "binance_spot" / "1h"
    end_ms = _ms(end) + HOUR_MS
    summary: dict[str, Any] = {}
    for symbol in symbols:
        start_ms = max(_resume_start(klines_root, symbol, SPOT_DISCOVERY_START), _ms(start))
        if start_ms < end_ms:
            bars = fetch_klines(symbol, start_ms, end_ms, SPOT_BASE, SPOT_KLINES, request)
            write_daily_partitions(bars, klines_root, symbol)
        stored = read_klines(klines_root, symbol)
        summary[symbol] = {
            "first_bar": stored["timestamp"].min().isoformat() if len(stored) else None,
            "n_bars": int(len(stored)),
        }
    return summary
