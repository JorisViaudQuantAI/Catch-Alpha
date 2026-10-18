#!/usr/bin/env python
"""Download public Binance data and rebuild the hourly panel (needs internet access).

UNVERIFIED in the release environment (no access to Binance); see
``catch_alpha.realdata.download``. Roughly 2-3 GB of parquet and an hour of
throttled requests.

    python scripts/real_data_build_panel.py --lake-dir data/lake --out-dir data/public_v3 --spot
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from catch_alpha.realdata.build_panel import build_panel
from catch_alpha.realdata.download import PANEL_END, download_futures, download_spot


def parse_end(value: str) -> pd.Timestamp:
    """UTC last bar-open time; a bare date means through the end of that day (23:00)."""
    end = pd.Timestamp(value)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if len(value.strip()) == 10:  # YYYY-MM-DD
        end += pd.Timedelta(hours=23)
    return end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lake-dir", required=True, help="where raw parquet files are stored")
    parser.add_argument("--out-dir", required=True, help="where public_v3_*.npydir are written")
    parser.add_argument(
        "--end",
        default=PANEL_END.isoformat(),
        help="last bar-open time, UTC (a bare date such as 2026-06-11 means through 23:00)",
    )
    parser.add_argument("--spot", action="store_true", help="also download spot (cash-and-carry)")
    parser.add_argument("--skip-download", action="store_true", help="build from an existing lake")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    end = parse_end(args.end)
    if not args.skip_download:
        print(json.dumps(download_futures(args.lake_dir, end=end), indent=2))
        if args.spot:
            print(json.dumps(download_spot(args.lake_dir, end=end), indent=2))
    print(json.dumps(build_panel(args.lake_dir, args.out_dir, end=end)["splits"], indent=2))


if __name__ == "__main__":
    main()
