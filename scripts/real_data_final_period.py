#!/usr/bin/env python
"""The single final-period evaluation on the locked 2025-01-01 to 2026-06-11 window.

This is the only script that reads the test split. It rebuilds the four frozen
books over train+valid+test, keeps the 12,648 test-period hours, and writes
``final_period.json`` (per-book descriptive intervals, the equal-weight portfolio
and its evaluator verdict). Cash-and-carry, and hence the portfolio, needs the spot
lake to cover the test period.

    python scripts/real_data_final_period.py --panel-dir data/public_v3 --spot-lake data/lake/binance_spot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catch_alpha.realdata import study
from catch_alpha.realdata.build_panel import load_spot_panel
from catch_alpha.realdata.panel import load_panel

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "real_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-dir", required=True, help="directory with public_v3_*.npydir")
    parser.add_argument(
        "--spot-lake",
        required=True,
        help="the binance_spot directory with 1h/ inside (cash-and-carry needs spot prices)",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    n_train_valid = load_panel(args.panel_dir).n_rows
    full = load_panel(args.panel_dir, ("train", "valid", "test"))
    spot = load_spot_panel(args.spot_lake, full)
    result = study.final_period(full, n_train_valid, *spot)
    path = study.save_json(result, Path(args.out_dir) / "final_period.json")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
