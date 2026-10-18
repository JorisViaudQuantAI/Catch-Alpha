#!/usr/bin/env python
"""Weekly momentum example (Section 5) and market-derived null check (Section 3).

Writes ``weekly_momentum.json`` and ``market_null_check.json`` (200 evaluator runs,
about half a minute).

    python scripts/real_data_weekly_momentum.py --panel-dir data/public_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catch_alpha.realdata import study
from catch_alpha.realdata.panel import load_panel

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "real_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-dir", required=True, help="directory with public_v3_*.npydir")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out_dir)
    panel = load_panel(args.panel_dir)
    study.save_json(study.weekly_momentum_benchmark(panel), out / "weekly_momentum.json")
    study.save_json(study.market_null_check(panel), out / "market_null_check.json")
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
