#!/usr/bin/env python
"""Price-signal screen: 32 signals x 4 horizons on train+valid (paper Section 4, Figure 3).

Writes ``signal_screen.json`` (every pair with per-fold statistics),
``signal_screen.csv`` (one row per pair), ``signal_screen_summary.json`` and
``signal_atlas_price.csv`` (Figure 3 input, same schema as the paper file).

    python scripts/real_data_signal_screen.py --panel-dir data/public_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catch_alpha.realdata import study
from catch_alpha.realdata.panel import load_panel
from catch_alpha.realdata.signals import (
    FIGURE_HEADER,
    figure_rows,
    run_signal_screen,
    summarize_screen,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "real_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-dir", required=True, help="directory with public_v3_*.npydir")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out_dir)

    results = run_signal_screen(load_panel(args.panel_dir))
    pairs = [r.to_dict() for r in results]
    study.save_json(pairs, out / "signal_screen.json")
    study.save_csv(
        [{k: v for k, v in p.items() if k != "folds"} for p in pairs], out / "signal_screen.csv"
    )
    study.save_csv(figure_rows(results), out / "signal_atlas_price.csv", FIGURE_HEADER)
    summary = summarize_screen(results)
    study.save_json(summary, out / "signal_screen_summary.json")
    print(f"{summary['n_stable']} of {summary['n_pairs']} pairs stable; wrote results to {out}")


if __name__ == "__main__":
    main()
