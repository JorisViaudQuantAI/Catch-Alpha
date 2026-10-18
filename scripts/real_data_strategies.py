#!/usr/bin/env python
"""Real-data portfolio study on train+valid (paper Section 4; Figure 4 and figure S1; gate outcomes).

Writes to ``--out-dir`` (default ``results/real_data``):

* ``strategy_folds.json``        six-fold mechanics of the four frozen strategies
* ``carry_decomposition.csv``    figure S1 input (paper schema; needs the spot lake)
* ``cash_and_carry_fee_sensitivity.csv`` Figure 4 input (paper schema; needs the spot lake)
* ``fee_sensitivity.json``       the same sweep with the break-even fee
* ``gate_verdicts.json``         the full evaluator on each strategy
* ``pbo_reversal.json``          PBO on the 2 x 6 reversal score matrix
* ``funding_timestamp_diagnostic.json``

Cash-and-carry needs the Binance spot lake (``--spot-lake``, the directory that contains
``1h/symbol=.../``); the script stops with an error if it is missing or empty.

    python scripts/real_data_strategies.py --panel-dir data/public_v3 --spot-lake data/lake/binance_spot
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
    parser.add_argument("--reps", type=int, default=study.BOOTSTRAP_REPS)
    args = parser.parse_args()
    out = Path(args.out_dir)

    panel = load_panel(args.panel_dir)  # train + valid only
    spot = load_spot_panel(args.spot_lake, panel)

    folds = study.strategy_study(panel, *spot, reps=args.reps)
    study.save_json(folds, out / "strategy_folds.json")
    # Figure 4 and figure S1 inputs (cash-and-carry)
    study.save_csv(
        study.carry_decomposition(folds),
        out / "carry_decomposition.csv",
        study.CARRY_DECOMPOSITION_HEADER,
    )
    fees = study.fee_sensitivity(panel, *spot, reps=args.reps)
    study.save_json(fees, out / "fee_sensitivity.json")
    study.save_csv(
        fees["paper_rows"],
        out / "cash_and_carry_fee_sensitivity.csv",
        (
            "# figure: F5 cash-and-carry net return vs fee per leg (train+valid)",
            "# source: catch_alpha.realdata.study.fee_sensitivity; floats copied with repr",
        ),
    )
    study.save_json(study.gate_verdicts(panel, *spot), out / "gate_verdicts.json")
    study.save_json(study.pbo_reversal(panel), out / "pbo_reversal.json")
    study.save_json(
        study.funding_timestamp_diagnostic(panel), out / "funding_timestamp_diagnostic.json"
    )
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
