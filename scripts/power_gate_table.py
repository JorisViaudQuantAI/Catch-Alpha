"""Supplementary figure S2: how many hourly observations Eq. (1) requires for a given target Sharpe.

    python scripts/power_gate_table.py

Rewrites ``results/real_data/power_gate.csv`` (seconds). Overlapping multi-hour returns
(horizons 24h and 168h) inflate the requirement through the ``horizon_bars`` argument.
"""

from __future__ import annotations

from pathlib import Path

from catch_alpha.stats import minimum_required_observations

OUT = Path(__file__).resolve().parents[1] / "results" / "real_data" / "power_gate.csv"
PERIODS_PER_YEAR = 8760.0  # hourly bars
HORIZONS = (1, 24, 168)
S_MIN_GRID = [round(0.25 + 0.05 * i, 2) for i in range(76)]  # 0.25 .. 4.00

HEADER = """\
# figure: F6 (power gate: required observations vs target annualized Sharpe)
# source: scripts/power_gate_table.py, catch_alpha.stats.minimum_required_observations
#   (alpha=0.05, power=0.80, 8760 hourly bars per year). horizon_h > 1 applies the
#   overlap inflation f(h,n) = 1 + 2 sum_{k<h} (1-k/h)(1-k/n) ~= h for n >> h.
# reference: 46,590 train+valid hourly returns; at s_min_annual=1.0, horizon 1h the
#   requirement is 54,160 > 46,590 -> INSUFFICIENT_DATA.
"""


def main() -> None:
    lines = ["s_min_annual,horizon_h,n_required"]
    for s_min in S_MIN_GRID:
        for horizon in HORIZONS:
            n_required = minimum_required_observations(
                periods_per_year=PERIODS_PER_YEAR,
                s_min_annual=s_min,
                alpha=0.05,
                power=0.80,
                horizon_bars=horizon,
            )
            lines.append(f"{s_min!r},{horizon},{n_required}")
    OUT.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
