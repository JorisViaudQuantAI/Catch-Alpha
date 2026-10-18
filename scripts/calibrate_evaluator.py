"""Step 1 of the paper: test the evaluator on synthetic data with known truth.

    python scripts/calibrate_evaluator.py null         # Figure 1      (~5-10 min)
    python scripts/calibrate_evaluator.py robustness   # other nulls   (~1-2 CPU-hours, 50k cells dominate)
    python scripts/calibrate_evaluator.py selection    # Table 1       (~3 CPU-hours; divide by --workers)
    python scripts/calibrate_evaluator.py true-v       # DSR variance check (seconds)
    python scripts/calibrate_evaluator.py all

Add ``--workers 4`` to use several CPU cores, and ``--quick`` for a fast smoke run
that writes to ``results/calibration/quick/`` instead of overwriting the paper files.
Each command overwrites its file in ``results/calibration/``; ``git diff`` then shows
whether you reproduced the published numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from catch_alpha import calibration

ROOT = Path(__file__).resolve().parents[1]


# Comment lines written above the data, so a rerun that reproduces the published numbers
# leaves the committed file unchanged.
HEADERS = {
    "null_robustness_grid.csv": [
        "# Zero-edge acceptance across null families x sample sizes (Section 2), 200 per cell.",
        "# Regenerate: python scripts/calibrate_evaluator.py robustness",
        "# params: ppy=252 sigma=0.01 n_trials=200 bootstrap_reps=150;"
        " student_t nu=3 (variance-normalized); garch a=0.08 b=0.90; ar1 phi=0.2",
        "# ci_lo/ci_hi: Wilson 95% intervals on the pass rate; all_status_ok: no abstention.",
    ],
    "selection_aware_campaigns.csv": [
        "# Selection among 200 equicorrelated candidates (Table 1): one row per campaign.",
        "# Regenerate: python scripts/calibrate_evaluator.py selection",
        "# params: K=200 n_obs=1800 per window ppy=252 sigma=0.01 n_trials=200"
        " bootstrap_reps=150 seed_base=5000000 t_alpha=0.05",
    ],
    "selection_aware_calibration.csv": [
        "# Selection among 200 equicorrelated candidates (Table 1): summary per cell and rule.",
        "# Regenerate: python scripts/calibrate_evaluator.py selection",
        "# params: K=200 n_obs=1800 per window ppy=252 sigma=0.01 n_trials=200"
        " bootstrap_reps=150 seed_base=5000000 t_alpha=0.05",
        "# headline_rate: null cells = false acceptances per campaign; planted cells ="
        " detections (accepted AND the winner is the planted candidate).",
        "# ci_lo/ci_hi: Wilson 95% intervals on headline_rate.",
    ],
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        for line in HEADERS.get(path.name, []):
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            )
    print(f"wrote {path.relative_to(ROOT)}")


def run_null(out: Path, workers: int, quick: bool) -> None:
    results = calibration.null_calibration(
        n_null=20 if quick else 200, n_power=8 if quick else 40, workers=workers
    )
    rows = []
    for r in results:
        lo, hi = r.wilson
        rows.append(
            {
                "batch": r.label,
                "effect_ann_sharpe": r.effect_ann_sharpe,
                "planted_mu": calibration.planted_mu(r.effect_ann_sharpe),
                "n_strategies": r.n,
                "n_pass": r.n_pass,
                "pass_rate": r.rate,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        print(f"  {r.label:5s} Sharpe {r.effect_ann_sharpe:.1f}: {r.n_pass}/{r.n} accepted")
    write_csv(out / "null_calibration.csv", rows)


def run_robustness(out: Path, workers: int, quick: bool) -> None:
    rows = calibration.robustness_grid(
        sizes=(1800,) if quick else calibration.ROBUSTNESS_SIZES,
        n_per_cell=4 if quick else 200,
        workers=workers,
    )
    for r in rows:
        print(f"  {r['family']:9s} n={r['n_obs']:6d}: {r['n_pass']}/{r['n_strategies']} accepted")
    write_csv(out / "null_robustness_grid.csv", rows)


def run_selection(out: Path, workers: int, quick: bool) -> None:
    campaigns, summary = calibration.selection_experiment(
        workers=workers,
        campaigns_per_cell=10 if quick else None,
        seed_base=9_000_000 if quick else calibration.SELECTION_SEED_BASE,
    )
    write_csv(out / "selection_aware_campaigns.csv", campaigns)
    write_csv(out / "selection_aware_calibration.csv", summary)
    for s in summary:
        if s["variant"] in ("V0", "V1", "V5"):
            print(
                f"  rho={s['rho']} S={s['effect_ann_sharpe']} {s['variant']} "
                f"({s['rule']}): {100 * s['headline_rate']:.1f}%"
            )


def run_true_v(out: Path, workers: int, quick: bool) -> None:
    result = calibration.true_v_check() | {"dsr_worked_example": calibration.dsr_worked_example()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "true_v_dsr_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


COMMANDS = {
    "null": run_null,
    "robustness": run_robustness,
    "selection": run_selection,
    "true-v": run_true_v,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--workers", type=int, default=1, help="CPU processes (default 1)")
    parser.add_argument("--quick", action="store_true", help="small smoke run, separate folder")
    args = parser.parse_args()

    out = ROOT / "results" / "calibration" / ("quick" if args.quick else "")
    for name in COMMANDS if args.command == "all" else [args.command]:
        start = time.perf_counter()
        print(f"[{name}]")
        COMMANDS[name](out, args.workers, args.quick)
        print(f"[{name}] done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
