"""Check every headline number of the paper against the files in ``results/``.

    python scripts/verify_paper_numbers.py

Each line prints the number as written in the paper, the value found in ``results/`` and
OK/FAIL. A value passes when, rounded to the precision the paper prints, it equals the
paper's number. Runs in about a second and needs no market data. ``tests/test_paper_numbers.py``
runs the same checks under pytest.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Check:
    section: str
    claim: str
    paper: float
    decimals: int  # precision printed in the paper
    actual: Callable[[], float]
    source: str

    def run(self) -> tuple[float, bool]:
        value = float(self.actual())
        return value, round(value, self.decimals) == round(self.paper, self.decimals)


# Table 1 of the paper, as printed: (rule, label, [null rho 0, 0.3, 0.7, detect S 1.5, 2.0]).
TABLE_1 = (
    ("V0", "uncorrected t-test, sel.", (100.0, 83.6, 48.2, 86.5, 97.5)),
    ("V1", "four checks N=200, sel.", (0.2, 0.0, 0.0, 35.0, 81.5)),
    ("V2", "four checks N=200, hold.", (0.0, 0.0, 0.0, 24.5, 81.5)),
    ("V3", "four checks N=1, sel.", (36.4, 20.6, 8.0, 74.5, 95.0)),
    ("V4", "DSR alone N=200, sel.", (0.2, 0.0, 0.0, 35.0, 82.0)),
    ("V5", "t-test alone, hold.", (5.0, 5.6, 7.6, 86.0, 97.5)),
)


def _calibration_checks() -> list[Check]:
    null = read_csv(RESULTS / "calibration" / "null_calibration.csv")
    grid = read_csv(RESULTS / "calibration" / "null_robustness_grid.csv")
    sel = read_csv(RESULTS / "calibration" / "selection_aware_calibration.csv")
    true_v = read_json(RESULTS / "calibration" / "true_v_dsr_check.json")

    def power(effect: float) -> float:
        row = next(
            r for r in null if r["batch"] == "power" and float(r["effect_ann_sharpe"]) == effect
        )
        return 100 * int(row["n_pass"]) / int(row["n_strategies"])

    def rate(rho: float, effect: float, rule: str) -> float:
        row = next(
            r
            for r in sel
            if float(r["rho"]) == rho
            and float(r["effect_ann_sharpe"]) == effect
            and r["variant"] == rule
        )
        return 100 * float(row["headline_rate"])

    def null_rates(rule: str) -> list[float]:
        return [rate(rho, 0.0, rule) for rho in (0.0, 0.3, 0.7)]

    src_null, src_sel = (
        "calibration/null_calibration.csv",
        "calibration/selection_aware_calibration.csv",
    )
    checks = [
        Check(
            "2",
            "Gaussian zero-edge strategies accepted (of 200)",
            0,
            0,
            lambda: int(null[0]["n_pass"]),
            src_null,
        ),
        Check(
            "2",
            "Wilson 95% upper bound on the null rate (%)",
            1.9,
            1,
            lambda: 100 * float(null[0]["ci_hi"]),
            src_null,
        ),
        Check(
            "2",
            "Accepted across the 12 robustness settings (of 2,400)",
            0,
            0,
            lambda: sum(int(r["n_pass"]) for r in grid),
            "calibration/null_robustness_grid.csv",
        ),
        Check(
            "2",
            "Robustness strategies evaluated",
            2400,
            0,
            lambda: sum(int(r["n_strategies"]) for r in grid),
            "calibration/null_robustness_grid.csv",
        ),
    ]
    for effect, paper in ((0.5, 0), (1.0, 0), (1.5, 30), (2.0, 85), (2.5, 97.5), (3.0, 100)):
        checks.append(
            Check(
                "2",
                f"Detection at planted Sharpe {effect} (%)",
                paper,
                1,
                lambda e=effect: power(e),
                src_null,
            )
        )
    # Table 1, every cell as printed (one decimal): rows are rules, columns are the
    # null cells at correlation 0 / 0.3 / 0.7 and the detection cells S = 1.5 / 2.0.
    columns = ((0.0, 0.0), (0.3, 0.0), (0.7, 0.0), (0.0, 1.5), (0.0, 2.0))
    for rule, label, printed in TABLE_1:
        for (rho, effect), paper in zip(columns, printed, strict=True):
            cell = f"rho={rho}" if effect == 0.0 else f"S={effect}"
            checks.append(
                Check(
                    "2",
                    f"Table 1, {label}, {cell} (%)",
                    paper,
                    1,
                    lambda rho=rho, effect=effect, rule=rule: rate(rho, effect, rule),
                    src_sel,
                )
            )
    checks += [
        Check(
            "2",
            "Four checks N=200, zero-edge winners accepted (of 1,500)",
            1,
            0,
            lambda: sum(5 * r for r in null_rates("V1")),
            src_sel,
        ),
        Check(
            "2",
            "Measured cross-trial Sharpe variance V (x1e-4)",
            4.6,
            1,
            lambda: 1e4 * true_v["v_true_cross_trial"],
            "calibration/true_v_dsr_check.json",
        ),
        Check(
            "2",
            "Approximation of V used by DSR (x1e-4)",
            5.6,
            1,
            lambda: 1e4 * true_v["v_proxy_mean_sigma_sr_sq"],
            "calibration/true_v_dsr_check.json",
        ),
        Check(
            "2",
            "DSR worked example, N = 3",
            0.99,
            2,
            lambda: true_v["dsr_worked_example"]["N=3"],
            "calibration/true_v_dsr_check.json",
        ),
        Check(
            "2",
            "DSR worked example, N = 347",
            0.63,
            2,
            lambda: true_v["dsr_worked_example"]["N=347"],
            "calibration/true_v_dsr_check.json",
        ),
    ]
    return checks


def _learning_checks() -> list[Check]:
    rows = {
        (r["policy"], r["cell"]): r
        for r in read_csv(RESULTS / "learning" / "capability_common_flat.csv")
    }
    src = "learning/capability_common_flat.csv"
    gp_worker = read_json(RESULTS / "learning" / "workers" / "E240_bc_lam1_s0.json")
    gp_gate = gp_worker["summary"]["denominator_gate"]

    def val(policy: str, cell: str, key: str) -> float:
        return float(rows[(policy, cell)][key])

    return [
        Check(
            "3",
            "BC, no cost: median score relative to GP",
            0.900,
            3,
            lambda: val("BC", "no_cost", "median_recovery"),
            src,
        ),
        Check(
            "3",
            "IQL, no cost: median score relative to GP",
            0.731,
            3,
            lambda: val("IQL", "no_cost", "median_recovery"),
            src,
        ),
        Check(
            "3",
            "CQL, no cost: median score relative to GP",
            0.335,
            3,
            lambda: val("CQL", "no_cost", "median_recovery"),
            src,
        ),
        Check(
            "3",
            "BC, with costs: median mean profit",
            -0.995,
            3,
            lambda: val("BC", "cost", "median_mean_pnl"),
            src,
        ),
        Check(
            "3",
            "IQL, with costs: median mean profit",
            -4.036,
            3,
            lambda: val("IQL", "cost", "median_mean_pnl"),
            src,
        ),
        Check(
            "3",
            "BC, with costs: losing fits (of 10)",
            10,
            0,
            lambda: val("BC", "cost", "negative_training_seeds"),
            src,
        ),
        Check(
            "3",
            "IQL, with costs: losing fits (of 10)",
            10,
            0,
            lambda: val("IQL", "cost", "negative_training_seeds"),
            src,
        ),
        Check(
            "3",
            "BC median turnover relative to GP",
            6.03,
            2,
            lambda: val("BC", "cost", "median_turnover_ratio"),
            src,
        ),
        Check(
            "3",
            "IQL median turnover relative to GP",
            11.97,
            2,
            lambda: val("IQL", "cost", "median_turnover_ratio"),
            src,
        ),
        Check(
            "3",
            "BC + previous position: median mean profit",
            -0.960,
            3,
            lambda: val("BC + previous position", "cost", "median_mean_pnl"),
            src,
        ),
        Check(
            "3",
            "BC + previous position: losing fits (of 10)",
            10,
            0,
            lambda: val("BC + previous position", "cost", "negative_training_seeds"),
            src,
        ),
        Check(
            "3",
            "BC + previous position: median turnover",
            5.92,
            2,
            lambda: val("BC + previous position", "cost", "median_turnover_ratio"),
            src,
        ),
        Check(
            "3",
            "GP reference: one-sided 95% lower bound",
            0.0395,
            4,
            lambda: gp_gate["denominator_one_sided_low"],
            "learning/workers/E240_bc_lam1_s0.json",
        ),
        Check(
            "3",
            "GP reference mean profit with costs",
            0.0461,
            4,
            lambda: val("GP reference", "cost", "median_mean_pnl"),
            src,
        ),
    ]


def _real_data_checks() -> list[Check]:
    rd = RESULTS / "real_data"
    folds = read_json(rd / "strategy_folds.json")
    strat = folds["strategies"]
    screen = {(r["signal"], int(r["horizon_h"])): r for r in read_csv(rd / "signal_screen.csv")}
    summary = read_json(rd / "signal_screen_summary.json")
    fees = read_json(rd / "fee_sensitivity.json")
    fee_rows = {float(r["fee_bps"]): r for r in read_csv(rd / "cash_and_carry_fee_sensitivity.csv")}
    final = read_json(rd / "final_period.json")
    weekly = read_json(rd / "weekly_momentum.json")
    pbo = read_json(rd / "pbo_reversal.json")
    ledger = read_json(rd / "trial_ledger_public_v3.json")["records"]
    market_null = read_json(rd / "market_null_check.json")
    onchain = {
        (r["candidate"], int(r["horizon_h"])): r for r in read_csv(rd / "onchain_mcap_residual.csv")
    }
    hash24 = onchain[("HashRate_level", 24)]
    addr72 = onchain[("AdrActCnt_level", 72)]
    z = NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)

    cc = strat["cash_and_carry"]["full_window"]
    checks = [
        Check(
            "4",
            "Train+valid hourly returns",
            46590,
            0,
            lambda: folds["power_gate"]["n_obs"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Required for annual Sharpe 1 (Eq. 1)",
            54160,
            0,
            lambda: folds["power_gate"]["n_required"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Price signal-horizon pairs screened",
            128,
            0,
            lambda: summary["n_pairs"],
            "real_data/signal_screen_summary.json",
        ),
        Check(
            "4",
            "Stable pairs",
            69,
            0,
            lambda: summary["n_stable"],
            "real_data/signal_screen_summary.json",
        ),
        Check(
            "4",
            "24h momentum -> next hour: rank IC",
            -0.032,
            3,
            lambda: float(screen[("momentum_24h", 1)]["mean_rank_ic"]),
            "real_data/signal_screen.csv",
        ),
        Check(
            "4",
            "24h momentum -> next hour: median fold t",
            -7.79,
            2,
            lambda: float(screen[("momentum_24h", 1)]["median_fold_t"]),
            "real_data/signal_screen.csv",
        ),
        Check(
            "4",
            "24h momentum -> next hour: stable folds (of 6)",
            6,
            0,
            lambda: int(screen[("momentum_24h", 1)]["folds_stable"]),
            "real_data/signal_screen.csv",
        ),
        Check(
            "4",
            "Inverse volatility at 8h: rank IC",
            0.046,
            3,
            lambda: float(screen[("inv_vol_168h", 8)]["mean_rank_ic"]),
            "real_data/signal_screen.csv",
        ),
        Check(
            "4",
            "Inverse volatility at 8h: stable folds (of 6)",
            6,
            0,
            lambda: int(screen[("inv_vol_168h", 8)]["folds_stable"]),
            "real_data/signal_screen.csv",
        ),
        Check(
            "4",
            "Long-loser/short-winner, 24h, gross",
            -1.45,
            2,
            lambda: strat["reversal_24h"]["gross_price_pnl"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Long-loser/short-winner, 8h, gross",
            -3.02,
            2,
            lambda: strat["reversal_8h"]["gross_price_pnl"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Mirror trend portfolio, 24h, gross",
            1.45,
            2,
            lambda: strat["reversal_24h"]["mirror_trend_gross_price_pnl"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Reversal folds with positive lower bound (24h + 8h)",
            0,
            0,
            lambda: (
                strat["reversal_24h"]["n_folds_ci_lb_gt_zero"]
                + strat["reversal_8h"]["n_folds_ci_lb_gt_zero"]
            ),
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Raw carry: folds with positive net lower bound",
            0,
            0,
            lambda: strat["raw_carry"]["n_folds_ci_lb_gt_zero"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Funding positive in folds (of 6)",
            6,
            0,
            lambda: strat["raw_carry"]["n_folds_funding_positive"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Cash-and-carry funding received",
            0.491,
            3,
            lambda: cc["funding"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Cash-and-carry basis drift",
            -0.023,
            3,
            lambda: cc["price"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Cash-and-carry two-leg fees",
            -0.732,
            3,
            lambda: cc["cost"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Cash-and-carry net",
            -0.263,
            3,
            lambda: cc["total"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Fee sweep: net at 0 bp",
            0.468,
            3,
            lambda: float(fee_rows[0.0]["full_window_sum_total"]),
            "real_data/cash_and_carry_fee_sensitivity.csv",
        ),
        Check(
            "4",
            "Fee sweep: net at 10 bp",
            -0.995,
            3,
            lambda: float(fee_rows[10.0]["full_window_sum_total"]),
            "real_data/cash_and_carry_fee_sensitivity.csv",
        ),
        Check(
            "4",
            "Fee sweep: net at 2 bp (maker)",
            0.176,
            3,
            lambda: float(fee_rows[2.0]["full_window_sum_total"]),
            "real_data/cash_and_carry_fee_sensitivity.csv",
        ),
        Check(
            "4",
            "Fee sweep: folds with positive lower bound at 2 bp",
            1,
            0,
            lambda: int(fee_rows[2.0]["n_folds_mean_ci_lb_gt_zero"]),
            "real_data/cash_and_carry_fee_sensitivity.csv",
        ),
        Check(
            "4",
            "Fee sweep: folds with positive lower bound at 0 bp",
            5,
            0,
            lambda: int(fee_rows[0.0]["n_folds_mean_ci_lb_gt_zero"]),
            "real_data/cash_and_carry_fee_sensitivity.csv",
        ),
        Check(
            "4",
            "Break-even fee (bp per leg)",
            3.2,
            1,
            lambda: fees["break_even_fee_bps"],
            "real_data/fee_sensitivity.json",
        ),
        Check(
            "4",
            "Beta-hedged carry: funding share of gross components",
            0.443,
            3,
            lambda: strat["hedged_carry"]["funding_share"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "4",
            "Beta-hedged carry: folds with positive lower bound",
            0,
            0,
            lambda: strat["hedged_carry"]["n_folds_ci_lb_gt_zero"],
            "real_data/strategy_folds.json",
        ),
        Check(
            "2",
            "Market-derived null: required weekly obs (Eq. 1, S_min 1.5)",
            144,
            0,
            lambda: math.ceil((z / 1.5) ** 2 * 365.25 / 7),
            "Eq. 1 (market_null_check.json: 276 weekly obs)",
        ),
        Check(
            "2",
            "Market-derived null: weekly observations",
            276,
            0,
            lambda: market_null["n_weekly_obs"],
            "real_data/market_null_check.json",
        ),
        Check(
            "4",
            "On-chain residual: hash rate 24h rank IC",
            0.049,
            3,
            lambda: float(hash24["neutral_mean_ic"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "4",
            "On-chain residual: hash rate 24h stable folds",
            4,
            0,
            lambda: int(hash24["neutral_folds_stable"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "4",
            "On-chain residual: hash rate 24h return difference",
            0.0017,
            4,
            lambda: float(hash24["neutral_net_spread"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "4",
            "On-chain residual: active addresses 72h rank IC",
            0.061,
            3,
            lambda: float(addr72["neutral_mean_ic"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "4",
            "On-chain residual: active addresses 72h stable folds",
            4,
            0,
            lambda: int(addr72["neutral_folds_stable"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "4",
            "On-chain residual: active addresses 72h return difference",
            0.0002,
            4,
            lambda: float(addr72["neutral_net_spread"]),
            "real_data/onchain_mcap_residual.csv",
        ),
        Check(
            "2",
            "Market-derived null: accepted (of 200)",
            0,
            0,
            lambda: market_null["n_full_gate_pass"],
            "real_data/market_null_check.json",
        ),
        Check(
            "2",
            "Ledger records on the real-data panel",
            353,
            0,
            lambda: len(ledger),
            "real_data/trial_ledger_public_v3.json",
        ),
        Check(
            "5",
            "Ledger final-period (test) records",
            1,
            0,
            lambda: sum(r["split"] == "test" for r in ledger),
            "real_data/trial_ledger_public_v3.json",
        ),
        Check(
            "5",
            "PBO on the reversal matrix",
            0.00,
            2,
            lambda: pbo["pbo"],
            "real_data/pbo_reversal.json",
        ),
        Check("5", "PBO splits", 20, 0, lambda: pbo["n_splits"], "real_data/pbo_reversal.json"),
        Check(
            "5",
            "Winning variant mean per-fold annual Sharpe",
            -1.98,
            2,
            lambda: pbo["in_sample_best_mean_fold_sharpe"],
            "real_data/pbo_reversal.json",
        ),
        Check(
            "5",
            "Weekly momentum: weekly observations",
            276,
            0,
            lambda: weekly["n_weekly_returns"],
            "real_data/weekly_momentum.json",
        ),
        Check(
            "5",
            "Weekly momentum: required",
            323,
            0,
            lambda: weekly["n_required"],
            "real_data/weekly_momentum.json",
        ),
        Check(
            "5",
            "Weekly momentum: mean net per week (%)",
            0.56,
            2,
            lambda: 100 * weekly["mean_weekly_net"],
            "real_data/weekly_momentum.json",
        ),
        Check(
            "5",
            "Weekly momentum: bootstrap lower bound (x1e-5)",
            6,
            0,
            lambda: 1e5 * weekly["mean_ci_95"][0],
            "real_data/weekly_momentum.json",
        ),
        Check(
            "5",
            "Final period: hourly returns",
            12648,
            0,
            lambda: final["window"]["n_returns"],
            "real_data/final_period.json",
        ),
        Check(
            "5",
            "Final period: books with interval above zero",
            0,
            0,
            lambda: sum(b["ci_above_zero"] for b in final["books"].values()),
            "real_data/final_period.json",
        ),
        Check(
            "5",
            "Final period: cash-and-carry mean per hour (x1e-5)",
            -2.88,
            2,
            lambda: 1e5 * final["books"]["cash_and_carry"]["mean_per_hour"],
            "real_data/final_period.json",
        ),
        Check(
            "5",
            "Final period: cash-and-carry interval entirely below zero",
            1,
            0,
            lambda: final["books"]["cash_and_carry"]["ci_below_zero"],
            "real_data/final_period.json",
        ),
        Check(
            "5",
            "Final period: equal-weight portfolio interval includes zero",
            1,
            0,
            lambda: (
                not (
                    final["equal_weight_portfolio"]["ci_above_zero"]
                    or final["equal_weight_portfolio"]["ci_below_zero"]
                )
            ),
            "real_data/final_period.json",
        ),
    ]
    return checks


def all_checks() -> list[Check]:
    return _calibration_checks() + _learning_checks() + _real_data_checks()


def main() -> int:
    failures = 0
    checks = all_checks()
    section = None
    names = {
        "2": "Evaluator calibration (Section 2)",
        "3": "Controlled learning test (Section 3)",
        "4": "Real-data study (Section 4)",
        "5": "Final period and illustrations (Section 5)",
    }
    for check in sorted(checks, key=lambda c: c.section):
        if check.section != section:
            section = check.section
            print(f"\n{names[section]}")
        value, ok = check.run()
        failures += not ok
        print(
            f"  {'OK  ' if ok else 'FAIL'} {check.claim:60s} paper {check.paper:>10g}  "
            f"results {value:>12.6g}"
        )
    print(f"\n{len(checks) - failures}/{len(checks)} paper numbers reproduced from results/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
