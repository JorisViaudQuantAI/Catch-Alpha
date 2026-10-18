"""Render the paper figures from the result files in ``results/``.

    python scripts/make_figures.py            # all figures -> figures/*.pdf|png
    python scripts/make_figures.py --only f1  # one figure

Inputs (all committed, so this runs in seconds on a fresh clone):

    F1 null calibration         results/calibration/null_calibration.csv
    F2 learning test            results/learning/capability_common_flat.csv
    F3 signal screen            results/real_data/signal_atlas_{price,exog}.csv
    F4 fee sensitivity          results/real_data/cash_and_carry_fee_sensitivity.csv
    S1 carry decomposition      results/real_data/carry_decomposition.csv
    S2 minimum-length curves    results/real_data/power_gate.csv

F1-F4 are the paper's Figures 1-4. S1 and S2 are supplementary views of numbers stated
in the paper's text (the cash-and-carry decomposition of Section 4 and Eq. 1).
No randomness and no network: the same CSVs always give the same pictures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# --- design system ----------------------------------------------------------
SURFACE = "#ffffff"
INK = "#1b1b1a"
INK2 = "#57564f"
MUTED = "#8c8b83"
GRID = "#e5e9ed"
BASELINE = "#aeb6bd"

# categorical series colors IN FIXED ORDER (never cycle / reorder)
CAT = ["#365f7c", "#4f7f72", "#9a7a3d", "#5f7a4e", "#6f6385", "#a15a4c"]
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED = CAT

# diverging sign ramp (blue positive <-> red negative, neutral gray midpoint)
POS = BLUE
NEG = RED
NEUTRAL = "#9a988e"

# entity -> color (color follows the entity across figures)
ENTITY_COLOR = {
    "gp": BLUE,
    "bc_repo": YELLOW,
    "iql_repo": GREEN,
    "cql_repo": VIOLET,
    "bc_repo_pos": YELLOW,  # a variant of the BC entity; distinguished by marker
}
ENTITY_LABEL = {
    "gp": "GP reference",
    "bc_repo": "BC",
    "iql_repo": "IQL",
    "cql_repo": "CQL",
    "bc_repo_pos": "BC (position-aware)",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.dpi": 120,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        header: list[str] | None = None
        for raw in reader:
            if not raw:
                continue
            if raw[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in raw]
                continue
            rows.append({header[i]: raw[i].strip() for i in range(min(len(header), len(raw)))})
    return rows


def _f(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _fv(x: str) -> float:
    """Required numeric CSV field: like ``_f`` but a missing value is an error."""
    v = _f(x)
    if v is None:
        raise ValueError(f"missing numeric value: {x!r}")
    return v


def _style_ax(ax: Any) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.tick_params(length=3, color=MUTED, direction="out")
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)


def _panel_letter(ax: Any, letter: str) -> None:
    """Bold panel letter above the top-left corner of the axes (journal style)."""
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-34, 5),
        textcoords="offset points",
        color=INK,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="baseline",
        annotation_clip=False,
    )


def _ci_bar(ax: Any, i: int, lo: float, hi: float, color: str) -> None:
    """Horizontal 95% CI bar with end caps for one forest-plot row."""
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.0, solid_capstyle="butt", zorder=3)
    ax.plot([lo, lo], [i - 0.17, i + 0.17], color=color, linewidth=1.5, zorder=3)
    ax.plot([hi, hi], [i - 0.17, i + 0.17], color=color, linewidth=1.5, zorder=3)


def _save(fig: Any, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf + {name}.png")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (robust at p=0)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --- F1: null calibration & detection power ---------------------------------
def fig_f1() -> None:
    path = RESULTS / "calibration" / "null_calibration.csv"
    if not path.exists():
        print(
            "  [F1] null_calibration.csv absent -> run scripts/calibrate_evaluator.py null first; skipping"
        )
        return
    rows = _read_csv(path)
    null_rows = [r for r in rows if r["batch"] == "null"]
    power_rows = sorted(
        (r for r in rows if r["batch"] == "power" and (_f(r["effect_ann_sharpe"]) or 0.0) > 0.0),
        key=lambda r: _fv(r["effect_ann_sharpe"]),
    )

    fig, (axa, axb) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        gridspec_kw={"width_ratios": [0.72, 2.0]},
    )

    # Panel A: null false-positive rate with binomial 95% CI
    for r in null_rows:
        n = int(float(r["n_strategies"]))
        k = int(float(r["n_pass"]))
        rate = k / n if n else 0.0
        lo, hi = _wilson_ci(k, n)
        axa.errorbar(
            [0],
            [rate],
            yerr=[[rate - lo], [hi - rate]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.4,
            capsize=4,
            markersize=7,
            zorder=5,
            clip_on=False,
        )
        axa.annotate(
            f"{k}/{n}",
            (0, rate),
            textcoords="offset points",
            xytext=(-33, 8),
            color=INK,
            fontsize=8,
        )
    axa.axhspan(0.0, 0.05, color=GRID, alpha=0.35, zorder=0)
    axa.axhline(0.05, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    axa.annotate(
        "5% reference line",
        (0.0, 0.05),
        textcoords="offset points",
        xytext=(-4, 6),
        ha="center",
        color=INK2,
        fontsize=7,
    )
    axa.set_xlim(-1, 1)
    axa.set_ylim(0.0, 0.12)
    axa.set_xticks([0])
    axa.set_xticklabels(["zero-edge\nstrategies"])
    axa.set_ylabel("four-check pass rate")
    axa.set_title("Null calibration", color=INK, loc="left")
    _style_ax(axa)

    # Panel B: detection power vs planted annualized-Sharpe effect size
    xs = [_fv(r["effect_ann_sharpe"]) for r in power_rows]
    ys = [_fv(r["pass_rate"]) for r in power_rows]
    los = [_fv(r["ci_lo"]) for r in power_rows]
    his = [_fv(r["ci_hi"]) for r in power_rows]
    yerr_lo = [y - lo for y, lo in zip(ys, los, strict=True)]
    yerr_hi = [hi - y for y, hi in zip(ys, his, strict=True)]
    axb.plot(xs, ys, color=BLUE, marker="o", markersize=4, zorder=4)
    axb.errorbar(
        xs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor=BLUE,
        elinewidth=1.0,
        capsize=3,
        alpha=0.9,
        zorder=3,
    )
    axb.axhline(0.05, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    axb.annotate(
        "5%",
        (max(xs), 0.05),
        textcoords="offset points",
        xytext=(4, -2),
        color=INK2,
        fontsize=7,
    )
    axb.axhline(0.80, color=GRID, linewidth=1.0, linestyle=":", zorder=1)
    power_kw: dict[str, Any] = {
        "xy": (min(xs), 0.80),
        "textcoords": "offset points",
        "xytext": (2, 4),
        "color": INK2,
        "fontsize": 7,
    }
    axb.annotate("80% power", **power_kw)
    # Shade the range where the simple mean-test length check fails. This is
    # not an end-to-end threshold for the complete decision rule.
    simple_floor = (6.18 * 252.0 / 1800.0) ** 0.5
    axb.axvspan(min(xs) - 0.2, simple_floor, color=GRID, alpha=0.35, zorder=0)
    axb.axvline(simple_floor, color=INK2, linewidth=1.0, linestyle="-.", zorder=2)
    short_kw: dict[str, Any] = {
        "xy": (simple_floor, 0.56),
        "textcoords": "offset points",
        "xytext": (-7, 0),
        "ha": "right",
        "color": INK2,
        "fontsize": 7,
    }
    axb.annotate("too short for\nsimple mean test", **short_kw)
    n_per_effect = int(float(power_rows[0]["n_strategies"]))
    dsr_kw: dict[str, Any] = {
        "xy": (max(xs), 0.16),
        "textcoords": "offset points",
        "xytext": (0, 0),
        "ha": "right",
        "color": INK2,
        "fontsize": 7,
    }
    dsr_fmt = "DSR counts N = 200 attempts;\n{n} draws per planted edge"
    axb.annotate(dsr_fmt.format(n=n_per_effect), **dsr_kw)
    # Rate labels sit left of each marker, clear of the rising line and of the
    # upper error-bar cap; a white box masks the grid underneath.
    for x, y in zip(xs, ys, strict=True):
        if y > 0:
            label = f"{y:.0%}" if abs(y * 100 - round(y * 100)) < 1e-9 else f"{y:.1%}"
            dx, dy, ha, va = -9, 4, "right", "bottom"
            axb.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                color=INK,
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9},
                zorder=6,
            )
    axb.set_xlim(*(min(xs) - 0.2, max(xs) + 0.25))
    axb.set_xlabel("planted edge (annualized Sharpe)")
    axb.set_ylabel("detection rate")
    axb.set_ylim(*(-0.03, 1.08))
    axb.set_title("Detection power", color=INK, loc="left")
    _style_ax(axb)

    fig.tight_layout()
    _save(fig, "F1_null_calibration")


# --- F2: capability curve ---------------------------------------------------
def fig_f2() -> None:
    rows = _read_csv(RESULTS / "learning" / "capability_common_flat.csv")

    def row(policy: str, cell: str) -> dict[str, str]:
        return next(r for r in rows if r["policy"] == policy and r["cell"] == cell)

    fig, (ax0, ax1) = plt.subplots(
        *(1, 2),
        figsize=(7.2, 3.5),
        gridspec_kw={"width_ratios": [0.95, 1.45]},
    )

    # Panel A: prerequisite signal-learning check without trading costs.
    policies0 = ["BC", "IQL", "CQL", "GP reference"]
    values0 = [float(row(p, "no_cost")["median_recovery"]) for p in policies0]
    colors0 = [YELLOW, GREEN, MUTED, BLUE]
    y0 = list(range(len(policies0)))
    bars0 = ax0.barh(y0, values0, color=colors0, height=0.62, zorder=3)
    ax0.axvline(0.50, color=BASELINE, linewidth=1.1, linestyle=":", zorder=2)
    threshold_kw: dict[str, Any] = {
        "xy": (0.50, -0.55),
        "xytext": (4, 0),
        "textcoords": "offset points",
        "color": INK2,
        "fontsize": 7.2,
        "va": "center",
    }
    ax0.annotate("required 0.50", **threshold_kw)
    for bar, value in zip(bars0, values0, strict=True):
        ax0.annotate(
            f"{value:.3f}",
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK,
        )
    ax0.set_yticks(y0, labels=policies0)
    ax0.invert_yaxis()
    ax0.set_xlim(*(0, 1.15))
    ax0.set_xlabel("median score relative to GP")
    ax0.set_title("No trading costs", color=INK, loc="left")
    _style_ax(ax0)
    _panel_letter(ax0, "A")

    # Panel B: economic score with quadratic costs. CQL is gray because its
    # no-cost prerequisite failed, so the value is descriptive only.
    policies1 = ["BC", "IQL", "CQL", "BC + previous position", "GP reference"]
    rows1 = [row(p, "cost") for p in policies1]
    values1 = [float(r["median_mean_pnl"]) for r in rows1]
    colors1 = [YELLOW, GREEN, "#d9d9d4", SURFACE, BLUE]
    edges1 = [YELLOW, GREEN, INK2, YELLOW, BLUE]
    widths1 = [0.0, 0.0, 0.9, 1.4, 0.0]
    y1 = list(range(len(policies1)))
    bars1 = ax1.barh(y1, values1, color=colors1, height=0.62, zorder=3)
    for bar, edge, width in zip(bars1, edges1, widths1, strict=True):
        bar.set_edgecolor(edge)
        bar.set_linewidth(width)
    ax1.axvline(0.0, color=INK2, linewidth=1.0, zorder=2)
    for i, value in enumerate(values1):
        value_txt = f"{value:+.3f}"
        ax1.annotate(
            value_txt,
            (value, i),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.4,
            color=INK if i != 2 else INK2,
        )
    ax1.legend(
        handles=[
            Patch(
                facecolor="#d9d9d4",
                edgecolor=INK2,
                linewidth=0.9,
                label="CQL: no conclusion under costs",
            )
        ],
        fontsize=7.0,
        loc="lower left",
    )
    ax1.set_yticks(y1, labels=policies1)
    ax1.invert_yaxis()
    ax1.set_xlim(*(-6.15, 0.85))
    ax1.set_xlabel("median mean profit after cost")
    ax1.set_title("Quadratic trading cost ($\\lambda=1$)", color=INK, loc="left")
    _style_ax(ax1)
    _panel_letter(ax1, "B")

    fig.tight_layout(w_pad=2.0)
    _save(fig, "F2_capability_curve")


# plain-English display names for atlas signals (paper is self-contained;
# artifact code names stay in the CSVs)
SIGNAL_LABEL = {
    "feat_ret_1": "1h return",
    "feat_compound_return_8": "8h compound return",
    "feat_rsi": "RSI",
    "momentum_24h": "24h momentum",
    "momentum_72h": "72h momentum",
    "feat_realized_vol": "realized vol",
    "feat_spread_proxy": "spread proxy",
    "feat_ema_fast": "fast EMA",
    "inv_vol_168h": "inverse vol (168h)",
    "feat_illiq": "illiquidity",
    "carry_funding": "funding carry",
    "meanrev_168h": "mean reversion (168h)",
    "perp_spot_basis": "perp-spot basis",
    "basis_change_24h": "24h basis change",
    "funding_rate_last": "last funding rate",
    "HashRate_level": "hash-rate level",
    "CapMrktCurUSD_level": "market-cap level",
    "AdrActCnt_level": "active-address level",
    "TxCnt_level": "transaction-count level",
    "taker_imbalance": "taker imbalance",
    "term_slope_btc": "BTC term slope",
    "dvol_market_change": "DVOL change",
    "xfunding_dispersion": "cross-venue funding disp.",
}


def _sig_label(name: str) -> str:
    return SIGNAL_LABEL.get(name, name.replace("_", " "))


def _atlas_label(r: dict[str, str]) -> str:
    """Row label for F3, e.g. "24h momentum @1h"."""
    name = {}.get(r["signal"], _sig_label(r["signal"]))
    return str("{name} @{h}h".format(name=name, h=r["horizon_h"]))


# --- F3: signal atlas forest plot -------------------------------------------
def fig_f3() -> None:
    price = _read_csv(RESULTS / "real_data" / "signal_atlas_price.csv")
    exog_all = _read_csv(RESULTS / "real_data" / "signal_atlas_exog.csv")
    # Keep panel B comparable: regime/time-series diagnostics are reported in
    # the table/artifact, not plotted on the cross-sectional rank-IC axis.
    exog = [r for r in exog_all if r["signal"] not in {"term_slope_btc", "dvol_market_change"}]

    # sort each panel by IC ascending so reversal (negative) sits at the bottom
    price = sorted(price, key=lambda r: _fv(r["mean_rank_ic"]))
    exog = sorted(exog, key=lambda r: _fv(r["mean_rank_ic"]))

    fig, (axp, axe) = plt.subplots(
        *(1, 2),
        figsize=(7.2, 5.3),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    # Panel A: price families with derived NW CI
    ylabels = []
    for i, r in enumerate(price):
        ic = _fv(r["mean_rank_ic"])
        t = _fv(r["nw_t"])
        se = abs(ic) / abs(t) if t else 0.0
        lo, hi = ic - 1.96 * se, ic + 1.96 * se
        color = POS if ic >= 0 else NEG
        _ci_bar(axp, i, lo, hi, color)
        axp.plot([ic], [i], "o", color=color, markersize=5.8, zorder=4)
        ylabels.append(_atlas_label(r))
        axp.annotate(
            f"{int(float(r['folds_stable']))}/{int(float(r['folds_total']))}",
            (hi, i),
            textcoords="offset points",
            xytext=(6, -3),
            color=INK2,
            fontsize=7.4,
        )
    axp.axvline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    axp.set_yticks(range(len(price)))
    axp.set_yticklabels(ylabels, fontsize=7.8, color=INK)
    axp.set_xlabel("rank correlation with future return (approx. interval)")
    axp.set_title("Price families", color=INK, loc="left")
    axp.set_ylim(-0.7, len(price) - 0.3)
    axp.margins(x=0.18)
    _style_ax(axp)
    _panel_letter(axp, "A")

    # Panel B: exogenous families (point IC + fold-stability; 95% NW CI pooled from the
    # fold-level artifact CIs where available — see signal_atlas_exog.csv header)
    ylabels = []
    for i, r in enumerate(exog):
        ic = _fv(r["mean_rank_ic"])
        fs = int(float(r["folds_stable"]))
        color = POS if ic >= 0 else NEG
        stable = fs >= 4
        ci_lo, ci_hi = _f(r.get("ci_low", "")), _f(r.get("ci_high", ""))
        has_ci = r.get("ci_available") == "true" and ci_lo is not None and ci_hi is not None
        if ci_lo is not None and ci_hi is not None and has_ci:
            _ci_bar(axe, i, ci_lo, ci_hi, color)
        axe.plot(
            [ic],
            [i],
            "o",
            color=color,
            markersize=6.6 if stable else 5.8,
            markerfacecolor=color if stable else "none",
            zorder=4,
        )
        ylabels.append(_atlas_label(r))
        axe.annotate(
            f"{fs}/{int(float(r['folds_total']))}",
            (ci_hi if has_ci and ci_hi is not None else ic, i),
            textcoords="offset points",
            xytext=(6, -3),
            color=INK2,
            fontsize=7.4,
        )
    axe.axvline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    axe.set_yticks(range(len(exog)))
    axe.set_yticklabels(ylabels, fontsize=7.8, color=INK)
    axe.set_xlabel("rank correlation (95% CI, folds pooled)")
    axe.set_title("Exogenous families", color=INK, loc="left")
    axe.set_ylim(-0.7, len(exog) - 0.3)
    axe.margins(x=0.22)
    _style_ax(axe)
    _panel_letter(axe, "B")

    # legend: sign + fold-stability convention
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POS,
            markersize=6,
            label="positive relation",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NEG,
            markersize=6,
            label="negative relation (reversal)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=INK2,
            markersize=6,
            label="unstable across folds (<4/6)",
        ),
    ]
    # Legend below both panels so it never sits on a data row.
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Cross-sectional rank correlation with future returns (stable folds annotated)",
        color=INK,
        fontsize=9,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, "F3_signal_atlas")


# --- S1: carry decomposition (supplementary) ---------------------------------
def fig_s1() -> None:
    rows = _read_csv(RESULTS / "real_data" / "carry_decomposition.csv")
    a = [r for r in rows if r["panel"] == "panel_a"]
    b = [r for r in rows if r["panel"] == "panel_b"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    # Panel A: H104 raw carry per-fold funding leg (positive every fold)
    xs = [int(float(r["fold"])) for r in a]
    ys = [_fv(r["value"]) for r in a]
    axa.bar(xs, ys, color=GREEN, width=0.68, edgecolor=SURFACE, linewidth=2.0, zorder=3)
    axa.axhline(0.0, color=BASELINE, linewidth=1.0)
    axa.set_xticks(xs)
    axa.set_xticklabels([f"f{x}" for x in xs])
    axa.set_xlabel("chronological fold")
    axa.set_ylabel("funding-leg PnL")
    axa.set_title("raw carry: funding leg positive 6/6", color=INK, loc="left", fontsize=8)
    axa.text(
        0.60,
        0.80,
        "unhedged price-noise leg\nkeeps the net book\nstatistically weak",
        transform=axa.transAxes,
        color=INK2,
        fontsize=7.0,
        va="top",
    )
    _style_ax(axa)

    # Panel B: H107 cash-and-carry three-way decomposition + net marker
    comp_order = ["funding", "basis", "cost"]
    comp_color = {"funding": GREEN, "basis": AQUA, "cost": RED}
    comp_label = {"funding": "funding\nincome", "basis": "basis\ndrift", "cost": "double-leg\ncost"}
    vals = {r["component_kind"]: _fv(r["value"]) for r in b}
    x = list(range(len(comp_order)))
    heights = [vals[c] for c in comp_order]
    axb.bar(
        x,
        heights,
        color=[comp_color[c] for c in comp_order],
        width=0.62,
        edgecolor=SURFACE,
        linewidth=2.0,
        zorder=3,
    )
    for xi, c in zip(x, comp_order, strict=True):
        axb.annotate(
            f"{vals[c]:+.3f}",
            (xi, vals[c]),
            textcoords="offset points",
            xytext=(0, 6 if vals[c] >= 0 else -12),
            ha="center",
            color=INK,
            fontsize=7,
        )
    net = vals["net"]
    net_x = len(comp_order)
    axb.plot([net_x], [net], marker="D", color=INK, markersize=8, zorder=5)
    axb.annotate(
        f"net {net:+.3f}",
        (net_x, net),
        textcoords="offset points",
        xytext=(0, -14),
        ha="center",
        color=INK,
        fontsize=7,
    )
    axb.axhline(0.0, color=BASELINE, linewidth=1.0)
    lo, hi = min(heights + [net]), max(heights + [net])
    axb.set_ylim(lo - 0.18 * (hi - lo), hi + 0.14 * (hi - lo))  # room for value labels
    axb.set_xticks(x + [net_x])
    axb.set_xticklabels([comp_label[c] for c in comp_order] + ["net"])
    axb.set_ylabel("PnL over train+valid window")
    axb.set_title(
        "cash-and-carry: cost drag exceeds funding income (break-even 3.2 bp per leg)",
        color=INK,
        loc="left",
        fontsize=8,
    )
    _style_ax(axb)

    fig.tight_layout()
    _save(fig, "S1_carry_decomposition")


# --- F4: cash-and-carry fee sensitivity (paper Figure 4) ----------------------
# Value-label placement per fee (offset in points, ha, va): upper-right where the
# falling line leaves room; lower-left at 3 bps (clear of the break-even line) and
# at 10 bps (right edge).
F4_VALUE_PLACE: dict[float, tuple[tuple[float, float], str, str]] = {
    0.0: ((4, 2), "left", "bottom"),
    1.0: ((4, 2), "left", "bottom"),
    2.0: ((4, 2), "left", "bottom"),
    3.0: ((-4, -2), "right", "top"),
    5.0: ((4, 2), "left", "bottom"),
    10.0: ((-4, -2), "right", "top"),
}


def fig_f4() -> None:
    rows = sorted(
        _read_csv(RESULTS / "real_data" / "cash_and_carry_fee_sensitivity.csv"),
        key=lambda r: _fv(r["fee_bps"]),
    )
    fees = [_fv(r["fee_bps"]) for r in rows]
    nets = [_fv(r["full_window_sum_total"]) for r in rows]
    folds = [int(_fv(r["n_folds_mean_ci_lb_gt_zero"])) for r in rows]
    frozen_bps = 5.0
    funding = _fv(
        next(r for r in rows if _fv(r["fee_bps"]) == frozen_bps)["full_window_sum_funding_term"]
    )
    # Break-even: linear interpolation between the bracketing fee levels.
    breakeven = next(
        f0 - n0 * (f1 - f0) / (n1 - n0)
        for f0, n0, f1, n1 in zip(fees, nets, fees[1:], nets[1:], strict=False)
        if n0 > 0 >= n1
    )
    label_box = {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.9}

    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvspan(frozen_bps - 0.15, frozen_bps + 0.15, color=GRID, alpha=0.9, zorder=0)
    ax.axvline(breakeven, color=INK2, linewidth=0.9, linestyle="--", zorder=2)
    ax.axhline(funding, color=INK2, linewidth=0.9, linestyle=":", zorder=2)
    ax.annotate(
        f"funding received {funding:+.3f}",
        (10.6, funding),
        xytext=(-1, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK2,
        fontsize=6.8,
    )
    ax.plot(fees, nets, color=MUTED, linewidth=0.9, zorder=3)
    ax.plot(fees, nets, "o", color=BLUE, markersize=3.8, zorder=4)
    for fee, net in zip(fees, nets, strict=True):
        offset, ha, va = F4_VALUE_PLACE.get(fee, ((4, 2), "left", "bottom"))
        ax.annotate(
            f"{net:+.3f}".replace("-", "−"),
            (fee, net),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=INK,
            fontsize=7,
            bbox=label_box,
            zorder=6,
        )
    # Folds (of 6) whose mean-return CI lower bound is > 0, in a row under the points.
    row_y = -1.24
    for fee, k in zip(fees, folds, strict=True):
        ax.text(
            fee,
            row_y,
            f"{k}/6",
            ha="center",
            va="center",
            color=INK2,
            fontsize=6.5,
            bbox=label_box,
            zorder=6,
        )
    ax.text(
        breakeven + 0.12,
        -0.62,
        f"break-even {breakeven:.1f}",
        rotation=90,
        ha="left",
        va="center",
        color=INK2,
        fontsize=6.5,
        transform=ax.transData,
    )
    ax.text(
        frozen_bps + 0.22,
        -0.75,
        "frozen cost",
        rotation=90,
        ha="left",
        va="center",
        color=INK2,
        fontsize=6.5,
    )
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.36, 0.68)
    ax.set_xticks(fees)
    ax.set_xticklabels([f"{f:g}" for f in fees])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_xlabel("fee per leg (bps)")
    ax.set_ylabel("net return, full window")
    _style_ax(ax)
    fig.tight_layout(pad=0.3)
    _save(fig, "F4_fee_sensitivity")


# --- S2: minimum length of Eq. 1 (supplementary) ----------------------------
def fig_s2() -> None:
    rows = _read_csv(RESULTS / "real_data" / "power_gate.csv")
    horizons = [1, 24, 168]
    hcolor = {1: BLUE, 24: AQUA, 168: YELLOW}
    hlabel = {1: "h=1 bar", 24: "h=24 (overlap)", 168: "h=168 (overlap)"}
    N_REF = 46591

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for h in horizons:
        pts = sorted(
            (
                (int(float(r["n_required"])), _fv(r["s_min_annual"]))
                for r in rows
                if int(float(r["horizon_h"])) == h
            ),
            key=lambda t: t[0],
        )
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color=hcolor[h], label=hlabel[h])
    ax.set_xscale("log")
    ax.axvline(N_REF, color=RED, linewidth=1.4, linestyle="--", zorder=2)
    ax.annotate(
        "panel\nn=46.6k",
        (N_REF, ax.get_ylim()[1]),
        textcoords="offset points",
        xytext=(-4, -20),
        ha="right",
        color=INK,
        fontsize=7,
    )
    ax.set_xlabel("required observations (1h bars, log scale)")
    ax.set_ylabel("target annualized Sharpe ($S_{\\min}$)")
    ax.set_title("Minimum length (Eq. 1) vs target Sharpe", color=INK, loc="left")
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "S2_minimum_length")


FIGURES = {
    "f1": fig_f1,
    "f2": fig_f2,
    "f3": fig_f3,
    "f4": fig_f4,
    "s1": fig_s1,
    "s2": fig_s2,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="render only this figure id (repeatable): f1 f2 f3 f4 s1 s2",
    )
    args = ap.parse_args()
    setup_style()
    keys = args.only if args.only else list(FIGURES)
    for k in keys:
        k = k.lower()
        if k not in FIGURES:
            raise SystemExit(f"unknown figure '{k}'; choices: {sorted(FIGURES)}")
        print(f"[render] {k}")
        FIGURES[k]()


if __name__ == "__main__":
    main()
