"""Controlled learning test (paper Section 3, Figure 2; research id E240).

Question: given a market where the alpha is *known* and a reference rule (Gârleanu–Pedersen)
that harvests it, do standard offline learners recover that alpha, and does the evaluator
catch it when they do not?

Design
------
* Market: 5 independent assets, IC = 0.05, AR(1) persistence 0.9, per-period vol 1%.
* Training data (per training seed 0..9): a 2,000-period path is simulated and its first
  1,000 periods are kept. The behaviour policy is the GP rule plus N(0, 0.05) noise
  (RNG seed ``2_000_003 + seed``), clipped to +/-2. Rewards are the per-step net PnL
  ``w_t . r_{t+1} - (lam/2) ||w_t - w_{t-1}||^2``.
* Learners: BC, IQL, CQL (see :mod:`catch_alpha.learning.learners`) on a flattened 16-period
  window of the 3 features per asset; optionally the previous position is appended
  ("position-aware").
* Evaluation: every fitted policy and the GP reference trade the *same* 100 independent
  1,000-period paths (seeds 1,000,000..1,000,099), all starting flat.
* Nine arms: {BC, IQL, CQL} x {lam = 0 (frictionless), lam = 1 (with costs)}, BC with the
  previous position in the observation (lam = 1), and two diagnostics that reproduce a
  legacy bug where the Bellman discount was set to the GP risk aversion (10).

Per worker (arm, training seed) the 100 paths are summarised by the mean PnL, the ratio of
means to GP (reported only when the GP denominator is identifiable, via a Fieller set),
and the median turnover ratio. Across the 10 training seeds we report the median, IQM,
the number of losing seeds with a Clopper–Pearson interval, and apply the pre-registered
decision rules.

Usage::

    python -m catch_alpha.learning.experiment --smoke
    python -m catch_alpha.learning.experiment --worker bc_lam0 0
    python -m catch_alpha.learning.experiment --run --parallel 2          # into results/learning/rerun/workers
    python -m catch_alpha.learning.experiment --collect results/learning/rerun/workers \
        --report results/learning/rerun/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import beta
from scipy.stats import t as student_t

from catch_alpha.learning.gp import evaluate_weight_path, gp_weights
from catch_alpha.learning.market import MarketPath, simulate_market

if TYPE_CHECKING:  # torch is only needed to train, not to collect results
    from catch_alpha.learning.learners import Learner, OfflineDataset

PROTOCOL_VERSION = "e240-common-flat-v1"

# Market and reference.
IC = 0.05
N_ASSETS = 5
PHI = 0.9
VOL = 0.01
RISK_AVERSION = 10.0
WEIGHT_CAP = 2.0
LAMBDAS = (0.0, 1.0)

# Learners.
LOOKBACK = 16
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 8
BEHAVIOR_NOISE = 0.05
BEHAVIOR_SEED_OFFSET = 2_000_003

# Data.
TRAIN_SOURCE_PERIODS = 2000
TRAIN_PERIODS = 1000
EVAL_PERIODS = 1000
TRAIN_SEEDS = tuple(range(10))
EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))

# Scoring.
MINIMUM_EFFECT = 0.01
CONFIDENCE = 0.95
COMPETENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str  # "bc_repo" | "iql_repo" | "cql_repo" (names kept from the original schema)
    lam_quad: float
    position_aware: bool
    discount_factor: float
    role: str

    @property
    def algorithm(self) -> str:
        return self.policy.removesuffix("_repo")


ARMS = (
    Arm("bc_lam0", "bc_repo", 0.0, False, 0.99, "primary"),
    Arm("iql_lam0", "iql_repo", 0.0, False, 0.99, "primary"),
    Arm("cql_lam0", "cql_repo", 0.0, False, 0.99, "primary"),
    Arm("bc_lam1", "bc_repo", 1.0, False, 0.99, "primary"),
    Arm("iql_lam1", "iql_repo", 1.0, False, 0.99, "primary"),
    Arm("cql_lam1", "cql_repo", 1.0, False, 0.99, "primary"),
    Arm("bc_pos_lam1", "bc_repo", 1.0, True, 0.99, "primary_rescue"),
    # Legacy-bug diagnostics: discount = 10 (the GP risk aversion, mistakenly reused).
    Arm("iql_lam1_legacy_discount10", "iql_repo", 1.0, False, 10.0, "diagnostic"),
    Arm("cql_lam1_legacy_discount10", "cql_repo", 1.0, False, 10.0, "diagnostic"),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


# ---------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------


def training_path(seed: int, *, smoke: bool = False) -> MarketPath:
    """First half of a longer simulated path (the RNG draws depend on the full length)."""
    source_periods = 240 if smoke else TRAIN_SOURCE_PERIODS
    train_periods = source_periods // 2 if smoke else TRAIN_PERIODS
    source = simulate_market(
        seed, n_periods=source_periods, n_assets=N_ASSETS, ic=IC, phi=PHI, vol=VOL
    )
    return source.head(train_periods)


def evaluation_path(seed: int, *, smoke: bool = False) -> MarketPath:
    n_periods = 100 if smoke else EVAL_PERIODS
    return simulate_market(seed, n_periods=n_periods, n_assets=N_ASSETS, ic=IC, phi=PHI, vol=VOL)


def reference_weights(path: MarketPath, lam_quad: float) -> np.ndarray:
    return gp_weights(path, lam_quad=lam_quad, gamma=RISK_AVERSION, weight_cap=WEIGHT_CAP)


def behaviour_weights(path: MarketPath, lam_quad: float, seed: int) -> np.ndarray:
    """Noisy GP (distinct from the clean GP used as the evaluation reference)."""
    clean = reference_weights(path, lam_quad)
    rng = np.random.default_rng(BEHAVIOR_SEED_OFFSET + int(seed))
    noisy = clean + rng.normal(0.0, BEHAVIOR_NOISE, size=clean.shape)
    return np.clip(noisy, -WEIGHT_CAP, WEIGHT_CAP)


def causal_observations(features: np.ndarray, lookback: int) -> np.ndarray:
    """Row ``t`` = ``features[t-lookback+1 : t+1]`` flattened oldest-first, zero-padded left."""
    t_len, n_assets, n_feat = features.shape
    flat_dim = lookback * n_assets * n_feat
    out = np.zeros((t_len, flat_dim), dtype=np.float64)
    for t in range(t_len):
        window = features[max(0, t - lookback + 1) : t + 1].reshape(-1)
        out[t, flat_dim - window.size :] = window
    return out


def offline_dataset(path: MarketPath, arm: Arm, seed: int, lookback: int) -> OfflineDataset:
    """Transitions ``(o_t, a_t, r_t, o_{t+1}, done)`` collected by the behaviour policy."""
    from catch_alpha.learning.learners import OfflineDataset

    behaviour = behaviour_weights(path, arm.lam_quad, seed)
    obs = causal_observations(path.features, lookback)
    if arm.position_aware:
        # The behaviour policy's own previous weight (flat at t = 0).
        previous_position = np.zeros_like(behaviour)
        previous_position[1:] = behaviour[:-1]
        obs = np.concatenate([obs, previous_position], axis=1)

    n = path.n_periods - 1
    actions = behaviour[:n]
    gross = np.sum(actions * path.returns[1 : n + 1], axis=1)
    previous = np.zeros_like(actions)
    previous[1:] = actions[:-1]
    delta = actions - previous
    cost = 0.5 * float(arm.lam_quad) * np.sum(delta * delta, axis=1)
    done = np.zeros((n,), dtype=np.float32)
    done[-1] = 1.0
    return OfflineDataset(
        obs=obs[:n].astype(np.float32),
        action=actions.astype(np.float32),
        reward=(gross - cost).astype(np.float32),
        next_obs=obs[1 : n + 1].astype(np.float32),
        done=done,
    )


# ---------------------------------------------------------------------------
# Fit, roll out, score.
# ---------------------------------------------------------------------------


@dataclass
class FittedPolicy:
    learner: Learner
    lookback: int
    position_aware: bool


def fit_policy(arm: Arm, seed: int, *, smoke: bool = False) -> FittedPolicy:
    import torch

    from catch_alpha.learning.learners import make_learner, set_global_seed

    lookback = 4 if smoke else LOOKBACK
    torch.set_num_threads(2)
    data = offline_dataset(training_path(seed, smoke=smoke), arm, seed, lookback)
    set_global_seed(int(seed))
    learner = make_learner(
        arm.algorithm,
        obs_dim=int(data.obs.shape[1]),
        action_dim=N_ASSETS,
        hidden=16 if smoke else HIDDEN_SIZE,
        action_clip=WEIGHT_CAP,
        gamma=arm.discount_factor,
    )
    learner.fit(data, epochs=1 if smoke else EPOCHS, batch_size=BATCH_SIZE)
    return FittedPolicy(learner=learner, lookback=lookback, position_aware=arm.position_aware)


def rollout(fitted: FittedPolicy, path: MarketPath, initial_weights: np.ndarray) -> np.ndarray:
    """Closed-loop rollout: features are exogenous; only the position is fed back."""
    previous = np.clip(np.asarray(initial_weights, dtype=np.float64), -WEIGHT_CAP, WEIGHT_CAP)
    base_obs = causal_observations(path.features, fitted.lookback)
    weights = np.zeros((path.n_periods, path.n_assets), dtype=np.float64)
    for t, feature_obs in enumerate(base_obs):
        obs = np.concatenate([feature_obs, previous]) if fitted.position_aware else feature_obs
        action = np.asarray(fitted.learner.act(obs.astype(np.float32)), dtype=np.float64)
        weights[t] = action.reshape(-1)[: path.n_assets]
        previous = weights[t].copy()
    return np.clip(weights, -WEIGHT_CAP, WEIGHT_CAP)


def score_one_path(fitted: FittedPolicy, path: MarketPath, lam_quad: float) -> dict[str, Any]:
    """Candidate and GP on the same path from the same flat start."""
    initial = np.zeros((N_ASSETS,), dtype=np.float64)
    candidate = evaluate_weight_path(
        rollout(fitted, path, initial), path.returns, lam_quad, initial_weights=initial
    )
    reference = evaluate_weight_path(
        reference_weights(path, lam_quad), path.returns, lam_quad, initial_weights=initial
    )
    return {
        "eval_seed": int(path.seed),
        "candidate": asdict(candidate),
        "reference": asdict(reference),
        "flat_anchor_pnl": 0.0,
        "turnover_ratio": (
            candidate.turnover_l1 / reference.turnover_l1 if reference.turnover_l1 > 0.0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Is the ratio to GP identifiable? (denominator gate with a Fieller set)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiellerInterval:
    low: float | None
    high: float | None
    bounded: bool
    reason: str


def fieller_interval(
    numerator: np.ndarray, denominator: np.ndarray, *, confidence: float = CONFIDENCE
) -> FiellerInterval:
    """Fieller confidence set for ``mean(numerator) / mean(denominator)`` on paired samples.

    The set ``{r : (x_bar - r y_bar)^2 <= t^2 Var(x_bar - r y_bar)}`` is a bounded interval
    only when the denominator mean is significantly non-zero; otherwise it is reported as
    unbounded rather than as a misleading finite interval.
    """
    x = np.asarray(numerator, dtype=np.float64).reshape(-1)
    y = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return FiellerInterval(None, None, False, "at least two paired observations required")
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    cov = np.cov(np.stack([x, y]), ddof=1) / float(x.size)
    var_x, var_y, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    crit = float(student_t.ppf(0.5 + confidence / 2.0, df=int(x.size) - 1))
    crit_sq = crit * crit
    # Solve a r^2 + b r + c <= 0.
    a = y_mean * y_mean - crit_sq * var_y
    b = -2.0 * x_mean * y_mean + 2.0 * crit_sq * cov_xy
    c = x_mean * x_mean - crit_sq * var_x
    disc = b * b - 4.0 * a * c
    if -1e-12 * max(1.0, abs(b * b), abs(4.0 * a * c)) < disc < 0.0:
        disc = 0.0
    if a <= 0.0:
        return FiellerInterval(None, None, False, "denominator confidence set includes zero")
    if disc < 0.0:
        return FiellerInterval(None, None, False, "Fieller quadratic has no bounded real set")
    root = math.sqrt(disc)
    low, high = (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)
    return FiellerInterval(float(min(low, high)), float(max(low, high)), True, "bounded")


def assess_denominator(
    reference_scores: np.ndarray, candidate_scores: np.ndarray
) -> dict[str, Any]:
    """Gate the ratio-to-GP: is the GP margin over the flat anchor real and large enough?

    The lower anchor and the inaction policy are both "stay flat" (score 0 on every path),
    so the denominator is the GP score itself. Field names follow the original schema.
    """
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    lower = np.zeros_like(reference)
    denominator = reference - lower
    mean = float(np.mean(denominator))
    sd = float(np.std(denominator, ddof=1)) if denominator.size > 1 else 0.0
    if sd == 0.0:
        low = mean
    else:
        crit = float(student_t.ppf(CONFIDENCE, df=int(denominator.size) - 1))
        low = mean - crit * (sd / math.sqrt(denominator.size))
    sign_correct, separated, practical = mean > 0.0, low > 0.0, mean >= MINIMUM_EFFECT
    interval = fieller_interval(candidate - lower, denominator)
    identifiable = bool(sign_correct and separated and practical and interval.bounded)
    competence_score = float(np.mean(lower)) + COMPETENCE_THRESHOLD * mean
    inaction_score = 0.0
    above_inaction = competence_score > inaction_score
    if not sign_correct or not separated:
        status = "RECOVERY_NOT_IDENTIFIABLE_DENOMINATOR_TOUCHES_ZERO"
    elif not practical:
        status = "RECOVERY_NOT_IDENTIFIABLE_DENOMINATOR_TOO_SMALL"
    elif not interval.bounded:
        status = "RECOVERY_NOT_IDENTIFIABLE_FIELLER_UNBOUNDED"
    elif not above_inaction:
        status = "RECOVERY_IDENTIFIABLE_DECISION_THRESHOLD_INVALID"
    else:
        status = "RECOVERY_AND_THRESHOLD_ADMISSIBLE"
    return {
        "denominator_mean": mean,
        "denominator_one_sided_low": float(low),
        "minimum_effect": MINIMUM_EFFECT,
        "sign_correct": sign_correct,
        "separated_from_zero": separated,
        "practically_large": practical,
        "reference_valid": True,
        "ratio_interval": asdict(interval),
        "recovery_identifiable": identifiable,
        "competence_threshold": COMPETENCE_THRESHOLD,
        "competence_score": competence_score,
        "inaction_score": inaction_score,
        "threshold_above_inaction": above_inaction,
        "competence_threshold_admissible": bool(identifiable and above_inaction),
        "status": status,
    }


def summarize_worker(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [float(row["candidate"]["pnl"]) for row in rows]
    reference = [float(row["reference"]["pnl"]) for row in rows]
    turnovers = [float(row["turnover_ratio"]) for row in rows if row["turnover_ratio"] is not None]
    gate = assess_denominator(np.asarray(reference), np.asarray(candidate))
    candidate_mean, reference_mean = float(np.mean(candidate)), float(np.mean(reference))
    return {
        "candidate_pnl_mean": candidate_mean,
        "reference_pnl_mean": reference_mean,
        "recovery_ratio_of_means": (
            candidate_mean / reference_mean if gate["recovery_identifiable"] else None
        ),
        "n_eval_paths_negative": int(np.sum(np.asarray(candidate) < 0.0)),
        "median_turnover_ratio": float(np.median(turnovers)) if turnovers else None,
        "denominator_gate": gate,
    }


# ---------------------------------------------------------------------------
# Worker.
# ---------------------------------------------------------------------------


def run_worker(arm_name: str, seed: int, *, smoke: bool = False) -> dict[str, Any]:
    """Train one (arm, training seed) and score it on the evaluation panel."""
    arm = ARM_BY_NAME[arm_name]
    if not smoke and seed not in TRAIN_SEEDS:
        raise ValueError(f"unregistered training seed {seed}")
    fitted = fit_policy(arm, seed, smoke=smoke)
    eval_seeds = EVAL_SEEDS[:3] if smoke else EVAL_SEEDS
    rows = [
        score_one_path(fitted, evaluation_path(s, smoke=smoke), arm.lam_quad) for s in eval_seeds
    ]
    source_periods = 240 if smoke else TRAIN_SOURCE_PERIODS
    return {
        "protocol_version": PROTOCOL_VERSION,
        "smoke": bool(smoke),
        "arm": asdict(arm),
        "training_seed": int(seed),
        "training_source_periods": source_periods,
        "training_periods": source_periods // 2 if smoke else TRAIN_PERIODS,
        "evaluation_seeds": list(eval_seeds),
        "evaluation_periods": 100 if smoke else EVAL_PERIODS,
        "initial_weights": [0.0] * N_ASSETS,
        "fitted_contract": {
            "risk_aversion": RISK_AVERSION,
            "discount_factor": arm.discount_factor,
            "lookback": fitted.lookback,
            "position_aware": arm.position_aware,
            "coordinate": "absolute",
        },
        "summary": summarize_worker(rows),
        "evaluation_rows": rows,
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation and decisions.
# ---------------------------------------------------------------------------


def _iqm(values: list[float]) -> float:
    """Interquartile mean of 10 values: mean of order statistics 3..8."""
    return float(statistics.fmean(sorted(values)[2:8]))


def _clopper_pearson(k: int, n: int) -> tuple[float, float]:
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return low, high


def _arm_aggregate(arm: Arm, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(p["summary"]["candidate_pnl_mean"]) for p in payloads]
    raw_recoveries = [p["summary"]["recovery_ratio_of_means"] for p in payloads]
    turnover = [float(p["summary"]["median_turnover_ratio"]) for p in payloads]
    recoveries = (
        None if any(r is None for r in raw_recoveries) else [float(r) for r in raw_recoveries]
    )
    ordered = sorted(pnls)
    n_negative = int(np.sum(np.asarray(pnls) < 0.0))
    return {
        "arm": asdict(arm),
        "per_training_seed_mean_pnl": pnls,
        "per_training_seed_recovery": recoveries,
        "median_mean_pnl": float(statistics.median(pnls)),
        "iqm_mean_pnl": _iqm(pnls),
        "median_order_stat_interval": [ordered[1], ordered[8]],
        "n_negative_training_seeds": n_negative,
        "negative_seed_rate_clopper_pearson": list(_clopper_pearson(n_negative, len(pnls))),
        "median_turnover_ratio": float(statistics.median(turnover)),
        "median_recovery": None if recoveries is None else float(statistics.median(recoveries)),
        "iqm_recovery": None if recoveries is None else _iqm(recoveries),
    }


def _rule_passes(agg: dict[str, Any]) -> bool:
    """Recovered: median ratio >= 0.5, <= 2 losing seeds, median turnover ratio <= 5."""
    return bool(
        agg["median_recovery"] is not None
        and agg["median_recovery"] >= 0.50
        and agg["n_negative_training_seeds"] <= 2
        and agg["median_turnover_ratio"] <= 5.0
    )


def decide(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered decision rules."""
    decisions: dict[str, Any] = {}
    for algorithm in ("bc", "iql", "cql"):
        free, cost = aggregates[f"{algorithm}_lam0"], aggregates[f"{algorithm}_lam1"]
        # The learner must work without costs before a collapse with costs means anything.
        sanity = bool(
            free["median_recovery"] is not None
            and free["median_recovery"] >= 0.50
            and free["n_negative_training_seeds"] <= 2
        )
        collapse = bool(
            sanity
            and cost["n_negative_training_seeds"] >= 8
            and cost["median_mean_pnl"] < 0.0
            and cost["median_turnover_ratio"] > 5.0
        )
        recovered = _rule_passes(cost)
        decisions[algorithm] = {
            "frictionless_sanity": sanity,
            "collapse_confirmed": collapse,
            "cost_aware_capability_recovered": recovered,
            "verdict": "collapse_confirmed"
            if collapse
            else ("recovered" if recovered else "mixed"),
        }
    decisions["bc_position_aware"] = {"rescue": _rule_passes(aggregates["bc_pos_lam1"])}
    for algorithm in ("iql", "cql"):
        corrected = aggregates[f"{algorithm}_lam1"]
        legacy = aggregates[f"{algorithm}_lam1_legacy_discount10"]
        decisions[f"{algorithm}_discount_overload"] = {
            "corrected_minus_legacy_median_pnl": (
                corrected["median_mean_pnl"] - legacy["median_mean_pnl"]
            ),
            "corrected_minus_legacy_median_recovery": (
                None
                if corrected["median_recovery"] is None or legacy["median_recovery"] is None
                else corrected["median_recovery"] - legacy["median_recovery"]
            ),
        }
    return decisions


def validate_worker(payload: dict[str, Any]) -> None:
    """Fail closed on stale, smoke, renamed, truncated or contract-inconsistent workers."""
    name = str(payload.get("arm", {}).get("name"))
    if name not in ARM_BY_NAME:
        raise ValueError(f"worker for unknown arm {name!r}")
    arm = ARM_BY_NAME[name]
    seed = payload.get("training_seed")
    where = f"worker {name}/{seed}"
    expected_scalars = {
        "protocol_version": PROTOCOL_VERSION,
        "smoke": False,
        "training_source_periods": TRAIN_SOURCE_PERIODS,
        "training_periods": TRAIN_PERIODS,
        "evaluation_periods": EVAL_PERIODS,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"{where} has {key}={payload.get(key)!r}; expected {expected!r}")
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"{where} uses an unregistered training seed")
    if payload.get("arm") != asdict(arm):
        raise ValueError(f"{where} contains a different arm contract")
    if payload.get("evaluation_seeds") != list(EVAL_SEEDS):
        raise ValueError(f"{where} contains a different evaluation panel")
    if payload.get("initial_weights") != [0.0] * N_ASSETS:
        raise ValueError(f"{where} does not start from the common flat position")
    expected_fitted = {
        "risk_aversion": RISK_AVERSION,
        "discount_factor": arm.discount_factor,
        "lookback": LOOKBACK,
        "position_aware": arm.position_aware,
        "coordinate": "absolute",
    }
    if payload.get("fitted_contract") != expected_fitted:
        raise ValueError(f"{where} has a mismatched fitted-policy contract")

    rows = payload.get("evaluation_rows")
    if not isinstance(rows, list) or len(rows) != len(EVAL_SEEDS):
        raise ValueError(f"{where} has an incomplete evaluation panel")
    for eval_seed, row in zip(EVAL_SEEDS, rows, strict=True):
        if row.get("eval_seed") != eval_seed or row.get("flat_anchor_pnl") != 0.0:
            raise ValueError(f"{where} violates the evaluation-seed order or the flat anchor")
        for policy in ("candidate", "reference"):
            score = row.get(policy)
            if not isinstance(score, dict):
                raise ValueError(f"{where} is missing {policy} accounting")
            numeric = np.asarray(
                [
                    score.get(k)
                    for k in (
                        "pnl",
                        "gross_pnl",
                        "quadratic_cost",
                        "turnover_l1",
                        "turnover_l2_squared",
                    )
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"{where} contains non-finite accounting")
            if score.get("n_scored_actions") != EVAL_PERIODS - 1:
                raise ValueError(f"{where} has the wrong scored horizon")
            if not np.isclose(numeric[0], numeric[1] - numeric[2], rtol=0.0, atol=1e-12):
                raise ValueError(f"{where} fails the decomposition pnl = gross - cost")


def summaries_match(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Recursive equality of two worker summaries, with a float tolerance.

    Recomputing a summary from its rows can move the last digits of a float with the
    numpy/scipy version (e.g. 0.8929274135169032 vs ...134); decisions and printed
    numbers never depend on those digits. Everything else must match exactly.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            summaries_match(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a
        )
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(
            summaries_match(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
            for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def collect(worker_payloads: Iterable[dict[str, Any]], *, check_summaries: bool = True) -> dict:
    """Aggregate the 90 worker payloads into the report (aggregates + decisions).

    Every payload must pass :func:`validate_worker` (registered contract, full panel,
    consistent accounting). With ``check_summaries`` every worker summary is recomputed
    from its rows and must match (floats to a relative 1e-9, see :func:`summaries_match`).
    The GP reference must be bit-identical across arms that share a cost level.
    """
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for payload in worker_payloads:
        validate_worker(payload)
        key = (str(payload["arm"]["name"]), int(payload["training_seed"]))
        if key in by_key:
            raise ValueError(f"duplicate worker {key}")
        if check_summaries and not summaries_match(
            payload["summary"], summarize_worker(payload["evaluation_rows"])
        ):
            raise ValueError(f"worker {key} summary does not reproduce from its rows")
        by_key[key] = payload
    missing = [(a.name, s) for a in ARMS for s in TRAIN_SEEDS if (a.name, s) not in by_key]
    if missing:
        raise ValueError(f"cannot collect; missing {len(missing)} workers, e.g. {missing[:3]}")

    for lam in LAMBDAS:
        panels = [
            np.asarray([row["reference"]["pnl"] for row in by_key[(a.name, s)]["evaluation_rows"]])
            for a in ARMS
            if a.lam_quad == lam
            for s in TRAIN_SEEDS
        ]
        if not all(np.array_equal(panels[0], panel) for panel in panels[1:]):
            raise ValueError(f"reference panel differs across lambda={lam} arms")

    aggregates = {
        arm.name: _arm_aggregate(arm, [by_key[(arm.name, s)] for s in TRAIN_SEEDS]) for arm in ARMS
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "contract": {
            "training_seeds": list(TRAIN_SEEDS),
            "evaluation_seeds": list(EVAL_SEEDS),
            "training_source_periods": TRAIN_SOURCE_PERIODS,
            "training_periods": TRAIN_PERIODS,
            "evaluation_periods": EVAL_PERIODS,
            "initial_weights": [0.0] * N_ASSETS,
            "risk_aversion": RISK_AVERSION,
            "primary_discount_factor": 0.99,
        },
        "aggregates": aggregates,
        "decisions": decide(aggregates),
    }


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------


def _worker_file(out_dir: Path, arm_name: str, seed: int) -> Path:
    return out_dir / f"E240_{arm_name}_s{seed}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def run_all(out_dir: Path, *, parallel: int = 2, seeds: Iterable[int] = TRAIN_SEEDS) -> int:
    """Run every missing worker in subprocesses (each uses 2 torch threads)."""
    jobs = [
        (arm.name, s)
        for arm in ARMS
        for s in seeds
        if not _worker_file(out_dir, arm.name, s).exists()
    ]
    print(f"{len(jobs)} workers to run, {parallel} in parallel", flush=True)

    def run_job(job: tuple[str, int]) -> int:
        cmd = [sys.executable, "-m", "catch_alpha.learning.experiment", "--worker", job[0]]
        cmd += [str(job[1]), "--out", str(out_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        lines = (result.stdout or result.stderr or "").strip().splitlines()
        print(lines[-1] if lines else f"{job}: rc={result.returncode}", flush=True)
        return result.returncode

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        codes = list(pool.map(run_job, jobs))
    return 0 if all(code == 0 for code in codes) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled learning test (E240).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny run of bc_lam1, seed 0")
    mode.add_argument("--worker", nargs=2, metavar=("ARM", "SEED"))
    mode.add_argument("--run", action="store_true", help="run all missing workers")
    mode.add_argument("--collect", metavar="WORKER_DIR", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/learning/rerun/workers"),
        help="where worker files go (never the published results/learning/workers)",
    )
    parser.add_argument("--report", type=Path, default=Path("results/learning/rerun/report.json"))
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args(argv)

    if args.smoke:
        summary = run_worker("bc_lam1", 0, smoke=True)["summary"]
        print(json.dumps(summary, indent=1))
        return 0
    if args.worker:
        arm_name, seed = args.worker[0], int(args.worker[1])
        target = _worker_file(args.out, arm_name, seed)
        if target.exists():  # never replace a finished (possibly published) run
            print(f"[{arm_name} seed={seed}] exists, skipped: {target}", flush=True)
            return 0
        payload = run_worker(arm_name, seed)
        _write_json(target, payload)
        s = payload["summary"]
        print(
            f"[{arm_name} seed={seed}] pnl={s['candidate_pnl_mean']:+.6f} "
            f"R={s['recovery_ratio_of_means']} turnover={s['median_turnover_ratio']}",
            flush=True,
        )
        return 0
    if args.run:
        return run_all(args.out, parallel=args.parallel)
    payloads = [json.loads(p.read_text()) for p in sorted(args.collect.glob("E240_*_s*.json"))]
    report = collect(payloads)
    _write_json(args.report, report)
    print(json.dumps(report["decisions"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
