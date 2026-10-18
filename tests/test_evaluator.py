"""Fast checks of the evaluator against numbers quoted in the paper."""

from pathlib import Path

import numpy as np
import pytest

from catch_alpha import calibration, stats
from catch_alpha.gate import evaluate
from catch_alpha.ledger import TrialLedger


def test_minimum_length_matches_paper():
    assert stats.minimum_required_observations(8760) == 54_160  # hourly, S_min = 1
    assert stats.minimum_required_observations(252) == 1_559  # daily calibration design
    assert stats.power_gate(46_590, 8760)["status"] == "insufficient_data"


def test_overlap_inflation_is_one_for_non_overlapping_returns():
    assert stats.overlap_variance_inflation(1, 1000) == 1.0
    assert 23.0 < stats.overlap_variance_inflation(24, 100_000) <= 24.0


def test_dsr_worked_example():
    # Section 2.2: per-period Sharpe 0.06 over 3,000 observations.
    assert stats.deflated_sharpe_ratio(0.06, 3, 3000) == pytest.approx(0.99, abs=0.005)
    assert stats.deflated_sharpe_ratio(0.06, 347, 3000) == pytest.approx(0.63, abs=0.005)


def test_pbo_extremes():
    folds = np.arange(6, dtype=float)
    consistent = np.vstack([folds, folds + 1.0])  # the better strategy wins everywhere
    assert stats.probability_of_backtest_overfitting(consistent) == 0.0
    assert np.isnan(stats.probability_of_backtest_overfitting(folds[None, :]))


def test_wilson_bound_for_zero_of_200():
    assert stats.wilson_interval(0, 200)[1] == pytest.approx(0.0188, abs=1e-4)


def test_short_sample_reports_insufficient_data():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    verdict = evaluate(r, np.zeros(500), {"cash": np.zeros(500)}, periods_per_year=252, n_trials=1)
    assert verdict["status"] == "INSUFFICIENT_DATA" and verdict["checks"] == {}


def test_first_null_and_strong_edge():
    assert calibration.one_strategy((0.0, 0, 40)) is False
    assert calibration.one_strategy((calibration.planted_mu(4.0), 1, 40)) is True


def test_ledger_budget(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.json", test_access_budget=1)
    ledger.record("panel", "valid", "signal_ic", "a", "v1")
    ledger.record("panel", "test", "sharpe", "final", "v1")
    assert ledger.n_trials("panel") == 1  # the final-period run is not a selection attempt
    assert ledger.test_accesses("panel") == 1
    with pytest.raises(RuntimeError):
        ledger.record("panel", "test", "sharpe", "again", "v1")


def test_ledger_concurrent_appends(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "ledger.json"
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_append_many, [path] * 4))
    records = TrialLedger(path).records()
    assert len(records) == 4 * 25


def _append_many(path):
    ledger = TrialLedger(path)
    for i in range(25):
        ledger.record("panel", "valid", "signal_ic", f"e{i}", "v1")


def test_ledger_requires_code_version(tmp_path):
    with pytest.raises(ValueError):
        TrialLedger(tmp_path / "l.json").record("panel", "valid", "x", "e", "  ")


def test_published_ledger_counts():
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "real_data"
        / "trial_ledger_public_v3.json"
    )
    ledger = TrialLedger(path)
    assert ledger.n_trials("public_v3") == 352
    assert ledger.test_accesses("public_v3") == 1


def test_overlap_requirement_is_bit_identical_to_the_original():
    # values from the original implementation (plain-loop summation; see stats.py)
    assert stats.minimum_required_observations(252, 0.16, horizon_bars=4) == 243_436
    assert stats.minimum_required_observations(8760, 0.10, horizon_bars=4) == 21_663_680
