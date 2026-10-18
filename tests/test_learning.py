"""Fast checks for the controlled learning test (market, GP reference, accounting, smoke)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catch_alpha.learning import experiment
from catch_alpha.learning.gp import (
    evaluate_weight_path,
    gp_aim_multiplier,
    gp_trading_rate,
    gp_weights,
)
from catch_alpha.learning.market import realized_ic, residual_std, signal_loading, simulate_market

GAMMA, SIGMA = 10.0, residual_std(0.05, 0.01)


# --- Market -----------------------------------------------------------------


def test_market_calibration_and_timing() -> None:
    path = simulate_market(123, n_periods=20_000, n_assets=5)
    assert np.all(path.returns[0] == 0.0)
    assert np.array_equal(path.features[:, :, 0], path.signal)
    # corr(s_t, r_{t+1}) = ic and std(r) = vol, up to sampling error (n = 1e5 pairs).
    assert realized_ic(path.signal, path.returns) == pytest.approx(0.05, abs=0.012)
    assert np.std(path.returns[1:]) == pytest.approx(0.01, rel=0.02)
    assert np.std(path.signal) == pytest.approx(1.0, rel=0.05)


def test_market_is_deterministic_and_head_is_a_prefix() -> None:
    a = simulate_market(7, n_periods=200)
    b = simulate_market(7, n_periods=200)
    assert np.array_equal(a.features, b.features)
    head = a.head(50)
    assert np.array_equal(head.returns, a.returns[:50])
    assert np.array_equal(head.features, a.features[:50])


# --- GP reference -----------------------------------------------------------


def test_gp_frictionless_limit_is_markowitz() -> None:
    assert gp_trading_rate(gamma=GAMMA, sigma_eps=SIGMA, lam=0.0) == 1.0
    assert gp_aim_multiplier(gamma=GAMMA, sigma_eps=SIGMA, lam=0.0, phi=0.9) == 1.0
    path = simulate_market(3, n_periods=50)
    g = GAMMA * SIGMA * SIGMA
    expected = np.clip(signal_loading(0.05, 0.01) * path.signal / g, -2.0, 2.0)
    np.testing.assert_allclose(gp_weights(path, lam_quad=0.0), expected, rtol=1e-6)


def test_gp_closed_form_identities() -> None:
    g = GAMMA * SIGMA * SIGMA
    for lam in (0.01, 1.0, 100.0):
        rate = gp_trading_rate(gamma=GAMMA, sigma_eps=SIGMA, lam=lam)
        c_xx = rate * lam  # A = c_xx / lam
        assert c_xx * c_xx + g * c_xx - lam * g == pytest.approx(0.0, abs=1e-12)
        assert gp_aim_multiplier(gamma=GAMMA, sigma_eps=SIGMA, lam=lam, phi=0.0) == pytest.approx(
            rate
        )
        assert gp_aim_multiplier(gamma=GAMMA, sigma_eps=SIGMA, lam=lam, phi=1.0) == pytest.approx(
            1.0
        )
    assert gp_trading_rate(gamma=GAMMA, sigma_eps=SIGMA, lam=1e9) < 1e-3
    mults = [gp_aim_multiplier(gamma=GAMMA, sigma_eps=SIGMA, lam=1.0, phi=p) for p in (0, 0.5, 0.9)]
    assert mults == sorted(mults)


def test_gp_partial_adjustment_recursion() -> None:
    path = simulate_market(5, n_periods=30)
    w = gp_weights(path, lam_quad=1.0)
    rate = gp_trading_rate(gamma=GAMMA, sigma_eps=SIGMA, lam=1.0)
    mult = gp_aim_multiplier(gamma=GAMMA, sigma_eps=SIGMA, lam=1.0, phi=0.9)
    aim = mult * signal_loading(0.05, 0.01) * path.signal / (GAMMA * SIGMA * SIGMA)
    prev = np.vstack([np.zeros((1, 5)), w[:-1]])
    np.testing.assert_allclose(w, prev + rate * (aim - prev), rtol=1e-5, atol=1e-6)


# --- Accounting -------------------------------------------------------------


def test_pnl_accounting_identity() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(40, 5))
    r = rng.normal(scale=0.01, size=(40, 5))
    start = rng.normal(size=5)
    s = evaluate_weight_path(w, r, 0.7, initial_weights=start)
    assert s.pnl == pytest.approx(s.gross_pnl - s.quadratic_cost, abs=1e-15)
    assert s.gross_pnl == pytest.approx(float(np.sum(w[:-1] * r[1:])))
    deltas = np.diff(np.vstack([start, w[:-1]]), axis=0)
    assert s.turnover_l1 == pytest.approx(float(np.abs(deltas).sum()))
    assert s.quadratic_cost == pytest.approx(0.35 * float((deltas**2).sum()))
    assert s.n_scored_actions == 39


# --- Learners and the worker ------------------------------------------------


def test_minibatches_follow_dataloader_rng() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from catch_alpha.learning.learners import OfflineDataset, shuffled_minibatches

    n = 37
    data = OfflineDataset(
        obs=np.arange(n, dtype=np.float32)[:, None],
        action=np.zeros((n, 1), dtype=np.float32),
        reward=np.zeros(n, dtype=np.float32),
        next_obs=np.zeros((n, 1), dtype=np.float32),
        done=np.zeros(n, dtype=np.float32),
    )
    torch.manual_seed(11)
    ours = [b["obs"][:, 0].long().tolist() for b in shuffled_minibatches(data, 8)]
    after_ours = torch.rand(1)
    torch.manual_seed(11)
    theirs = [b.tolist() for b in DataLoader(list(range(n)), batch_size=8, shuffle=True)]
    after_theirs = torch.rand(1)
    assert ours == theirs
    assert torch.equal(after_ours, after_theirs)


@pytest.mark.parametrize("arm", ["bc_lam1", "iql_lam0", "cql_lam1", "bc_pos_lam1"])
def test_smoke_worker(arm: str) -> None:
    pytest.importorskip("torch")
    from catch_alpha.learning.experiment import run_worker

    payload = run_worker(arm, 0, smoke=True)
    rows = payload["evaluation_rows"]
    assert len(rows) == 3
    for row in rows:
        for who in ("candidate", "reference"):
            score = row[who]
            assert np.isfinite(score["pnl"])
            assert score["pnl"] == pytest.approx(score["gross_pnl"] - score["quadratic_cost"])
            assert score["n_scored_actions"] == 99
    summary = payload["summary"]
    assert set(summary) >= {"candidate_pnl_mean", "recovery_ratio_of_means", "denominator_gate"}
    # Same seed, same result.
    assert run_worker(arm, 0, smoke=True)["summary"] == summary


def _published_worker(name: str = "bc_lam1_s0") -> dict:
    path = Path(__file__).resolve().parents[1] / "results" / "learning" / "workers"
    return json.loads((path / f"E240_{name}.json").read_text(encoding="utf-8"))


def test_published_workers_pass_validation():
    experiment.validate_worker(_published_worker())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(protocol_version="stale-v0"),
        lambda p: p.update(smoke=True),
        lambda p: p["fitted_contract"].update(discount_factor=10.0),
        lambda p: p["evaluation_rows"][0]["candidate"].update(gross_pnl=123.0),
        lambda p: p["evaluation_rows"].pop(),
    ],
    ids=["stale-protocol", "smoke", "wrong-discount", "broken-accounting", "truncated"],
)
def test_validation_rejects_inconsistent_workers(mutate):
    payload = _published_worker()
    mutate(payload)
    payload["summary"] = experiment.summarize_worker(payload["evaluation_rows"])
    with pytest.raises(ValueError):
        experiment.validate_worker(payload)


def test_worker_cli_never_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "E240_bc_lam0_s0.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(experiment, "run_worker", lambda *a, **k: pytest.fail("retrained"))
    assert experiment.main(["--worker", "bc_lam0", "0", "--out", str(tmp_path)]) == 0
    assert target.read_text(encoding="utf-8") == "{}"


def test_collect_reproduces_the_published_report():
    """`learning_test.py collect` on the 90 published runs gives the published decisions."""
    learning = Path(__file__).resolve().parents[1] / "results" / "learning"
    workers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((learning / "workers").glob("E240_*_s*.json"))
    ]
    assert len(workers) == 90
    report = experiment.collect(workers)
    published = json.loads((learning / "report.json").read_text(encoding="utf-8"))
    assert report["decisions"] == published["decisions"]
    assert experiment.summaries_match(report["aggregates"], published["aggregates"])


def test_summaries_match_tolerates_last_digit_noise_only():
    assert experiment.summaries_match({"x": [0.8929274135169032]}, {"x": [0.8929274135169134]})
    assert not experiment.summaries_match({"x": 0.89292}, {"x": 0.89293})
    assert not experiment.summaries_match({"ok": True}, {"ok": False})
