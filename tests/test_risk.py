import numpy as np
import pandas as pd

from arena_limits.evaluate import calibration, detection_delay, false_freeze_rate
from arena_limits.profile.correlation import (
    degenerate_pairs,
    tetrachoric_corr,
    wilson_interval,
)
from arena_limits.risk.es_alloc import portfolio_es
from arena_limits.risk.limits import pair_capital, quote_pair
from arena_limits.risk.vasicek import vasicek_pd


def test_vasicek_decreases_in_systemic_factor():
    stressed = float(vasicek_pd([0.2], 0.2, [-2.0])[0, 0])
    benign = float(vasicek_pd([0.2], 0.2, [2.0])[0, 0])
    assert stressed > benign


def test_vasicek_mean_matches_unconditional_pd():
    z = np.random.default_rng(0).standard_normal(20000)
    assert abs(float(vasicek_pd([0.3], 0.3, z).mean()) - 0.3) < 0.02


def test_portfolio_es_orders_var_and_es():
    out = portfolio_es([0.1, 0.1, 0.1], [1.0, 1.0, 1.0], rho_b=0.3, n_sims=4000, seed=1)
    assert out["es"] >= out["var"] >= 0.0
    assert len(out["contributions"]) == 3


def test_pair_capital_grows_with_probability():
    assert pair_capital(0.5, 0.3) > pair_capital(0.05, 0.3) >= 0.0


def test_quote_pair_capped_by_request():
    assert quote_pair(0.2, 0.3, budget=1e9, requested=1000.0) == 1000.0
    assert quote_pair(0.9, 0.3, budget=1.0, requested=1000.0) <= 1000.0


def test_wilson_interval_brackets_estimate():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_tetrachoric_nan_on_degenerate_marginals():
    y = np.zeros(10)
    assert np.isnan(tetrachoric_corr(y, y))
    assert degenerate_pairs(np.array([[1.0, np.nan], [np.nan, 1.0]])) == 1


def test_calibration_zero_when_perfect():
    df = pd.DataFrame({"agent_id": ["a", "a"], "p_hat": [0.5, 0.5], "breach": [1, 0]})
    assert calibration(df)["brier"] < 1e-9


def test_detection_delay_finite_under_drift():
    assert detection_delay(0.05, 0.5, 0.05, 0.1) is not None


def test_false_freeze_rate_within_budget():
    assert false_freeze_rate(0.2, 0.05, 0.1, n=150, trials=100, seed=0) <= 0.1
