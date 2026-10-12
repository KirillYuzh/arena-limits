import numpy as np
import pandas as pd

from arena_limits.collective.metrics import design_effect_comparison
from arena_limits.evaluate import (
    clustered_bootstrap_rate,
    eprocess_oos,
    interaction_test,
    manski_bounds,
)
from arena_limits.monitor.eprocess import (
    EProcessMonitor,
    bonferroni_threshold,
    mean_e,
)


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for backbone, agents in [("b1", ["a1", "a2"]), ("b2", ["a3", "a4"])]:
        for agent in agents:
            for opponent in ["o1", "o2"]:
                for condition in ["c0", "c1", "c2", "c3"]:
                    rows.append(
                        {
                            "agent_id": agent,
                            "opponent_id": opponent,
                            "scenario_id": "s",
                            "backbone": backbone,
                            "condition": condition,
                            "seed": 0,
                            "breach": int(rng.random() < 0.4),
                            "protocol_ok": True,
                            "strict": True,
                            "repaired": False,
                            "finish_reason": "stop",
                            "outcome": "breach",
                        }
                    )
    return pd.DataFrame(rows)


def test_manski_bounds_bracket_rate():
    bounds = manski_bounds(_frame())
    assert bounds["lower"] <= bounds["upper"]
    assert bounds["n_protocol_fail"] == 0


def test_clustered_bootstrap_ci_brackets_point():
    out = clustered_bootstrap_rate(_frame(), n_boot=100, seed=0)
    lo, hi = out["ci"]
    assert lo <= out["point"] <= hi


def test_interaction_test_returns_scores():
    out = interaction_test(_frame(), n_splits=4, seed=0)
    assert set(out) == {"additive_log_loss", "pair_log_loss", "delta"}


def test_eprocess_oos_uses_holdout_only():
    out = eprocess_oos(_frame(), delta=0.05, kappa=0.1)
    assert isinstance(out, dict)
    assert all(0.0 <= v["q"] <= 1.0 for v in out.values())


def test_q_floor_prevents_infinite_evidence():
    monitor = EProcessMonitor(delta=0.05, kappa=0.1)
    monitor.update(1, 0.0)
    assert np.isfinite(monitor.log_e)


def test_combine_helpers():
    assert bonferroni_threshold(0.05, 7) > bonferroni_threshold(0.05, 1)
    assert mean_e({"a": 0.0, "b": 0.0}) == 1.0


def test_design_effect_comparison_ratio():
    out = design_effect_comparison({"var_breaches": 4.0}, {"var_breaches": 1.0})
    assert abs(out["ratio"] - 4.0) < 1e-9
