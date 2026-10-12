import numpy as np
import pandas as pd
from fakes import FakeChatClient

from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, Scenario
from arena_limits.profile.correlation import (
    between_backbone_gap,
    effective_n,
    model_vs_persona_effect,
    tetrachoric_corr,
    within_backbone_variance,
)
from arena_limits.profile.irt import fit_irt
from arena_limits.profile.pipeline import profile_matches
from arena_limits.storage import Storage

SCENARIO = Scenario("parts_nego_v1", "lot", 100.0, 60.0, 95.0, 8)
OPPONENT = Opponent("opp_lowball", "lowball")
AGENTS = [
    AgentSpec("soft_a", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("soft_b", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("neutral_a", "fake/neutral", "neutral", backbone="fake/neutral"),
    AgentSpec("neutral_b", "fake/neutral", "neutral", backbone="fake/neutral"),
    AgentSpec("strict_a", "fake/strict", "strict", backbone="fake/strict"),
    AgentSpec("strict_b", "fake/strict", "strict", backbone="fake/strict"),
]


def _frame(k=40):
    client = FakeChatClient()
    results = [
        run_match(agent, OPPONENT, SCENARIO, seed=seed, client=client)
        for agent in AGENTS
        for seed in range(k)
    ]
    storage = Storage(":memory:")
    storage.insert_matches(results)
    return storage, storage.load_matches()


def test_effective_n_identity():
    assert effective_n(np.eye(5)) == 5.0


def test_effective_n_perfectly_correlated():
    assert abs(effective_n(np.ones((4, 4))) - 1.0) < 1e-9


def test_tetrachoric_high_for_identical():
    y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    assert tetrachoric_corr(y, y) > 0.9


def test_tetrachoric_near_zero_for_independent():
    y1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    y2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert abs(tetrachoric_corr(y1, y2)) < 0.4


def test_fit_irt_orders_models():
    _, df = _frame()
    result = fit_irt(df)
    assert result.p_hat["soft_a"] > result.p_hat["strict_a"]
    assert set(result.agents) == {agent.agent_id for agent in AGENTS}


def test_pipeline_writes_profiles_and_corr():
    storage, df = _frame()
    out = profile_matches(df, bootstrap=20)
    storage.upsert_profiles(out["profiles"])
    storage.upsert_opponents(out["opponents"])
    storage.upsert_corr(out["corr"])

    assert len(storage.load_profiles()) == len(AGENTS)
    assert not storage.load_corr().empty
    assert out["n_eff"] < len(AGENTS)


def test_pipeline_reports_monoculture_gap():
    _, df = _frame()
    out = profile_matches(df, bootstrap=0)
    assert out["rho"]["within"] > out["rho"]["between"]


def test_within_backbone_variance_zero_for_identical():
    df = pd.DataFrame({"backbone": ["a", "a", "a"], "p_hat": [0.5, 0.5, 0.5]})
    out = within_backbone_variance(df)
    assert out["a"]["std"] == 0.0
    assert out["a"]["n_agents"] == 3


def test_between_backbone_gap_positive():
    df = pd.DataFrame({"backbone": ["a", "a", "b", "b"], "p_hat": [0.1, 0.2, 0.8, 0.9]})
    assert between_backbone_gap(df) > 0.5


def test_model_vs_persona_effect_detects_gap():
    df = pd.DataFrame(
        {
            "backbone": ["a", "a", "a", "b", "b", "b"],
            "p_hat": [0.10, 0.11, 0.12, 0.80, 0.81, 0.82],
        }
    )
    out = model_vs_persona_effect(df)
    assert out["ratio"] > 5
    assert out["anova_p"] < 0.01


def test_model_vs_persona_effect_flat_when_no_gap():
    df = pd.DataFrame(
        {
            "backbone": ["a", "a", "a", "b", "b", "b"],
            "p_hat": [0.50, 0.51, 0.49, 0.50, 0.51, 0.49],
        }
    )
    out = model_vs_persona_effect(df)
    assert out["anova_p"] > 0.05
