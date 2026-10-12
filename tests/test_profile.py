import numpy as np
from arena_limits.profile.correlation import effective_n, tetrachoric_corr
from arena_limits.profile.irt import fit_irt
from arena_limits.profile.pipeline import profile_matches

from arena_limits.arena.client import MockChatClient
from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, Scenario
from arena_limits.storage import Storage

SCENARIO = Scenario("parts_nego_v1", "lot", 100.0, 60.0, 95.0, 8)
OPPONENT = Opponent("opp_lowball", "lowball")
AGENTS = [
    AgentSpec("soft_a", "mock/soft", "soft", backbone="mock/soft"),
    AgentSpec("soft_b", "mock/soft", "soft", backbone="mock/soft"),
    AgentSpec("neutral_a", "mock/neutral", "neutral", backbone="mock/neutral"),
    AgentSpec("neutral_b", "mock/neutral", "neutral", backbone="mock/neutral"),
    AgentSpec("strict_a", "mock/strict", "strict", backbone="mock/strict"),
    AgentSpec("strict_b", "mock/strict", "strict", backbone="mock/strict"),
]


def _frame(k=40):
    client = MockChatClient()
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
