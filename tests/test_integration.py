import pytest
from fakes import FakeChatClient

from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, Scenario
from arena_limits.profile.pipeline import profile_matches
from arena_limits.risk.limits import compute_limits, pair_probability_from_storage
from arena_limits.storage import Storage

SCENARIO = Scenario("parts", "lot", 100.0, 60.0, 95.0, 8)
OPPONENTS = [Opponent("o_low", "lowball"), Opponent("o_guilt", "guilt_trip")]
AGENTS = [
    AgentSpec("soft_a", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("soft_b", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("neutral_a", "fake/neutral", "neutral", backbone="fake/neutral"),
    AgentSpec("strict_a", "fake/strict", "strict", backbone="fake/strict"),
]


def _storage(k=20):
    client = FakeChatClient()
    results = [
        run_match(agent, opponent, SCENARIO, seed=seed, client=client)
        for agent in AGENTS
        for opponent in OPPONENTS
        for seed in range(k)
    ]
    storage = Storage(":memory:")
    storage.insert_matches(results)
    return storage


def test_profiles_feed_pair_limits():
    storage = _storage()
    out = profile_matches(storage.load_matches(), bootstrap=0)
    storage.upsert_profiles(out["profiles"])
    storage.upsert_opponents(out["opponents"])
    assert not storage.load_profiles().empty

    p_pair = pair_probability_from_storage(storage, "soft_a", "o_low")
    assert 0.0 <= p_pair <= 1.0

    report = compute_limits(
        storage,
        [{"agent_id": "soft_a", "opponent_id": "o_low", "requested": 1000.0}],
        rho_b=0.2,
        budget_total=100.0,
    )
    assert report["rows"][0]["amount"] >= 0.0
    assert not storage.load_limits().empty
    storage.close()


def test_api_app_builds():
    pytest.importorskip("fastapi")
    from arena_limits.api import main

    assert main.app.title == "arena-limits"
