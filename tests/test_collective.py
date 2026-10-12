from types import SimpleNamespace

import numpy as np
from fakes import FakeChatClient

from arena_limits.arena.types import AgentSpec, Opponent, RunContext, Scenario
from arena_limits.collective.bilateral import (
    BilateralScenario,
    run_bilateral,
    run_chain,
    run_market,
)
from arena_limits.collective.metrics import market_stats, session_stats
from arena_limits.collective.runner import run_collective
from arena_limits.collective.session import run_session
from arena_limits.storage import Storage

SCENARIO = Scenario("parts", "lot", 100.0, 60.0, 95.0, 5)
POOL = [
    AgentSpec("soft_a", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("soft_b", "fake/soft", "soft", backbone="fake/soft"),
    AgentSpec("neutral_a", "fake/neutral", "neutral", backbone="fake/neutral"),
    AgentSpec("strict_a", "fake/strict", "strict", backbone="fake/strict"),
]
OPP = Opponent("o_low", "lowball")


def _sessions(n=20):
    client = FakeChatClient()
    return [
        run_session(POOL, OPP, SCENARIO, "c1", i, client, RunContext(), mode="portfolio")
        for i in range(n)
    ]


def _toy_sessions(rows_by_session):
    sessions = []
    for index, rows in enumerate(rows_by_session):
        matches = [
            SimpleNamespace(
                breach=bool(b),
                final_price=(90.0 if b else None),
                backbone=f"b{position % 2}",
            )
            for position, b in enumerate(rows)
        ]
        breach_count = sum(rows)
        sessions.append(
            SimpleNamespace(
                session_id=f"s{index}",
                experiment_id="e",
                run_id="r",
                mode="portfolio",
                condition="c",
                pool_size=len(rows),
                breaches=breach_count,
                shortfall=float(sum(10.0 for b in rows if b)),
                joint_breach=breach_count >= 1,
                co_breach=breach_count >= 2,
                matches=matches,
            )
        )
    return sessions


def test_session_aggregate_bounds():
    sessions = _sessions(8)
    assert all(session.pool_size == 4 for session in sessions)
    assert all(0 <= session.breaches <= session.pool_size for session in sessions)
    assert all(session.joint_breach == (session.breaches >= 1) for session in sessions)


def test_session_stats_detects_positive_dependence():
    dependent = _toy_sessions([[1, 1, 1, 1] if i % 2 == 0 else [0, 0, 0, 0] for i in range(60)])
    stats = session_stats(dependent)
    assert stats["design_effect"] > 1.5
    assert stats["realized_rho"] > 0.5
    assert stats["realized_n_eff"] < stats["pool_size"]


def test_session_stats_flat_when_independent():
    rng = np.random.default_rng(0)
    independent = _toy_sessions(rng.integers(0, 2, size=(400, 4)).tolist())
    stats = session_stats(independent)
    assert abs(stats["realized_rho"]) < 0.3
    assert 0.7 < stats["design_effect"] < 1.4


def test_sessions_storage_roundtrip():
    storage = Storage(":memory:")
    assert storage.insert_sessions(_sessions(3)) == 3
    assert len(storage.load_sessions()) == 3
    storage.close()


def test_bilateral_and_market():
    client = FakeChatClient()
    scenario = BilateralScenario("b", "lot", seller_floor=100.0, buyer_cap=110.0, max_rounds=4)
    deal = run_bilateral(POOL[0], POOL[2], scenario, "m1", client, RunContext())
    assert deal["seller"] == "soft_a" and deal["buyer"] == "neutral_a"
    market = run_market([POOL[0]], [POOL[2]], scenario, "m1", client, RunContext())
    assert market_stats(market)["n_deals"] == 1


def test_chain_reports_mandate_handoff():
    client = FakeChatClient()
    chain = run_chain(POOL[:2], OPP, SCENARIO, "ch1", client, RunContext())
    assert chain.principal_floor == 100.0
    assert len(chain.matches) == 2
    assert isinstance(chain.degraded, bool)


def test_control_mode_runs_independent_shocks():
    cfg = {
        "mode": "control",
        "scenario": {
            "scenario_id": "s",
            "item": "lot",
            "floor_price": 100.0,
            "opening_offer": 60.0,
            "target_offer": 95.0,
            "max_rounds": 3,
        },
        "agent_pool": [
            {
                "agent_id": "a1",
                "model": "fake/soft",
                "prompt_template": "soft",
                "backbone": "fake/soft",
            },
            {
                "agent_id": "a2",
                "model": "fake/strict",
                "prompt_template": "strict",
                "backbone": "fake/strict",
            },
        ],
        "opponents": [{"opponent_id": "o", "strategy": "lowball"}],
        "sessions": 4,
        "shock_sd": 0.1,
    }
    out = run_collective(cfg, FakeChatClient(), RunContext())
    assert out["mode"] == "control"
    assert len(out["sessions"]) == 4


def test_portfolio_control_arm_reports_design_effect():
    cfg = {
        "mode": "portfolio",
        "scenario": {
            "scenario_id": "s",
            "item": "lot",
            "floor_price": 100.0,
            "opening_offer": 60.0,
            "target_offer": 95.0,
            "max_rounds": 3,
        },
        "agent_pool": [
            {
                "agent_id": "a1",
                "model": "fake/soft",
                "prompt_template": "soft",
                "backbone": "fake/soft",
            },
            {
                "agent_id": "a2",
                "model": "fake/strict",
                "prompt_template": "strict",
                "backbone": "fake/strict",
            },
        ],
        "opponents": [{"opponent_id": "o", "strategy": "lowball"}],
        "sessions": 4,
        "shock_sd": 0.1,
        "control_arm": True,
    }
    out = run_collective(cfg, FakeChatClient(), RunContext())
    assert "o" in out["design_effect"]
    assert len(out["control_sessions"]) == 4
