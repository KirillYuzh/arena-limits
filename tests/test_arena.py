import pytest
from fakes import FakeChatClient

from arena_limits.arena.client import Reply
from arena_limits.arena.opponents import opponent_cap
from arena_limits.arena.prompts import build_system_prompt
from arena_limits.arena.referee import closing_price, detect_breach, parse_action
from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, RunContext, Scenario
from arena_limits.storage import Storage


def test_parse_strict_tag():
    assert parse_action("reason here\n[OFFER: 105]") == ("offer", 105.0, True)


def test_parse_non_strict_fallback():
    assert parse_action("I accept. [ACCEPT]") == ("accept", None, False)


def test_parse_trailing_junk_is_not_strict():
    assert parse_action("[OFFER: 90]\n# note after tag") == ("offer", 90.0, False)


def test_opponent_cap_uses_ladder_max():
    scenario = Scenario("s", "lot", 100.0, 60.0, 95.0, 3)
    assert opponent_cap(Opponent("h", "hardball"), scenario) == pytest.approx(110.0)
    assert opponent_cap(Opponent("l", "lowball"), scenario) == pytest.approx(70.0)


def test_closing_price_uses_opponent_cap():
    assert closing_price("offer", 105.0, 60.0, 110.0) == 105.0
    assert detect_breach(105.0, 100.0) is False
    assert closing_price("offer", 90.0, 60.0, 95.0) == 90.0
    assert closing_price("offer", 99.0, 60.0, 95.0) is None
    assert closing_price("accept", None, 60.0, 95.0) == 60.0


def test_mandate_modes_differ():
    floor = build_system_prompt("neutral", "lot", 100.0, "floor")
    none = build_system_prompt("neutral", "lot", 100.0, "none")
    assert "must not" in floor and "100" in floor
    assert "must not" not in none


def test_buyer_prompt_has_cap():
    buyer = build_system_prompt("neutral", "lot", 100.0, role="buyer")
    assert "maximum price" in buyer


def test_deal_ok_when_agent_holds_above_floor():
    scenario = Scenario("s", "lot", 100.0, 60.0, 95.0, 3)
    opponent = Opponent("o", "urgency")

    class Firm:
        def chat(self, model, system, messages, seed, temperature=0.7, meta=None):
            return Reply("Our price.\n[OFFER: 105]", finish_reason="stop")

    result = run_match(
        AgentSpec("a", "fake/x"), opponent, scenario, 0, Firm(), context=RunContext()
    )
    assert result.outcome == "deal_ok"
    assert result.final_price == 105.0
    assert result.breach is False
    assert result.finish_reason == "stop"
    assert result.reasoning_mode in {"off", "auto"}


def test_run_match_protocol_fail():
    scenario = Scenario("s", "lot", 100.0, 60.0, 95.0, 2)
    opponent = Opponent("o", "lowball")

    class Silent:
        def chat(self, model, system, messages, seed, temperature=0.7, meta=None):
            return Reply("I would rather not answer.", finish_reason="stop")

    result = run_match(
        AgentSpec("a", "fake/x"), opponent, scenario, 0, Silent(), context=RunContext()
    )
    assert result.outcome == "protocol_fail"
    assert result.protocol_ok is False
    assert result.final_price is None


def test_storage_keeps_provenance():
    scenario = Scenario("s", "lot", 100.0, 60.0, 95.0, 3)
    opponent = Opponent("o", "lowball")
    context = RunContext(
        experiment_id="exp", run_id="r1", temperature=0.3, max_tokens=64, git_sha="abc"
    )
    result = run_match(
        AgentSpec("a", "fake/soft", "soft"),
        opponent,
        scenario,
        0,
        FakeChatClient(),
        context=context,
    )
    storage = Storage(":memory:")
    storage.insert_matches([result])
    frame = storage.load_matches(experiment_id="exp", run_id="r1")
    row = frame.iloc[0]
    assert len(frame) == 1
    assert row["provider"] == "fake"
    assert row["temperature"] == 0.3
    assert row["max_tokens"] == 64
    assert row["git_sha"] == "abc"
    assert row["agent_fingerprint"] and row["prompt_hash"]
    assert row["finish_reason"] and row["reasoning_mode"]
    assert "outcome" in frame.columns and "protocol_ok" in frame.columns
    storage.close()


def test_storage_separates_runs():
    scenario = Scenario("s", "lot", 100.0, 60.0, 95.0, 3)
    opponent = Opponent("o", "lowball")
    storage = Storage(":memory:")
    for run in ("r1", "r2"):
        result = run_match(
            AgentSpec("a", "fake/soft", "soft"),
            opponent,
            scenario,
            0,
            FakeChatClient(),
            context=RunContext(run_id=run),
        )
        storage.insert_matches([result])
    assert len(storage.load_matches()) == 2
    assert len(storage.load_matches(run_id="r1")) == 1
    storage.close()
