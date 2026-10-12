from .client import ChatClient
from .opponents import render_offer
from .prompts import SYSTEM_PROMPTS
from .referee import detect_breach, parse_action
from .types import AgentSpec, MatchResult, Opponent, Scenario


def run_match(
    agent: AgentSpec,
    opponent: Opponent,
    scenario: Scenario,
    seed: int,
    client: ChatClient,
) -> MatchResult:
    """
    Провести один состязательный раунд переговоров.

    Агент и противник обмениваются предложениями до закрытия сделки или
    исчерпания ``scenario.max_rounds``. Пробой фиксирует рефери, если агент
    принял цену ниже цены-пола.

    Parametrs
    ----------
    agent : AgentSpec
        Конфигурация агента (модель и промпт).
    opponent : Opponent
        Якорный противник.
    scenario : Scenario
        Сценарий переговоров с ценой-полом.
    seed : int
        Seed прогона; при детерминированном клиенте результат воспроизводим.
    client : ChatClient
        Клиент генерации ответов агента.

    Returns
    -------
    MatchResult
        Транскрипт, итоговая цена, флаг пробоя и число раундов.
    """
    system = SYSTEM_PROMPTS.get(agent.prompt_template, SYSTEM_PROMPTS["neutral"])
    messages: list[dict] = []
    transcript: list[dict] = []
    final_price = None
    breach = False
    rounds = 0

    for r in range(scenario.max_rounds):
        rounds = r + 1

        opponent_message, opponent_price = render_offer(opponent, scenario, r)
        messages.append({"role": "user", "content": opponent_message})
        transcript.append(
            {"role": "opponent", "content": opponent_message, "price": opponent_price}
        )

        agent_message = client.chat(
            agent.model,
            system,
            messages,
            seed=seed,
            meta={
                "round": r,
                "max_rounds": scenario.max_rounds,
                "agent_id": agent.agent_id,
                "floor": scenario.floor_price,
            },
        )
        messages.append({"role": "assistant", "content": agent_message})
        transcript.append({"role": "agent", "content": agent_message})

        action, price = parse_action(agent_message)
        if action == "accept":
            final_price = opponent_price
            breach = detect_breach(final_price, scenario.floor_price)
            break
        if action == "offer" and price is not None and price <= scenario.target_offer:
            final_price = price
            breach = detect_breach(final_price, scenario.floor_price)
            break

    return MatchResult(
        match_id=f"{agent.agent_id}|{opponent.opponent_id}|{scenario.scenario_id}|{seed}",
        agent_id=agent.agent_id,
        opponent_id=opponent.opponent_id,
        scenario_id=scenario.scenario_id,
        seed=seed,
        backbone=agent.backbone or agent.model,
        llm_model=agent.model,
        final_price=final_price,
        breach=breach,
        rounds=rounds,
        transcript=transcript,
    )
