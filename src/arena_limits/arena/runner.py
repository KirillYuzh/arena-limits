from .client import ChatClient, reasoning_mode
from .opponents import opponent_cap, render_offer
from .prompts import PROTOCOL_REMINDER, build_system_prompt
from .provenance import prompt_hash, provider_of
from .referee import closing_price, detect_breach, parse_action
from .types import AgentSpec, MatchResult, Opponent, RunContext, Scenario


def run_match(
    agent: AgentSpec,
    opponent: Opponent,
    scenario: Scenario,
    seed: int,
    client: ChatClient,
    repair: bool = True,
    context: RunContext | None = None,
    condition: str | None = None,
) -> MatchResult:
    """
    Провести одну состязательную сделку и зафиксировать исход с провинансом.

    Агент и противник обмениваются предложениями до закрытия сделки или
    исчерпания ``scenario.max_rounds``. Различаются исходы ``breach`` (сделка
    ниже мандата), ``deal_ok`` (сделка не ниже мандата), ``declined`` (валидный
    тег, сделка не закрыта) и ``protocol_fail`` (агент не выдал тег). Флаг
    ``conceded`` фиксирует уступку ниже пола даже без закрытия сделки.

    Parameters
    ----------
    agent : AgentSpec
        Конфигурация агента (модель, персона, мандат).
    opponent : Opponent
        Противник; его лестница задаёт давление и потолок готовности.
    scenario : Scenario
        Сценарий переговоров с ценой-полом.
    seed : int
        Номер прогона в ячейке.
    client : ChatClient
        Клиент генерации ответов агента.
    repair : bool
        Разрешить одну попытку починки протокола при отсутствии тега.
    context : RunContext or None
        Провинанс прогона.
    condition : str or None
        Идентификатор условия ячейки; по умолчанию совпадает с seed.

    Returns
    -------
    MatchResult
        Транскрипт, исход, флаги протокола, `finish_reason` и провинанс.
    """
    context = context or RunContext()
    condition = condition or str(seed)
    system = build_system_prompt(
        agent.prompt_template, scenario.item, scenario.floor_price, agent.mandate
    )
    cap = opponent_cap(opponent, scenario)
    messages: list[dict] = []
    transcript: list[dict] = []
    final_price = None
    opponent_final_offer = None
    conceded = False
    repaired = False
    strict = False
    action, price = "none", None
    finish_reason = ""
    served_model = ""
    rounds = 0

    for r in range(scenario.max_rounds):
        rounds = r + 1
        opponent_message, opponent_price = render_offer(opponent, scenario, r)
        opponent_final_offer = opponent_price
        messages.append({"role": "user", "content": opponent_message})
        transcript.append(
            {"role": "opponent", "content": opponent_message, "price": opponent_price}
        )

        meta = {
            "round": r,
            "max_rounds": scenario.max_rounds,
            "agent_id": agent.agent_id,
            "floor": scenario.floor_price,
        }
        reply = client.chat(
            agent.model,
            system,
            messages,
            seed=seed,
            temperature=context.temperature,
            meta=meta,
        )
        agent_message, finish_reason, served_model = (
            reply.text,
            reply.finish_reason,
            reply.served_model,
        )
        action, price, strict = parse_action(agent_message)

        if action == "none" and repair and not repaired:
            retry = [
                *messages,
                {"role": "assistant", "content": agent_message},
                {"role": "user", "content": PROTOCOL_REMINDER},
            ]
            reply = client.chat(
                agent.model,
                system,
                retry,
                seed=seed,
                temperature=context.temperature,
                meta=meta,
            )
            agent_message, finish_reason, served_model = (
                reply.text,
                reply.finish_reason,
                reply.served_model,
            )
            action, price, strict = parse_action(agent_message)
            repaired = True

        messages.append({"role": "assistant", "content": agent_message})
        transcript.append({"role": "agent", "content": agent_message})

        if action == "offer" and price is not None and price < scenario.floor_price:
            conceded = True

        price_close = closing_price(action, price, opponent_price, cap)
        if price_close is not None:
            final_price = price_close
            break

    protocol_ok = action != "none"
    breach = detect_breach(final_price, scenario.floor_price)
    if not protocol_ok:
        outcome = "protocol_fail"
    elif breach:
        outcome = "breach"
    elif final_price is not None:
        outcome = "deal_ok"
    else:
        outcome = "declined"

    return MatchResult(
        match_id=(
            f"{context.experiment_id}|{context.run_id}|{agent.agent_id}|"
            f"{opponent.opponent_id}|{scenario.scenario_id}|{condition}|{seed}"
        ),
        experiment_id=context.experiment_id,
        run_id=context.run_id,
        agent_id=agent.agent_id,
        opponent_id=opponent.opponent_id,
        scenario_id=scenario.scenario_id,
        condition=condition,
        seed=seed,
        backbone=agent.backbone or agent.model,
        llm_model=agent.model,
        provider=provider_of(agent.model),
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        prompt_hash=prompt_hash(system),
        agent_fingerprint=agent.fingerprint,
        git_sha=context.git_sha,
        final_price=final_price,
        opponent_final_offer=opponent_final_offer,
        breach=breach,
        outcome=outcome,
        protocol_ok=protocol_ok,
        strict=strict,
        repaired=repaired,
        conceded=conceded,
        finish_reason=finish_reason,
        served_model=served_model,
        reasoning_mode=reasoning_mode(agent.model),
        rounds=rounds,
        transcript=transcript,
    )
