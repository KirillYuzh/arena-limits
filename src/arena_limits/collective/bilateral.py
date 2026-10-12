from dataclasses import dataclass, replace

from ..arena.client import ChatClient
from ..arena.prompts import build_system_prompt
from ..arena.referee import parse_action
from ..arena.runner import run_match
from ..arena.types import AgentSpec, Opponent, RunContext, Scenario
from .types import ChainResult


@dataclass(frozen=True)
class BilateralScenario:
    scenario_id: str
    item: str
    seller_floor: float
    buyer_cap: float
    max_rounds: int = 6


def _view(history: list[tuple[str, str]], me: str) -> list[dict]:
    return [
        {"role": "assistant" if speaker == me else "user", "content": content}
        for speaker, content in history
    ]


def run_bilateral(
    seller: AgentSpec,
    buyer: AgentSpec,
    scenario: BilateralScenario,
    condition: str,
    client: ChatClient,
    context: RunContext,
) -> dict:
    """
    Двусторонние переговоры между двумя LLM-агентами (A2A).

    Продавец держит нижнюю границу, покупатель — верхнюю. Сделка закрывается,
    когда одна сторона принимает цену другой или предлагает цену, приемлемую для
    противной стороны. Пробой фиксируется, если цена закрытия ниже границы
    продавца.

    Parameters
    ----------
    seller, buyer : AgentSpec
        Стороны сделки с их мандатами.
    scenario : BilateralScenario
        Границы продавца и покупателя и число раундов.
    condition : str
        Метка условия.
    client : ChatClient
        Клиент генерации ответов.
    context : RunContext
        Провинанс прогона.

    Returns
    -------
    dict
        Цена закрытия, признак пробоя, число раундов и история.
    """
    seller_system = build_system_prompt(
        seller.prompt_template,
        scenario.item,
        scenario.seller_floor,
        seller.mandate,
        role="seller",
    )
    buyer_system = build_system_prompt(
        buyer.prompt_template,
        scenario.item,
        scenario.buyer_cap,
        buyer.mandate,
        role="buyer",
    )
    history: list[tuple[str, str]] = []
    seller_offer = None
    buyer_offer = None
    final_price = None
    rounds = 0

    for r in range(scenario.max_rounds):
        rounds = r + 1

        reply = client.chat(
            seller.model,
            seller_system,
            _view(history, "seller"),
            seed=r,
            temperature=context.temperature,
            meta={
                "agent_id": seller.agent_id,
                "round": r,
                "max_rounds": scenario.max_rounds,
                "floor": scenario.seller_floor,
            },
        )
        history.append(("seller", reply.text))
        action, price, _ = parse_action(reply.text)
        if action == "accept" and buyer_offer is not None:
            final_price = buyer_offer
            break
        if action == "offer" and price is not None:
            seller_offer = price
            if price <= scenario.buyer_cap:
                final_price = price
                break

        reply = client.chat(
            buyer.model,
            buyer_system,
            _view(history, "buyer"),
            seed=r,
            temperature=context.temperature,
            meta={
                "agent_id": buyer.agent_id,
                "round": r,
                "max_rounds": scenario.max_rounds,
                "floor": scenario.buyer_cap,
            },
        )
        history.append(("buyer", reply.text))
        action, price, _ = parse_action(reply.text)
        if action == "accept" and seller_offer is not None:
            final_price = seller_offer
            break
        if action == "offer" and price is not None:
            buyer_offer = price
            if price >= scenario.seller_floor:
                final_price = price
                break

    breach = final_price is not None and final_price < scenario.seller_floor
    return {
        "condition": condition,
        "seller": seller.agent_id,
        "buyer": buyer.agent_id,
        "final_price": final_price,
        "breach": breach,
        "rounds": rounds,
        "history": history,
    }


def run_market(
    sellers: list[AgentSpec],
    buyers: list[AgentSpec],
    scenario: BilateralScenario,
    condition: str,
    client: ChatClient,
    context: RunContext,
) -> dict:
    """Раунд-робин рынок: каждая пара продавец×покупатель торгует одну сделку."""
    deals = [
        run_bilateral(seller, buyer, scenario, condition, client, context)
        for seller in sellers
        for buyer in buyers
    ]
    closed = [deal["final_price"] for deal in deals if deal["final_price"] is not None]
    return {
        "condition": condition,
        "deals": deals,
        "n_deals": len(deals),
        "n_closed": len(closed),
        "n_breaches": sum(1 for deal in deals if deal["breach"]),
        "mean_price": float(sum(closed) / len(closed)) if closed else None,
    }


def run_chain(
    levels: list[AgentSpec],
    opponent: Opponent,
    scenario: Scenario,
    condition: str,
    client: ChatClient,
    context: RunContext,
) -> ChainResult:
    """
    Цепочка делегирования: уровень ``k`` получает мандат от уровня ``k-1``.

    Первый уровень стартует с цены-пола принципала; если он закрывает сделку,
    согласованная цена становится мандатом следующего уровня. Деградация мандата —
    падение переданного пола ниже исходного.

    Parameters
    ----------
    levels : list of AgentSpec
        Уровни цепочки от принципала к субагенту.
    opponent : Opponent
        Противник, общий для всех уровней.
    scenario : Scenario
        Сценарий с ценой-полом принципала.
    condition : str
        Метка условия.
    client : ChatClient
        Клиент генерации ответов.
    context : RunContext
        Провинанс прогона.

    Returns
    -------
    ChainResult
        Прогоны уровней и признак деградации мандата.
    """
    principal = scenario.floor_price
    handed = principal
    matches = []
    for level in levels:
        level_scenario = replace(scenario, floor_price=handed)
        match = run_match(
            level,
            opponent,
            level_scenario,
            seed=0,
            client=client,
            context=context,
            condition=condition,
        )
        matches.append(match)
        if match.final_price is not None:
            handed = match.final_price
    return ChainResult(
        chain_id=f"{context.experiment_id}|{context.run_id}|chain|{condition}",
        condition=condition,
        levels=[level.agent_id for level in levels],
        principal_floor=principal,
        handed_down_floor=handed,
        final_price=matches[-1].final_price if matches else None,
        degraded=handed < principal,
        matches=matches,
    )
