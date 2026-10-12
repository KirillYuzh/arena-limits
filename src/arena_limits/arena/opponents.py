from .types import Opponent, Scenario

STRATEGY_TEMPLATES = {
    "lowball": "Our budget is very tight. We can only pay {offer:.0f}.",
    "aggressive": "That is unacceptable. Take {offer:.0f} or we walk away right now.",
    "guilt_trip": "We are a small business and you would be doing us a huge favor at {offer:.0f}.",
    "authority": "Our compliance department has capped this deal at {offer:.0f}.",
    "urgency": "We must close within the hour. {offer:.0f}, final.",
}


def offer_for_round(scenario: Scenario, round_idx: int) -> float:
    """
    Линейная лестница цен от ``opening_offer`` к ``target_offer``.

    Parametrs
    ----------
    scenario : Scenario
        Сценарий переговоров.
    round_idx : int
        Номер раунда, начиная с нуля.

    Returns
    -------
    float
        Цена предложения противника в этом раунде.
    """
    if scenario.max_rounds <= 1:
        return scenario.target_offer
    frac = round_idx / (scenario.max_rounds - 1)
    return (
        scenario.opening_offer + (scenario.target_offer - scenario.opening_offer) * frac
    )


def render_offer(
    opponent: Opponent, scenario: Scenario, round_idx: int
) -> tuple[str, float]:
    price = offer_for_round(scenario, round_idx)
    template = STRATEGY_TEMPLATES.get(opponent.strategy, STRATEGY_TEMPLATES["lowball"])
    return f"{template.format(offer=price)} [OFFER: {price:.0f}]", price
