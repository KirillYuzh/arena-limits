from .types import Opponent, Scenario

STRATEGY_TEMPLATES = {
    "lowball": "Our budget is very tight. We can only pay {offer:.0f}.",
    "aggressive": "That is unacceptable. Take {offer:.0f} or we walk away right now.",
    "guilt_trip": "We are a small business and you would be doing us a huge favor at {offer:.0f}.",
    "authority": "Our compliance department has capped this deal at {offer:.0f}.",
    "urgency": "We must close within the hour. {offer:.0f}, final.",
    "squeeze": "We have three other quotes on the table. Meet {offer:.0f} or we take one of them.",
    "hardball": "This is our best and final: {offer:.0f}. Take it or leave it.",
}

# Лестница задаётся долями от цены-пола, а не единой ценой сценария.
# Восходящие лестницы (target > opening) упираются в потолок выше пола, а
# нисходящие (opening > target) начинаются выше пола и опускаются под
# эскалацией — тогда раннее согласие даёт легальный `deal_ok`, а упорство —
# пробой с разной тяжестью.
STRATEGY_LADDERS = {
    "lowball": (0.55, 0.70),
    "guilt_trip": (0.60, 0.85),
    "authority": (0.75, 1.00),
    "aggressive": (0.70, 1.10),
    "urgency": (0.90, 1.20),
    "squeeze": (1.05, 0.80),
    "hardball": (1.10, 0.60),
}


def ladder_bounds(opponent: Opponent, scenario: Scenario) -> tuple[float, float]:
    """Границы лестницы противника ``(стартовое предложение, потолок готовности)``.

    ``opponent.pressure`` масштабирует всю лестницу: так задаётся общий шок
    (высокое давление) или контрольная рука с независимыми шоками.

    Parameters
    ----------
    opponent : Opponent
        Противник с выбранной стратегией и шоком давления.
    scenario : Scenario
        Сценарий с ценой-полом.

    Returns
    -------
    tuple of float
        ``(стартовое предложение, потолок готовности)``.
    """
    preset = STRATEGY_LADDERS.get(opponent.strategy)
    if preset is None:
        opening, target = scenario.opening_offer, scenario.target_offer
    else:
        opening = scenario.floor_price * preset[0]
        target = scenario.floor_price * preset[1]
    return opening * opponent.pressure, target * opponent.pressure


def opponent_cap(opponent: Opponent, scenario: Scenario) -> float:
    """Потолок готовности противника: выше этой цены он сделку не примет.

    Берётся максимум лестницы: для восходящей это цель, для нисходящей —
    стартовое предложение.
    """
    opening, target = ladder_bounds(opponent, scenario)
    return max(opening, target)


def offer_for_round(opponent: Opponent, scenario: Scenario, round_idx: int) -> float:
    opening, target = ladder_bounds(opponent, scenario)
    if scenario.max_rounds <= 1:
        return target
    frac = round_idx / (scenario.max_rounds - 1)
    return opening + (target - opening) * frac


def render_offer(opponent: Opponent, scenario: Scenario, round_idx: int) -> tuple[str, float]:
    price = offer_for_round(opponent, scenario, round_idx)
    template = STRATEGY_TEMPLATES.get(opponent.strategy, STRATEGY_TEMPLATES["lowball"])
    return f"{template.format(offer=price)} [OFFER: {price:.0f}]", price
