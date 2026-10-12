from collections.abc import Sequence
from dataclasses import replace

from ..arena.client import ChatClient
from ..arena.runner import run_match
from ..arena.types import AgentSpec, Opponent, RunContext, Scenario
from .types import SessionResult


def session_shortfall(matches, floor_price: float) -> float:
    return float(
        sum(
            max(floor_price - match.final_price, 0.0)
            for match in matches
            if match.breach and match.final_price is not None
        )
    )


def run_session(
    pool: Sequence[AgentSpec],
    opponent: Opponent,
    scenario: Scenario,
    condition: str,
    session_index: int,
    client: ChatClient,
    context: RunContext,
    mode: str = "portfolio",
    pressures: Sequence[float] | None = None,
) -> SessionResult:
    """
    Провести одну коллективную сессию: пул агентов против одного противника.

    Все агенты пула работают в одном условии — одна траектория давления. Это и
    есть «общий шок»: если пробои агентов скоррелированы, они случаются в одних
    и тех же сессиях, и это можно измерить напрямую, а не выводить из корреляций.

    Parameters
    ----------
    pool : sequence of AgentSpec
        Агенты на нашей стороне.
    opponent : Opponent
        Противник, общий для всей сессии.
    scenario : Scenario
        Сценарий переговоров с ценой-полом.
    condition : str
        Метка условия (траектории давления).
    session_index : int
        Номер повтора сессии в этом условии.
    client : ChatClient
        Клиент генерации ответов.
    context : RunContext
        Провинанс прогона.
    mode : str
        Режим коллективного исследования (``portfolio`` или ``concentration``).
    pressures : sequence of float or None
        Множители давления на агента: общие для сессии (общий шок) или разные
        (контрольная рука с независимыми шоками).

    Returns
    -------
    SessionResult
        Прогоны агентов и агрегаты сессии (число пробоев, недостача, флаги).
    """
    matches = []
    for index, agent in enumerate(pool):
        current = opponent
        if pressures is not None:
            current = replace(opponent, pressure=float(pressures[index]))
        matches.append(
            run_match(
                agent,
                current,
                scenario,
                seed=session_index,
                client=client,
                context=context,
                condition=condition,
            )
        )
    breaches = sum(1 for match in matches if match.breach)
    return SessionResult(
        session_id=(
            f"{context.experiment_id}|{context.run_id}|{mode}|{condition}|{session_index}"
        ),
        experiment_id=context.experiment_id,
        run_id=context.run_id,
        mode=mode,
        condition=condition,
        pool_size=len(pool),
        breaches=breaches,
        shortfall=session_shortfall(matches, scenario.floor_price),
        joint_breach=breaches >= 1,
        co_breach=breaches >= 2,
        matches=matches,
    )
