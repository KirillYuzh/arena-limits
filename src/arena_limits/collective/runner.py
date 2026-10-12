import numpy as np

from ..arena.client import ChatClient
from ..arena.types import AgentSpec, Opponent, RunContext, Scenario
from .bilateral import BilateralScenario, run_chain, run_market
from .metrics import design_effect_comparison, market_stats, session_stats
from .session import run_session


def _draw_pressures(rng, pool_size: int, shock_sd: float, independent: bool):
    """
    Множители давления на агентов сессии.

    ``independent=False`` — один общий шок на сессию (общий внешний фактор).
    ``independent=True`` — свой шок каждому агенту (контрольная рука): тогда
    зависимость в пуле может дать только сам агент, а не общее условие.

    Parameters
    ----------
    rng : numpy.random.Generator
        Генератор случайных чисел.
    pool_size : int
        Размер пула агентов.
    shock_sd : float
        Стандартное отклонение шока давления.
    independent : bool
        Рисовать ли шок отдельно каждому агенту.

    Returns
    -------
    list of float or None
        Множители давления; ``None``, если шок выключен.
    """
    if shock_sd <= 0.0:
        return None
    if independent:
        return list(1.0 + rng.normal(0.0, shock_sd, pool_size))
    return [1.0 + float(rng.normal(0.0, shock_sd))] * pool_size


def run_collective(cfg: dict, client: ChatClient, context: RunContext) -> dict:
    """
    Провести коллективное исследование по конфигу.

    Режимы: ``portfolio`` и ``concentration`` (пул против скриптованного
    противника, общий шок), ``control`` (тот же пул, но шоки независимы —
    контрольная рука для эффекта дизайна), ``market`` (агент против агента,
    раунд-робин) и ``delegation`` (цепочка мандата).

    Parameters
    ----------
    cfg : dict
        Конфиг: ``mode``, ``scenario``, ``agent_pool``, ``opponents``,
        ``sessions``, ``shock_sd`` (сила шока давления), ``shock_seed``; для
        ``market`` — ``seller_pool``/``buyer_pool``.
    client : ChatClient
        Клиент генерации ответов.
    context : RunContext
        Провинанс прогона.

    Returns
    -------
    dict
        Сессии (или сделки рынка, или цепочка) и агрегированные метрики.
    """
    mode = str(cfg.get("mode", "portfolio"))
    scenario = Scenario(**cfg["scenario"])
    n_sessions = int(cfg.get("sessions", 60))

    if mode in ("portfolio", "concentration", "control"):
        pool = [AgentSpec(**item) for item in cfg["agent_pool"]]
        shock_sd = float(cfg.get("shock_sd", 0.0))
        rng = np.random.default_rng(int(cfg.get("shock_seed", 0)))
        sessions = []
        for opponent_cfg in cfg["opponents"]:
            opponent = Opponent(**opponent_cfg)
            condition = f"{mode}|{opponent.opponent_id}|{scenario.scenario_id}"
            for index in range(n_sessions):
                pressures = _draw_pressures(
                    rng, len(pool), shock_sd, independent=(mode == "control")
                )
                sessions.append(
                    run_session(
                        pool,
                        opponent,
                        scenario,
                        condition,
                        index,
                        client,
                        context,
                        mode=mode,
                        pressures=pressures,
                    )
                )
        stats = {}
        for condition in sorted({session.condition for session in sessions}):
            group = [s for s in sessions if s.condition == condition]
            stats[condition] = session_stats(group)
        result = {"mode": mode, "sessions": sessions, "stats": stats}

        if mode == "portfolio" and cfg.get("control_arm"):
            control_sessions = []
            comparisons = {}
            for opponent_cfg in cfg["opponents"]:
                opponent = Opponent(**opponent_cfg)
                shared_condition = f"portfolio|{opponent.opponent_id}|{scenario.scenario_id}"
                condition = f"control|{opponent.opponent_id}|{scenario.scenario_id}"
                control = [
                    run_session(
                        pool,
                        opponent,
                        scenario,
                        condition,
                        index,
                        client,
                        context,
                        mode="control",
                        pressures=_draw_pressures(rng, len(pool), shock_sd, independent=True),
                    )
                    for index in range(n_sessions)
                ]
                control_sessions.extend(control)
                comparisons[opponent.opponent_id] = design_effect_comparison(
                    stats[shared_condition], session_stats(control)
                )
            result["control_sessions"] = control_sessions
            result["design_effect"] = comparisons
        return result

    if mode == "market":
        sellers = [AgentSpec(**item) for item in cfg["seller_pool"]]
        buyers = [AgentSpec(**item) for item in cfg["buyer_pool"]]
        bilateral = BilateralScenario(
            scenario_id=scenario.scenario_id,
            item=scenario.item,
            seller_floor=scenario.floor_price,
            buyer_cap=float(cfg.get("buyer_cap", scenario.target_offer)),
            max_rounds=scenario.max_rounds,
        )
        market = run_market(
            sellers,
            buyers,
            bilateral,
            f"market|{scenario.scenario_id}",
            client,
            context,
        )
        return {"mode": mode, "market": market, "stats": market_stats(market)}

    if mode == "delegation":
        levels = [AgentSpec(**item) for item in cfg["agent_pool"]]
        opponent = Opponent(**cfg["opponents"][0])
        chain = run_chain(
            levels,
            opponent,
            scenario,
            f"delegation|{scenario.scenario_id}",
            client,
            context,
        )
        return {"mode": mode, "chain": chain}

    raise ValueError(f"неизвестный режим: {mode}")
