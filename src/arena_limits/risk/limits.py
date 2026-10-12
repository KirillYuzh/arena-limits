import numpy as np
from scipy.stats import norm

from .es_alloc import portfolio_es
from .vasicek import vasicek_pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def pair_capital(p_pair: float, rho_b: float, alpha: float = 0.95, lgd: float = 1.0) -> float:
    """
    Капитал на единицу экспозиции для одной пары.

    Это незапланированные потери ``UL = lgd * (PD(z_stress) - p)`` при стрессовом
    квантиле системного фактора ``z_stress = Phi^-1(1 - alpha)`` (плохое
    состояние рынка — нижний хвост).

    Parameters
    ----------
    p_pair : float
        Безусловная вероятность пробоя пары.
    rho_b : float
        Корреляция внутри бэкбона.
    alpha : float
        Уровень уверенности стресса.
    lgd : float
        Доля потерь при пробое.

    Returns
    -------
    float
        Капитал на единицу экспозиции, не меньше нуля.
    """
    stressed = float(vasicek_pd([p_pair], rho_b, [norm.ppf(1.0 - alpha)])[0, 0])
    return float(lgd * max(stressed - p_pair, 0.0))


def quote_pair(
    p_pair: float,
    rho_b: float,
    budget: float,
    requested: float,
    alpha: float = 0.95,
    lgd: float = 1.0,
) -> float:
    """
    Лимит по паре: минимум из запрошенного и того, что выдерживает бюджет
    """
    capital = pair_capital(p_pair, rho_b, alpha, lgd)
    if capital <= 0.0:
        return float(requested)
    return float(min(requested, budget / capital))


def pair_probability_from_storage(storage, agent_id: str, opponent_id: str) -> float:
    """Парная вероятность пробоя из профилей: ``sigmoid(a + a_i + b_j)``."""
    profile = storage.get_agent_factors(agent_id)
    b_j = storage.get_opponent_factor(opponent_id)
    if profile is None or b_j is None:
        return float("nan")
    return _sigmoid(float(profile[1]) + float(profile[0]) + b_j)


def compute_limits(
    storage,
    requests: list[dict],
    rho_b: float,
    budget_total: float,
    alpha: float = 0.95,
    lgd: float = 1.0,
    n_sims: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Собрать парные лимиты из профилей и бюджета ES и записать их в хранилище.

    Бюджет делится между парами пропорционально запрошенным суммам, затем
    портфель проверяется Монте-Карло-симуляцией: реальный ``ES`` не должен
    превышать бюджет.

    Parameters
    ----------
    storage : Storage
        Хранилище с профилями и противниками.
    requests : list of dict
        Пары с ключами ``agent_id``, ``opponent_id`` и ``requested``.
    rho_b : float
        Корреляция внутри бэкбона.
    budget_total : float
        Общий бюджет хвостовых потерь.

    Returns
    -------
    dict
        ``rows`` (для записи), ``portfolio`` (var/es после лимитов) и ``budget``.
    """
    total_requested = sum(float(r["requested"]) for r in requests) or 1.0
    rows = []
    for request in requests:
        agent_id, opponent_id = request["agent_id"], request["opponent_id"]
        p_pair = pair_probability_from_storage(storage, agent_id, opponent_id)
        share = float(request["requested"]) / total_requested
        limit = quote_pair(
            p_pair, rho_b, budget_total * share, float(request["requested"]), alpha, lgd
        )
        rows.append(
            {
                "agent_id": agent_id,
                "opponent_id": opponent_id,
                "cluster": str(request.get("cluster", "")),
                "amount": limit,
                "p_pair": p_pair,
                "rho_b": rho_b,
                "n_eff": request.get("n_eff"),
                "frozen": False,
            }
        )

    assets = [r for r in rows if not np.isnan(r["p_pair"])]
    if assets:
        portfolio = portfolio_es(
            [r["p_pair"] for r in assets],
            [r["amount"] for r in assets],
            rho_b,
            alpha=alpha,
            n_sims=n_sims,
            lgd=lgd,
            seed=seed,
        )
    else:
        portfolio = {"var": 0.0, "es": 0.0, "contributions": [], "tail_prob": 0.0}

    storage.upsert_limits(rows)
    return {"rows": rows, "portfolio": portfolio, "budget": budget_total}
