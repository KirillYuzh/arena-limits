from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass
class IRTResult:
    agents: list[str]
    opponents: list[str]
    scenarios: list[str]
    a: dict[str, float]
    b: dict[str, float]
    gamma: dict[str, float]
    intercept: float
    p_hat: dict[str, float]
    log_loss: float


def _onehot(series: pd.Series, categories: list[str]) -> np.ndarray:
    return (
        pd.get_dummies(series)
        .reindex(columns=categories, fill_value=0)
        .to_numpy(dtype=float)
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fit_irt(df: pd.DataFrame, C: float = 1.0) -> IRTResult:
    """
    Оценить парную логит-модель пробоя.

    Модель имеет вид ``logit P(breach | i, j, c) = alpha + a_i + b_j + gamma_c``,
    где ``a_i`` — восприимчивость агента, ``b_j`` — убедительность противника,
    ``gamma_c`` — фиксированный эффект сценария. Оценивается L2-логистической
    регрессией по one-hot признакам, поэтому отдельная реализация IRT не нужна.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов со столбцами ``agent_id``, ``opponent_id``,
        ``scenario_id`` и ``breach``.
    C : float
        Обратная сила L2-регуляризации.

    Returns
    -------
    IRTResult
        Коэффициенты, свободный член и ``p_hat`` — вероятность пробоя,
        маргинализованная по всем парам противник×сценарий.
    """
    agents = sorted(df["agent_id"].unique())
    opponents = sorted(df["opponent_id"].unique())
    scenarios = sorted(df["scenario_id"].unique())

    design = np.hstack(
        [
            _onehot(df["agent_id"], agents),
            _onehot(df["opponent_id"], opponents),
            _onehot(df["scenario_id"], scenarios),
        ]
    )
    target = df["breach"].astype(int).to_numpy()
    if np.unique(target).size < 2:
        raise ValueError("для оценки модели нужны оба класса исхода")

    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(design, target)

    coef = model.coef_[0]
    intercept = float(model.intercept_[0])
    n_agents, n_opponents = len(agents), len(opponents)
    a = dict(zip(agents, coef[:n_agents].tolist()))
    b = dict(zip(opponents, coef[n_agents : n_agents + n_opponents].tolist()))
    gamma = dict(zip(scenarios, coef[n_agents + n_opponents :].tolist()))

    p_hat = {
        agent: float(
            np.mean(
                [
                    _sigmoid(intercept + a[agent] + b[opponent] + gamma[scenario])
                    for opponent in opponents
                    for scenario in scenarios
                ]
            )
        )
        for agent in agents
    }

    return IRTResult(
        agents=agents,
        opponents=opponents,
        scenarios=scenarios,
        a=a,
        b=b,
        gamma=gamma,
        intercept=intercept,
        p_hat=p_hat,
        log_loss=float(log_loss(target, model.predict_proba(design)[:, 1])),
    )
