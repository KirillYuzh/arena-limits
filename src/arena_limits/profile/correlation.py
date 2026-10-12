import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm


def _bvn_cdf(h1: float, h2: float, rho: float) -> float:
    if abs(rho) < 1e-10:
        return float(norm.cdf(h1) * norm.cdf(h2))
    cov = [[1.0, rho], [rho, 1.0]]
    return float(multivariate_normal.cdf([h1, h2], mean=[0.0, 0.0], cov=cov))


def tetrachoric_corr(y1, y2) -> float:
    """
    Тетрахорическая корреляция двух бинарных векторов.

    Оценивает корреляцию латентных нормальных переменных по таблице
    сопряжённости 2×2, то есть решает обратную задачу к двумерной нормальной
    CDF. Обычная корреляция Пирсона для индикаторов пробоя смещена, поэтому
    нужна именно тетрахорическая.

    Parameters
    ----------
    y1, y2 : array-like
        Бинарные векторы одинаковой длины; NaN игнорируются попарно.

    Returns
    -------
    float
        Оценка в ``[-0.999, 0.999]``. При вырожденных маргиналах (все нули или
        все единицы) возвращается ``0.0``; при выходе корня за границы диапазона —
        ближайшая граница.
    """
    a = np.asarray(y1, dtype=float)
    b = np.asarray(y2, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 4:
        return 0.0

    p1, p2 = a.mean(), b.mean()
    if p1 <= 0.0 or p1 >= 1.0 or p2 <= 0.0 or p2 >= 1.0:
        return 0.0

    p11 = float(np.mean((a == 1) & (b == 1)))
    h1, h2 = float(norm.ppf(p1)), float(norm.ppf(p2))

    def gap(rho: float) -> float:
        return _bvn_cdf(h1, h2, rho) - p11

    lo, hi = -0.999, 0.999
    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo * g_hi > 0.0:
        return hi if g_hi < 0.0 else lo
    return float(brentq(gap, lo, hi, xtol=1e-4))


def aligned_outcome_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица ``ячейка × агент`` для корректного выравнивания исходов.

    Строка соответствует одной ячейке эксперимента ``(scenario_id,
    opponent_id, seed)``. Только такое выравнивание делает корреляции между
    агентами осмысленными: сравнивать нужно исходы в одинаковых условиях, а не
    по позиции в таблице.

    Returns
    -------
    pandas.DataFrame
        Индекс — ячейки, столбцы — ``agent_id``, значения — ``breach``.
    """
    cell = (
        df["scenario_id"].astype(str)
        + "|"
        + df["opponent_id"].astype(str)
        + "|"
        + df["seed"].astype(str)
    )
    return df.assign(_cell=cell).pivot_table(
        index="_cell", columns="agent_id", values="breach", aggfunc="first"
    )


def correlation_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    matrix = aligned_outcome_matrix(df)
    agents = list(matrix.columns)
    corr = np.eye(len(agents))
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            rho = tetrachoric_corr(
                matrix.iloc[:, i].to_numpy(), matrix.iloc[:, j].to_numpy()
            )
            corr[i, j] = corr[j, i] = rho
    return corr, agents


def effective_n(corr: np.ndarray) -> float:
    """
    Эффективное число независимых агентов по спектру матрицы корреляций.

    Величина ``N_eff = (sum lambda_k)^2 / sum lambda_k^2``; чем ближе ``N_eff``
    к единице при большом ``N``, тем хуже диверсификация портфеля из-за
    монокультуры моделей.
    """
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues[eigenvalues > 0], 1e-12, None)
    if eigenvalues.size == 0:
        return 0.0
    return float((eigenvalues.sum() ** 2) / (eigenvalues**2).sum())


def rho_by_group(
    corr: np.ndarray, agents: list[str], backbones: dict[str, str]
) -> dict[str, float]:
    within, between = [], []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if backbones.get(agents[i]) == backbones.get(agents[j]):
                within.append(corr[i, j])
            else:
                between.append(corr[i, j])
    return {
        "within": float(np.mean(within)) if within else 0.0,
        "between": float(np.mean(between)) if between else 0.0,
    }


def bootstrap_ci(
    df: pd.DataFrame,
    agents: list[str],
    backbones: dict[str, str],
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """
    Бутстрэп-интервалы для корреляционных метрик монокультуры.

    Пересэмплирует ячейки с возвращением и на каждой выборке заново считает
    матрицу корреляций, ``rho_within``, ``rho_between`` и ``N_eff``. Интервалы
    нужны для честного отчёта: точечные оценки на малых ``K`` шумные.

    Returns
    -------
    dict
        Интервалы ``within_ci``, ``between_ci`` и ``n_eff_ci``.
    """
    rng = np.random.default_rng(seed)
    matrix = aligned_outcome_matrix(df)
    n_cells = len(matrix)
    within_samples, between_samples, n_eff_samples = [], [], []

    for _ in range(n_boot):
        sample = matrix.iloc[rng.integers(0, n_cells, n_cells)]
        corr = np.eye(len(agents))
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                rho = tetrachoric_corr(
                    sample.iloc[:, i].to_numpy(), sample.iloc[:, j].to_numpy()
                )
                corr[i, j] = corr[j, i] = rho
        gap = rho_by_group(corr, agents, backbones)
        within_samples.append(gap["within"])
        between_samples.append(gap["between"])
        n_eff_samples.append(effective_n(corr))

    alpha = (1.0 - level) / 2.0

    def interval(values: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi)

    return {
        "within_ci": interval(within_samples),
        "between_ci": interval(between_samples),
        "n_eff_ci": interval(n_eff_samples),
    }
