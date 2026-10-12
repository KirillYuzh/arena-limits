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
    сопряжённости 2×2. При вырожденных маргиналах (вектор без вариации) или
    слишком коротком векторе возвращает ``nan``, а не ноль: вырождение должно
    быть видно в данных, а не молча превращаться в «независимость».

    Parameters
    ----------
    y1, y2 : array-like
        Бинарные векторы одинаковой длины; NaN игнорируются попарно.

    Returns
    -------
    float
        Оценка в ``[-0.999, 0.999]`` или ``nan``, если она не определена.
    """
    a = np.asarray(y1, dtype=float)
    b = np.asarray(y2, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 4:
        return float("nan")

    p1, p2 = a.mean(), b.mean()
    if p1 <= 0.0 or p1 >= 1.0 or p2 <= 0.0 or p2 >= 1.0:
        return float("nan")

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

    Строка — одно условие эксперимента ``(scenario_id, opponent_id, condition)``.
    Только такое выравнивание делает корреляции осмысленными: сравнивать нужно
    исходы в одинаковых условиях, а не по позиции в таблице.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.

    Returns
    -------
    pandas.DataFrame
        Индекс — условия, столбцы — агенты, значения — пробой.
    """
    condition = df["condition"].astype(str) if "condition" in df.columns else df["seed"].astype(str)
    cell = df["scenario_id"].astype(str) + "|" + df["opponent_id"].astype(str) + "|" + condition
    return df.assign(_cell=cell).pivot_table(
        index="_cell", columns="agent_id", values="breach", aggfunc="first"
    )


def correlation_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    matrix = aligned_outcome_matrix(df)
    agents = list(matrix.columns)
    corr = np.eye(len(agents))
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            rho = tetrachoric_corr(matrix.iloc[:, i].to_numpy(), matrix.iloc[:, j].to_numpy())
            corr[i, j] = corr[j, i] = rho
    return corr, agents


def degenerate_pairs(corr: np.ndarray) -> int:
    return int(np.isnan(corr).sum() // 2)


def effective_n(corr: np.ndarray) -> float:
    """
    Эффективное число независимых агентов по спектру матрицы корреляций.

    Величина ``N_eff = (sum lambda_k)^2 / sum lambda_k^2``. Неопределённые
    (``nan``) пары трактуются как нулевая корреляция — это консервативно, и их
    число отдельно сообщает :func:`degenerate_pairs`.

    Parameters
    ----------
    corr : numpy.ndarray
        Матрица корреляций агентов.

    Returns
    -------
    float
        Эффективное число независимых агентов.
    """
    clean = np.nan_to_num(corr, nan=0.0)
    eigenvalues = np.linalg.eigvalsh(clean)
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
            rho = corr[i, j]
            if np.isnan(rho):
                continue
            if backbones.get(agents[i]) == backbones.get(agents[j]):
                within.append(rho)
            else:
                between.append(rho)
    return {
        "within": float(np.mean(within)) if within else float("nan"),
        "between": float(np.mean(between)) if between else float("nan"),
    }


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """
    Интервал Уилсона для доли ``k / n`` — честная оценка на малых выборках
    """
    if n == 0:
        return (0.0, 1.0)
    z = float(norm.ppf(1 - (1 - level) / 2))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(
    df: pd.DataFrame,
    agents: list[str],
    backbones: dict[str, str],
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """
    Бутстрэп-интервалы для ``rho_within``, ``rho_between`` и ``N_eff``.

    Пересэмплирует условия с возвращением. Интервалы нужны для честного отчёта:
    точечные оценки на малых ``K`` шумные.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.
    agents : list of str
        Агенты в порядке строк матрицы корреляций.
    backbones : dict
        Соответствие агент → бэкбон.
    n_boot : int
        Число бутстрэп-выборок.
    seed : int
        Seed генератора.
    level : float
        Уровень доверия.

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
                corr[i, j] = corr[j, i] = tetrachoric_corr(
                    sample.iloc[:, i].to_numpy(), sample.iloc[:, j].to_numpy()
                )
        gap = rho_by_group(corr, agents, backbones)
        within_samples.append(gap["within"])
        between_samples.append(gap["between"])
        n_eff_samples.append(effective_n(corr))

    alpha = (1.0 - level) / 2.0

    def interval(values: list[float]) -> tuple[float, float]:
        finite = [v for v in values if not np.isnan(v)]
        if not finite:
            return (float("nan"), float("nan"))
        lo, hi = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi)

    return {
        "within_ci": interval(within_samples),
        "between_ci": interval(between_samples),
        "n_eff_ci": interval(n_eff_samples),
    }


def within_backbone_variance(
    profiles: pd.DataFrame,
    backbone_col: str = "backbone",
    value_col: str = "p_hat",
) -> dict[str, dict]:
    """
    Разброс риска внутри бэкбона — основная метрика монокультуры.

    Совместность пробоев по ячейкам у реальных LLM не идентифицируется: агенты
    одной модели — независимые сэмплы. Монокультура проявляется как близость
    уровней риска: если персоны на одном бэкбоне дают почти одинаковый ``p_hat``,
    диверсификации внутри бэкбона нет.

    Returns
    -------
    dict
        По каждому бэкбону: ``std``, ``range`` и ``n_agents``.
    """
    out: dict[str, dict] = {}
    for backbone, group in profiles.groupby(backbone_col):
        values = group[value_col].astype(float)
        out[str(backbone)] = {
            "std": float(values.std(ddof=0)),
            "range": float(values.max() - values.min()),
            "n_agents": int(values.size),
        }
    return out


def between_backbone_gap(
    profiles: pd.DataFrame,
    backbone_col: str = "backbone",
    value_col: str = "p_hat",
) -> float:
    """Разница средних уровней риска между бэкбонами."""
    means = profiles.groupby(backbone_col)[value_col].mean()
    return float(means.max() - means.min())


def model_vs_persona_effect(
    profiles: pd.DataFrame,
    backbone_col: str = "backbone",
    value_col: str = "p_hat",
) -> dict:
    """
    Во сколько раз модель объясняет риск сильнее персоны.

    Сравнивает дисперсию средних по бэкбонам с дисперсией внутри бэкбона и
    проверяет значимость различия бэкбонов однофакторным ANOVA. Отношение много
    больше единицы и малое ``anova_p`` означают, что уровень риска задаёт модель.

    Returns
    -------
    dict
        ``between_var``, ``within_var``, ``ratio``, ``anova_f``, ``anova_p``.
    """
    from scipy.stats import f_oneway

    groups = [
        group[value_col].astype(float).to_numpy() for _, group in profiles.groupby(backbone_col)
    ]
    backbone_means = profiles.groupby(backbone_col)[value_col].mean()
    within_vars = profiles.groupby(backbone_col)[value_col].var()
    between_var = float(backbone_means.var(ddof=0))
    within_var = float(within_vars.mean())
    f_stat, p_value = f_oneway(*groups)
    return {
        "between_var": between_var,
        "within_var": within_var,
        "ratio": float(between_var / within_var) if within_var > 0 else float("inf"),
        "anova_f": float(f_stat),
        "anova_p": float(p_value),
    }
