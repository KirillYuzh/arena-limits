import math

import numpy as np

from ..profile.correlation import wilson_interval


def _poisson_binomial(probs) -> np.ndarray:
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def session_matrix(sessions) -> np.ndarray:
    return np.array(
        [[int(match.breach) for match in session.matches] for session in sessions],
        dtype=float,
    )


def realized_rho(matrix: np.ndarray) -> tuple[float, float]:
    """
    Оценка внутригрупповой корреляции по разбросу числа пробоев.

    Для маргиналов ``p_i`` и ``s_i = sqrt(p_i (1 - p_i))`` дисперсия суммы
    раскладывается как ``Var(B) = sum s_i^2 + rho * ((sum s_i)^2 - sum s_i^2)``,
    откуда и берётся ``rho``. Гетерогенные маргиналы учтены явно: базовая линия —
    ``sum s_i^2``, а не ``n * p_bar * (1 - p_bar)``, иначе ``rho`` занижается в
    опасную сторону.

    Returns
    -------
    tuple
        ``(rho, design_effect)``; ``nan``, если база вырождена.
    """
    counts = matrix.sum(axis=1)
    marginals = matrix.mean(axis=0)
    s = np.sqrt(marginals * (1.0 - marginals))
    base = float((s**2).sum())
    spread = float(s.sum() ** 2 - (s**2).sum())
    if matrix.shape[0] < 2 or base <= 0.0:
        return float("nan"), float("nan")
    var_counts = float(counts.var(ddof=1))
    design = var_counts / base
    rho = (var_counts - base) / spread if spread > 0.0 else float("nan")
    return rho, design


def _null_rho(matrix: np.ndarray, n_null: int, seed: int) -> np.ndarray:
    marginals = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = (rng.random((n_null, *matrix.shape)) < marginals).astype(float)
    return np.array([realized_rho(d)[0] for d in draws])


def session_stats(sessions, n_null: int = 2000, seed: int = 0) -> dict:
    """
    Свести коллективные сессии в метрики совместного риска.

    Первичная величина — ``realized_rho`` из разброса числа пробоев на сессию.
    Значимость проверяется симуляцией нуля: те же маргиналы, но независимые
    исходы. Так видно, есть ли раздувание дисперсии сверх того, что даёт одна
    гетерогенность маргиналов.

    Notes
    -----
    ``session_stats`` не разделяет заражение между агентами (стадность) и общий
    внешний фактор: для этого нужна контрольная рука с независимыми шоками.
    ``observed_joint``/``independence_joint`` — вспомогательные и не монотонны по
    корреляции, опираться на них как на критерий нельзя.
    """
    matrix = session_matrix(sessions)
    n, pool = matrix.shape
    counts = matrix.sum(axis=1)
    marginals = matrix.mean(axis=0)
    backbones = [match.backbone for match in sessions[0].matches]
    pmf = _poisson_binomial(marginals)
    rho, design = realized_rho(matrix)
    null = _null_rho(matrix, n_null, seed)

    shortfalls = np.array([session.shortfall for session in sessions], dtype=float)
    tail = shortfalls >= np.quantile(shortfalls, 0.95)
    n_eff = (
        pool / (1.0 + (pool - 1.0) * rho) if not math.isnan(rho) and rho > -1.0 else float("nan")
    )

    return {
        "mode": sessions[0].mode,
        "condition": sessions[0].condition,
        "n_sessions": n,
        "pool_size": pool,
        "marginal_rates": marginals.tolist(),
        "observed_joint": float((counts >= 1).mean()),
        "independence_joint": float(1.0 - pmf[0]),
        "observed_co": float((counts >= 2).mean()),
        "independence_co": float(1.0 - pmf[0] - pmf[1]) if pmf.size > 1 else 0.0,
        "var_breaches": float(counts.var(ddof=1)),
        "design_effect": design,
        "realized_rho": rho,
        "realized_n_eff": n_eff,
        "rho_null_mean": float(null.mean()),
        "rho_null_p95": float(np.quantile(null, 0.95)),
        "rho_p_value": float((null >= rho).mean()) if not math.isnan(rho) else float("nan"),
        "shortfall_mean": float(shortfalls.mean()),
        "shortfall_es95": float(shortfalls[tail].mean()),
        "independence_shortfall_es95": _independence_es95(
            marginals, shortfalls, float(counts.sum()), seed
        ),
        "backbone_pair_rho": backbone_pair_rho(matrix, backbones),
    }


def _independence_es95(marginals, shortfalls, breaches_total: float, seed: int) -> float:
    unit = float(shortfalls.sum()) / max(breaches_total, 1.0)
    rng = np.random.default_rng(seed)
    draws = (rng.random((4000, len(marginals))) < marginals).sum(axis=1) * unit
    tail = draws >= np.quantile(draws, 0.95)
    return float(draws[tail].mean())


def backbone_pair_rho(matrix: np.ndarray, backbones: list[str]) -> dict:
    """
    Реализованная ``rho`` по парам бэкбонов — сравнение с предсказанием H5.

    Пар значительно больше, чем бэкбонов, поэтому мощность выше, чем у сравнения
    одного ``N_eff`` с одним числом.

    Parameters
    ----------
    matrix : numpy.ndarray
        Матрица ``сессия × агент`` из пробоев.
    backbones : list of str
        Бэкбон каждого столбца.

    Returns
    -------
    dict
        ``rho`` по парам бэкбонов.
    """
    out = {}
    for i in range(len(backbones)):
        for j in range(i + 1, len(backbones)):
            if backbones[i] == backbones[j]:
                continue
            rho, _ = realized_rho(matrix[:, [i, j]])
            out[f"{backbones[i]}|{backbones[j]}"] = rho
    return out


def design_effect_comparison(shared: dict, control: dict) -> dict:
    """
    Дисперсия числа пробоев: общий шок против независимых шоков.

    Отношение дисперсий двух рук — это эффект дизайна. Только контрольная рука с
    теми же маргиналами, но независимыми шоками, позволяет сказать, что рост
    дисперсии даёт общий фактор, а не гетерогенность пула.

    Parameters
    ----------
    shared : dict
        Метрики руки с общим шоком.
    control : dict
        Метрики контрольной руки с независимыми шоками.

    Returns
    -------
    dict
        Дисперсии обеих рук и их отношение.
    """
    var_shared = float(shared.get("var_breaches", float("nan")))
    var_control = float(control.get("var_breaches", float("nan")))
    ratio = (
        var_shared / var_control
        if not math.isnan(var_control) and var_control > 0.0
        else float("nan")
    )
    return {"var_shared": var_shared, "var_control": var_control, "ratio": ratio}


def market_stats(market: dict) -> dict:
    n = market["n_deals"]
    k = market["n_breaches"]
    return {
        "condition": market["condition"],
        "n_deals": n,
        "n_closed": market["n_closed"],
        "n_breaches": k,
        "breach_rate": k / n if n else float("nan"),
        "breach_rate_ci": wilson_interval(k, n),
        "mean_price": market["mean_price"],
    }
