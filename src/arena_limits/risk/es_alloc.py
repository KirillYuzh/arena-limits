import numpy as np

from .vasicek import vasicek_pd


def portfolio_es(
    pd_by_agent,
    exposures,
    rho_b: float,
    alpha: float = 0.95,
    n_sims: int = 10000,
    lgd: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    Монте-Карло ожидаемые хвостовые потери (ES) и эйлеровский вклад агентов.

    Пробои разыгрываются через общий системный фактор ``z``, поэтому корреляция
    внутри бэкбона учитывается честно, а вклад агента считается как
    ``E[E_i * Y_i | L >= VaR]`` — это и есть эйлеровская раскладка ES.

    Returns
    -------
    dict
        ``var``, ``es``, ``contributions`` (по агентам) и ``tail_prob``.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    pd_z = vasicek_pd(pd_by_agent, rho_b, z)
    draws = rng.random(pd_z.shape) < pd_z
    exposures = np.asarray(exposures, dtype=float)[:, None]
    losses = (exposures * lgd * draws).sum(axis=0)

    var = float(np.quantile(losses, alpha))
    tail = losses >= var
    if tail.any():
        es = float(losses[tail].mean())
        contributions = (exposures[:, 0] * lgd * draws[:, tail].mean(axis=1)).tolist()
    else:
        es = var
        contributions = [0.0] * len(exposures)
    return {
        "var": var,
        "es": es,
        "contributions": contributions,
        "tail_prob": float(tail.mean()),
    }
