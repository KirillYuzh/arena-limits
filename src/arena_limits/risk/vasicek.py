import numpy as np
from scipy.stats import norm


def vasicek_pd(p, rho_b, z):
    """
    Условная вероятность пробоя по модели Vasicek.

    ``PD_i(z) = Phi((Phi^-1(p_i) - sqrt(rho_b) * z) / sqrt(1 - rho_b))``, где
    ``z`` — системный фактор («волна атаки»), общий для бэкбона. Именно общий
    фактор и делает риски внутри бэкбона коррелированными.

    Parameters
    ----------
    p : float or array-like
        Безусловные вероятности пробоя.
    rho_b : float
        Корреляция внутри бэкбона.
    z : array-like
        Реализации системного фактора ``N(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Массив формы ``(len(p), len(z))``.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    rho_b = float(np.clip(rho_b, 0.0, 0.999))
    z = np.asarray(z, dtype=float)
    numerator = norm.ppf(p)[:, None] - np.sqrt(rho_b) * z[None, :]
    return norm.cdf(numerator / np.sqrt(1.0 - rho_b))
