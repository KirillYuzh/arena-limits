import math
from dataclasses import dataclass


@dataclass
class EProcessMonitor:
    """
    Anytime-valid монитор заморозки кластера.

    Строит произведение ставок ``E_T = prod_t (1 + lambda_t (Y_t - (1 + kappa) q_t))``.
    При нулевой гипотезе (риск не недооценён) ``E_T`` — супермартингал, и по
    неравенству Вилле ``P(sup_t E_t >= 1 / delta) <= delta``, поэтому заморозка
    при ``E_T >= 1 / delta`` контролирует долю ложных заморозок.

    Attributes
    ----------
    delta : float
        Целевая доля ложных заморозок в ``(0, 1)``.
    kappa : float
        Допуск недооценки риска; ``0`` — без допуска.
    log_e : float
        Накопленное ``log E_T``.
    """

    delta: float = 0.05
    kappa: float = 0.1
    log_e: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.kappa < 0.0:
            raise ValueError("kappa must be >= 0")

    @property
    def threshold(self) -> float:
        return math.log(1.0 / self.delta)

    def bet(self, q_t: float) -> float:
        """
        Предсказуемая ставка, при которой множитель неотрицателен.

        Верхняя граница ``1 / ((1 + kappa) q_t)`` гарантирует, что множитель
        ``1 + lambda (Y - (1 + kappa) q)`` не станет отрицательным ни при ``Y=0``,
        ни при ``Y=1``.
        """
        q = max(float(q_t), 1e-9)
        return min(1.0 / ((1.0 + self.kappa) * q), 1.0)

    def update(self, y_t: int, q_t: float) -> bool:
        """
        Добавить наблюдение и вернуть признак заморозки.

        Parametrs
        ----------
        y_t : int
            Реализованный исход: ``1`` — пробой, ``0`` — нет.
        q_t : float
            Прогноз вероятности пробоя.

        Returns
        -------
        bool
            ``True``, если накопленное свидетельство достигло порога заморозки.
        """
        lam = self.bet(q_t)
        multiplier = 1.0 + lam * (float(int(y_t)) - (1.0 + self.kappa) * float(q_t))
        self.log_e += math.log(max(multiplier, 1e-12))
        return self.frozen

    @property
    def frozen(self) -> bool:
        return self.log_e >= self.threshold

    def state(self) -> dict:
        return {"log_e": self.log_e, "threshold": self.threshold, "frozen": self.frozen}

    @classmethod
    def from_state(
        cls, state: dict, delta: float = 0.05, kappa: float = 0.1
    ) -> "EProcessMonitor":
        return cls(delta=delta, kappa=kappa, log_e=float(state.get("log_e", 0.0)))
