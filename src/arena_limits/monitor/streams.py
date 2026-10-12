import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    cluster: str
    y: int
    q: float
    label: str = ""


class FreezeMonitor(Protocol):
    def update(self, y_t: int, q_t: float) -> bool: ...

    def state(self) -> dict: ...


class OutcomeStream(ABC):
    """
    Источник наблюдений ``(y_t, q_t)``, сгруппированных по кластерам.

    Единый интерфейс закрывает три задачи MVP одним кодом: оценка, демо и
    стресс-тесты используют один и тот же монитор, меняется только источник.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Observation]: ...

    def by_cluster(self) -> dict[str, list[Observation]]:
        out: dict[str, list[Observation]] = {}
        for observation in self:
            out.setdefault(observation.cluster, []).append(observation)
        return out


class ReplayStream(OutcomeStream):
    """
    Реплей реальных прогонов Arena — для оценки на фактических данных.

    Parametrs
    ----------
    records : iterable of dict
        Записи вида ``{"cluster": str, "y": 0|1, "q": float, "label": str}``.
    """

    def __init__(self, records: Iterable[dict]):
        self.records = list(records)

    def __iter__(self) -> Iterator[Observation]:
        for record in self.records:
            yield Observation(
                cluster=str(record["cluster"]),
                y=int(record["y"]),
                q=float(record["q"]),
                label=str(record.get("label", "")),
            )


class ScriptedStream(OutcomeStream):
    """
    Скриптованный поток для демо: заморозка видна в реальном времени
    """

    def __init__(
        self,
        cluster: str,
        q: float,
        script: Sequence[int],
        label_prefix: str = "deal",
    ):
        self.cluster = cluster
        self.q = float(q)
        self.script = [int(x) for x in script]
        self.label_prefix = label_prefix

    def __iter__(self) -> Iterator[Observation]:
        for index, y in enumerate(self.script):
            yield Observation(self.cluster, y, self.q, f"{self.label_prefix}-{index}")


class SyntheticStream(OutcomeStream):
    """
    Синтетический поток для стресс-тестов: Bernoulli с заданной частотой
    """

    def __init__(
        self,
        cluster: str,
        q: float,
        n: int,
        breach_rate: float,
        seed: int = 0,
    ):
        self.cluster = cluster
        self.q = float(q)
        self.n = int(n)
        self.breach_rate = float(breach_rate)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Observation]:
        rng = random.Random(self.seed)
        for index in range(self.n):
            y = 1 if rng.random() < self.breach_rate else 0
            yield Observation(self.cluster, y, self.q, f"sim-{index}")


def run_monitor(
    stream: OutcomeStream,
    monitor_factory: Callable[[], FreezeMonitor],
) -> dict[str, dict]:
    """
    Прогнать поток через независимые мониторы, по одному на кластер.

    Parametrs
    ----------
    stream : OutcomeStream
        Источник наблюдений.
    monitor_factory : callable
        Returns новый монитор с методами ``update`` и ``state``.

    Returns
    -------
    dict
        Последнее состояние монитора по каждому кластеру.
    """
    monitors: dict[str, FreezeMonitor] = {}
    result: dict[str, dict] = {}
    for observation in stream:
        monitor = monitors.get(observation.cluster)
        if monitor is None:
            monitor = monitors[observation.cluster] = monitor_factory()
        monitor.update(observation.y, observation.q)
        result[observation.cluster] = monitor.state()
    return result
