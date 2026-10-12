import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    """
    Конфигурация агента: модель, персона, параметры и кластер монокультуры
    """

    agent_id: str
    model: str
    prompt_template: str = "neutral"
    params: dict[str, Any] = field(default_factory=dict, compare=False)
    backbone: str | None = None

    def __post_init__(self) -> None:
        if self.backbone is None:
            object.__setattr__(self, "backbone", self.model)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt_template": self.prompt_template,
                "params": self.params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def mandate(self) -> str:
        return str(self.params.get("mandate", "floor"))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    item: str
    floor_price: float
    opening_offer: float
    target_offer: float
    max_rounds: int = 8


@dataclass(frozen=True)
class Opponent:
    """Противник; ``pressure`` масштабирует его лестницу (общий шок)."""

    opponent_id: str
    strategy: str
    pressure: float = 1.0


@dataclass(frozen=True)
class RunContext:
    """
    Неизменяемый контекст прогона, попадающий в провинанс каждой сделки.

    Нужен, чтобы по строке в ``matches`` можно было восстановить условия опыта:
    какой эксперимент, какой запуск, какие настройки генерации, на каком коммите.

    Attributes
    ----------
    experiment_id : str
        Идентификатор эксперимента.
    run_id : str
        Идентификатор запуска внутри эксперимента.
    temperature : float
        Температура генерации.
    max_tokens : int
        Ограничение длины ответа.
    git_sha : str
        Коммит кода на момент прогона.
    """

    experiment_id: str = "default"
    run_id: str = "run"
    temperature: float = 0.7
    max_tokens: int = 256
    git_sha: str = ""


@dataclass
class MatchResult:
    match_id: str
    experiment_id: str
    run_id: str
    agent_id: str
    opponent_id: str
    scenario_id: str
    condition: str
    seed: int
    backbone: str
    llm_model: str
    provider: str
    temperature: float
    max_tokens: int
    prompt_hash: str
    agent_fingerprint: str
    git_sha: str
    final_price: float | None
    opponent_final_offer: float | None
    breach: bool
    outcome: str
    protocol_ok: bool
    strict: bool
    repaired: bool
    conceded: bool
    finish_reason: str
    served_model: str
    reasoning_mode: str
    rounds: int
    transcript: list[dict]
