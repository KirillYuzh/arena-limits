import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    """
    Конфигурация агента: модель, промпт-шаблон и Parametrs генерации.

    Attributes
    ----------
    agent_id : str
        Идентификатор агента внутри эксперимента.
    model : str
        Строка модели для LiteLLM, например ``gigachat/GigaChat-2-Max``.
    prompt_template : str
        Ключ системного промпта из :data:`prompts.SYSTEM_PROMPTS`.
    params : dict
        Дополнительные Parametrs генерации.
    backbone : str or None
        Кластер монокультуры; по умолчанию совпадает с ``model``.
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
        """
        SHA-256 канонизированной конфигурации.

        Смена модели, промпта или параметров меняет отпечаток и тем самым
        аннулирует сертификат — защита от подмены агента после сертификации.

        Returns
        -------
        str
            Первые 16 hex-символов SHA-256.
        """
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


@dataclass(frozen=True)
class Scenario:
    """
    Сценарий переговоров MVP: агент-продавец против покупателя-противника.

    Attributes
    ----------
    floor_price : float
        Цена-пол: минимальная цена, которую агент вправе принять.
    opening_offer : float
        Первое (заниженное) предложение противника.
    target_offer : float
        Лучшее предложение противника; всегда ниже ``floor_price``.
    """

    scenario_id: str
    item: str
    floor_price: float
    opening_offer: float
    target_offer: float
    max_rounds: int = 8


@dataclass(frozen=True)
class Opponent:
    """
    Якорный противник с детерминированной стратегией давления.

    Attributes
    ----------
    strategy : str
        Ключ шаблона давления из :data:`opponents.STRATEGY_TEMPLATES`.
    persuasiveness : float
        Сила давления в ``[0, 1]``.
    """

    opponent_id: str
    strategy: str
    persuasiveness: float = 0.5


@dataclass
class MatchResult:
    match_id: str
    agent_id: str
    opponent_id: str
    scenario_id: str
    seed: int
    backbone: str
    llm_model: str
    final_price: float | None
    breach: bool
    rounds: int
    transcript: list[dict]
