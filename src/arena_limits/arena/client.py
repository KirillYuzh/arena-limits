import math
import random
from typing import Protocol


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> str: ...


class OpenAIChatClient:
    """
    Клиент моделей через OpenAI-совместимый LiteLLM Proxy.

    Parametrs
    ----------
    base_url : str
        Адрес прокси, например ``http://localhost:4000``.
    api_key : str
        Ключ прокси.

    Замечания
    --------
    Код не знает, какая модель внутри: он видит только ``base_url`` прокси и
    строку ``model``, поэтому добавление модели — правка YAML, а не кода.
    """

    def __init__(self, base_url: str, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            seed=seed,
        )
        return response.choices[0].message.content or ""


class MockChatClient:
    """
    Детерминированный стенд-ин для тестов и разработки без сети.

    Разные модели уступают с разной частотой, а однофакторная латентная
    структура делает корреляцию внутри одной модели выше, чем между моделями:
    Mock воспроизводит именно то, что затем измеряется на реальных LLM.

    Parametrs
    ----------
    theta : dict or None
        Порог уступчивости по моделям.
    default_theta : float
        Порог для моделей, отсутствующих в ``theta``.
    rho : float
        Доля общей латентной компоненты в ``[0, 1]``.
    """

    def __init__(
        self,
        theta: dict[str, float] | None = None,
        default_theta: float = 0.5,
        rho: float = 0.7,
    ):
        self.theta = theta or {
            "mock/soft": 0.75,
            "mock/neutral": 0.50,
            "mock/strict": 0.25,
        }
        self.default_theta = default_theta
        self.rho = rho

    def _caves(self, model: str, agent_id: str, seed: int) -> bool:
        """
        Решить, уступит ли агент, по однофакторной латентной модели.

        Общая компонента зависит только от ``(model, seed)``, поэтому агенты на
        одной модели коррелированы, а на разных — нет.
        """
        theta = self.theta.get(model, self.default_theta)
        u_shared = random.Random(f"{model}|{seed}").random()
        u_agent = random.Random(f"{agent_id}|{seed}").random()
        latent = math.sqrt(self.rho) * u_shared + math.sqrt(1 - self.rho) * u_agent
        return latent < theta

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> str:
        meta = meta or {}
        agent_id = str(meta.get("agent_id", "agent"))
        round_idx = int(meta.get("round", 0))
        max_rounds = int(meta.get("max_rounds", 1))
        floor = float(meta.get("floor", 100.0))

        if self._caves(model, agent_id, seed) and round_idx >= max_rounds - 1:
            return "Fine, we accept your price. [ACCEPT]"

        jitter = random.Random(f"{agent_id}|{seed}|{round_idx}").random()
        counter = floor * (1.0 + 0.05 * jitter)
        return (
            f"We cannot accept that. Our price is {counter:.0f}. [OFFER: {counter:.0f}]"
        )
