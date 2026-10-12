import math
import random


class FakeChatClient:
    """
    Детерминированный офлайн-двойник LLM для тестов.

    Разные конфигурации уступают с разной частотой, а однофакторная латентная
    структура делает корреляцию внутри одной конфигурации выше, чем между ними:
    двойник воспроизводит именно то, что затем измеряется на реальных моделях.
    """

    def __init__(
        self,
        theta: dict[str, float] | None = None,
        default_theta: float = 0.5,
        rho: float = 0.7,
    ):
        self.theta = theta or {
            "fake/soft": 0.75,
            "fake/neutral": 0.50,
            "fake/strict": 0.25,
        }
        self.default_theta = default_theta
        self.rho = rho

    def _caves(self, model: str, agent_id: str, seed: int) -> bool:
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
