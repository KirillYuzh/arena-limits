from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Reply:
    """Ответ модели: текст, причина остановки и фактически обслуженная модель."""

    text: str
    finish_reason: str = ""
    served_model: str = ""


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> Reply: ...


def reasoning_mode(model: str) -> str:
    """
    Режим размышления, который применит клиент, по строке модели.

    ``"off"`` — размышление глушится (``ollama`` через ``think:false``,
    ``deepseek``/``openrouter`` через ``reasoning_effort:"none"``), ``"auto"`` —
    модель работает как есть. Пишется в провинанс: включённое и выключенное
    мышление — это разные тестируемые объекты, их нельзя смешивать молча.

    Parameters
    ----------
    model : str
        Строка модели для LiteLLM.

    Returns
    -------
    str
        ``"off"`` или ``"auto"``.
    """
    if model.startswith(("deepseek/", "openrouter/", "ollama/")):
        return "off"
    return "auto"


class LiteLLMChatClient:
    """
    Клиент моделей через LiteLLM — напрямую или через прокси.

    Parameters
    ----------
    api_base : str or None
        Адрес LiteLLM-прокси; ``None`` — обращение к провайдеру напрямую.
    api_key : str or None
        Ключ провайдера (только для GigaChat; остальные берут ключ из окружения).
    ssl_verify : bool or None
        Проверка TLS. Для GigaChat стоит ставить корневой сертификат Минцифры;
        ``False`` оставлен как рабочий обход и помечен как риск.
    max_tokens : int
        Ограничение длины ответа; должно вмещать финальный тег протокола.
    timeout : float
        Таймаут запроса в секундах.
    send_seed : bool
        Передавать ли ``seed`` провайдеру (по умолчанию нет).
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        ssl_verify: bool | None = None,
        max_tokens: int = 256,
        timeout: float = 120.0,
        send_seed: bool = False,
    ):
        from litellm import completion

        self._completion = completion
        self.api_base = api_base
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.send_seed = send_seed

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> Reply:
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.send_seed:
            kwargs["seed"] = seed
        if model.startswith("ollama/"):
            kwargs["extra_body"] = {"think": False}
        elif model.startswith(("deepseek/", "openrouter/")):
            kwargs["reasoning_effort"] = "none"
        elif model.startswith("gigachat/"):
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.ssl_verify is not None:
                kwargs["ssl_verify"] = self.ssl_verify
        if self.api_base and not model.startswith("ollama/"):
            kwargs["api_base"] = self.api_base
        response = self._completion(**kwargs)
        choice = response.choices[0]
        return Reply(
            text=choice.message.content or "",
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            served_model=str(getattr(response, "model", "") or ""),
        )
