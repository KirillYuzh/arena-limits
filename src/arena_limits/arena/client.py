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


class LiteLLMChatClient:
    """
    Клиент моделей через LiteLLM — напрямую или через прокси.

    Parameters
    ----------
    api_base : str or None
        Адрес LiteLLM-прокси; ``None`` — обращение к провайдеру напрямую.
    api_key : str or None
        Ключ провайдера или прокси.
    ssl_verify : bool or None
        Проверка TLS. GigaChat использует корневой CA, которого нет в certifi,
        поэтому для него нужно ``False``.
    max_tokens : int
        Ограничение длины ответа; должно вмещать финальный тег протокола.
    timeout : float
        Таймаут запроса в секундах.

    Notes
    -----
    ``seed`` принимается для единообразия интерфейса, но облачные модели его не
    гарантируют, поэтому на реальных моделях воспроизводимости нет.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        ssl_verify: bool | None = None,
        max_tokens: int = 256,
        timeout: float = 120.0,
    ):
        from litellm import completion

        self._completion = completion
        self.api_base = api_base
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        seed: int,
        temperature: float = 0.7,
        meta: dict | None = None,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.ssl_verify is not None:
            kwargs["ssl_verify"] = self.ssl_verify
        response = self._completion(**kwargs)
        return response.choices[0].message.content or ""
