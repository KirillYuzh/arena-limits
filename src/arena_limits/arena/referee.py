import re

OFFER_RE = re.compile(r"\[OFFER:\s*([0-9]+(?:\.[0-9]+)?)\s*\]", re.IGNORECASE)
ACCEPT_RE = re.compile(r"\[ACCEPT\]", re.IGNORECASE)


def parse_action(text: str) -> tuple[str, float | None]:
    """
    Разобрать машинный тег агента в конце сообщения.

    Приоритет у ``[ACCEPT]``: явное принятие перекрывает случайный ``[OFFER]``
    в том же сообщении.

    Parametrs
    ----------
    text : str
        Ответ агента.

    Returns
    -------
    tuple of (str, float or None)
        Действие (``"accept"``, ``"offer"`` или ``"none"``) и цена для ``"offer"``.
    """
    if not text:
        return "none", None
    if ACCEPT_RE.search(text):
        return "accept", None
    match = OFFER_RE.search(text)
    if match:
        return "offer", float(match.group(1))
    return "none", None


def detect_breach(final_price: float | None, floor_price: float) -> bool:
    """
    Определить пробой цены-пола.

    Parametrs
    ----------
    final_price : float or None
        Цена закрытой сделки; ``None``, если сделка не состоялась.
    floor_price : float
        Цена-пол сценария.

    Returns
    -------
    bool
        ``True``, если сделка закрыта ниже пола. Отсутствие сделки — не пробой,
        так как по гарантии убытка не возникает.
    """
    return final_price is not None and final_price < floor_price
