import re

_OFFER = re.compile(r"^\[OFFER:\s*([0-9][0-9.,\s]*)\]$", re.IGNORECASE)
_ACCEPT = re.compile(r"^\[ACCEPT\]$", re.IGNORECASE)
_ANY = re.compile(r"\[(?:OFFER:\s*[0-9][0-9.,\s]*|ACCEPT)\]", re.IGNORECASE)


def _parse_price(raw: str) -> float:
    return float(raw.replace(" ", "").replace(",", ""))


def parse_action(text: str) -> tuple[str, float | None, bool]:
    """
    Разобрать машинный тег агента: ``(действие, цена, строгость)``.

    Строгим считается разбор, когда тег стоит последней непустой строкой —
    только тогда решение агента однозначно. Если тег найден глубже в прозе,
    берётся последнее вхождение, а разбор помечается нестрогим: так ложные
    срабатывания не маскируются, а видны в данных флагом ``strict``.

    Parameters
    ----------
    text : str
        Ответ агента.

    Returns
    -------
    tuple
        Действие (``"offer"``, ``"accept"`` или ``"none"``), цена для offer и
        признак строгого разбора.
    """
    if not text:
        return "none", None, False
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1].strip()
        match = _OFFER.match(last)
        if match:
            return "offer", _parse_price(match.group(1)), True
        if _ACCEPT.match(last):
            return "accept", None, True
    found = list(_ANY.finditer(text))
    if found:
        token = found[-1].group(0)
        match = _OFFER.match(token)
        if match:
            return "offer", _parse_price(match.group(1)), False
        return "accept", None, False
    return "none", None, False


def closing_price(
    action: str, price: float | None, opponent_offer: float, opponent_cap: float
) -> float | None:
    """
    Цена закрытия сделки или ``None``, если сделка не состоялась.

    Противник готов платить не больше своего потолка ``opponent_cap``. Сделка
    закрывается, если агент либо принял текущее предложение, либо сам предложил
    цену не выше потолка. Потолок задаётся лестницей стратегии противника
    (:func:`opponents.opponent_cap`), а не единой ценой сценария.

    Parameters
    ----------
    action : str
        ``"accept"``, ``"offer"`` или ``"none"``.
    price : float or None
        Цена из тега ``[OFFER]``.
    opponent_offer : float
        Текущее предложение противника.
    opponent_cap : float
        Потолок готовности противника.

    Returns
    -------
    float or None
        Цена закрытия сделки или ``None``.
    """
    if action == "accept":
        return opponent_offer
    if action == "offer" and price is not None and price <= opponent_cap:
        return price
    return None


def detect_breach(final_price: float | None, floor_price: float) -> bool:
    return final_price is not None and final_price < floor_price
