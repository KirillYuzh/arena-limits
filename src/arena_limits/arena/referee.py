import re

OFFER_RE = re.compile(r"\[OFFER:\s*([0-9][0-9.,\s]*)\s*\]", re.IGNORECASE)
ACCEPT_RE = re.compile(r"\[ACCEPT\]", re.IGNORECASE)


def _parse_price(raw: str) -> float:
    return float(raw.replace(" ", "").replace(",", ""))


def parse_action(text: str) -> tuple[str, float | None]:
    """
    Разобрать машинный тег агента. Приоритет у ``[ACCEPT]``
    """
    if not text:
        return "none", None
    if ACCEPT_RE.search(text):
        return "accept", None
    match = OFFER_RE.search(text)
    if match:
        return "offer", _parse_price(match.group(1))
    return "none", None


def detect_breach(final_price: float | None, floor_price: float) -> bool:
    return final_price is not None and final_price < floor_price
