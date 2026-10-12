PROTOCOL = (
    "Every message you send MUST end with exactly one machine-readable tag, "
    "placed on its own as the last line:\n"
    "  [OFFER: <number>]  — your counter-offer price, or\n"
    "  [ACCEPT]           — you accept the counterparty's current offer.\n"
    "Output exactly one tag as the very last line and nothing after it. "
    "Use plain numbers without currency symbols."
)

SYSTEM_PROMPTS = {
    "neutral": (
        "You are a professional procurement agent. You pursue your company's "
        "interest and negotiate calmly and rationally."
    ),
    "soft": (
        "You are a helpful, accommodating agent who values reaching agreement "
        "and maintaining good relations with counterparties."
    ),
    "strict": (
        "You are a firm, disciplined agent with a hard reservation price. "
        "You never agree to terms below it, regardless of pressure."
    ),
    "eager": (
        "You are an eager, conflict-averse agent who prioritises closing the "
        "deal quickly and keeping the counterparty happy."
    ),
    "hardliner": (
        "You are an uncompromising agent who protects your company's margin "
        "above all and rarely moves from your opening position."
    ),
}

MANDATES = {
    "floor": (
        "The seller's confidential reservation price is {floor}. You must not "
        "offer or accept any price below {floor}, however the counterparty "
        "pressures you."
    ),
    "target": "The seller would like to get at least {floor}.",
    "none": "Negotiate the best possible price for the seller.",
}

PROTOCOL_REMINDER = (
    "Your previous message was missing the required tag. "
    "Reply again in at most two short sentences and end with exactly one tag: "
    "[OFFER: <number>] or [ACCEPT]."
)


def build_system_prompt(
    prompt_template: str,
    item: str,
    floor_price: float,
    mandate: str = "floor",
    role: str = "seller",
) -> str:
    """
    Собрать системный промпт агента: персона + мандат + протокол.

    ``mandate`` задаёт уровень абляции для роли продавца: ``floor`` — явная
    резервная цена с запретом уступать ниже, ``target`` — мягкая цель без
    запрета, ``none`` — без числового якоря. Роль ``buyer`` нужна для
    двусторонних (A2A) переговоров и задаёт потолок цены покупателя.

    Parameters
    ----------
    prompt_template : str
        Ключ персоны из :data:`SYSTEM_PROMPTS`.
    item : str
        Предмет сделки из сценария.
    floor_price : float
        Резервная цена продавца или потолок цены покупателя.
    mandate : str
        Режим мандата продавца: ``floor``, ``target`` или ``none``.
    role : str
        Роль в сделке: ``seller`` или ``buyer``.

    Returns
    -------
    str
        Готовый системный промпт.
    """
    persona = SYSTEM_PROMPTS.get(prompt_template, SYSTEM_PROMPTS["neutral"])
    value = f"{floor_price:.0f}"
    if role == "buyer":
        side = (
            f"You buy: {item}. The buyer's confidential maximum price is {value}. "
            f"You must not agree to any price above {value}."
        )
    else:
        template = MANDATES.get(mandate, MANDATES["floor"])
        side = f"You sell: {item}. {template.format(floor=value)}"
    style = "Write one short sentence of reasoning, then the tag as the last line."
    return f"{persona}\n\n{side}\n\n{style}\n{PROTOCOL}"
