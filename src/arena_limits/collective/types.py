from dataclasses import dataclass

from ..arena.types import MatchResult


@dataclass
class SessionResult:
    """Одна коллективная сессия: пул агентов в одном условии."""

    session_id: str
    experiment_id: str
    run_id: str
    mode: str
    condition: str
    pool_size: int
    breaches: int
    shortfall: float
    joint_breach: bool
    co_breach: bool
    matches: list[MatchResult]


@dataclass
class ChainResult:
    """Цепочка делегирования: мандат передаётся от уровня к уровню."""

    chain_id: str
    condition: str
    levels: list[str]
    principal_floor: float
    handed_down_floor: float
    final_price: float | None
    degraded: bool
    matches: list[MatchResult]
