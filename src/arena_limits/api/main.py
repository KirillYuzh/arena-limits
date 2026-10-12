import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from arena_limits import config
from arena_limits.monitor.eprocess import EProcessMonitor
from arena_limits.risk.limits import pair_probability_from_storage, quote_pair
from arena_limits.storage import Storage

app = FastAPI(title="arena-limits")

_storage: Storage | None = None


def storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage(config.DB_PATH)
    return _storage


class QuoteRequest(BaseModel):
    agent_id: str
    opponent_id: str
    requested_amount: float
    rho_b: float = 0.1
    budget: float = 1_000_000.0


class ObserveRequest(BaseModel):
    cluster: str
    y: int
    q: float


@app.post("/guarantees/quote")
def quote(request: QuoteRequest) -> dict:
    """
    Лимит по паре «агент–контрагент» с объяснением
    """
    p_pair = pair_probability_from_storage(
        storage(), request.agent_id, request.opponent_id
    )
    if math.isnan(p_pair):
        raise HTTPException(404, "агент или противник не сертифицирован")
    limit = quote_pair(p_pair, request.rho_b, request.budget, request.requested_amount)
    return {
        "limit": limit,
        "requested": request.requested_amount,
        "pair_breach_prob": p_pair,
        "rho_b": request.rho_b,
        "explanation": {
            "agent_id": request.agent_id,
            "opponent_id": request.opponent_id,
            "budget": request.budget,
        },
    }


@app.post("/guarantees/observe")
def observe(event: ObserveRequest) -> dict:
    """
    Добавить исход сделки в e-процесс кластера; вернуть состояние заморозки
    """
    row = storage().get_eprocess(event.cluster)
    state = {"log_e": row[1]} if row else {}
    monitor = EProcessMonitor.from_state(
        state, delta=config.DEFAULT_DELTA, kappa=config.DEFAULT_KAPPA
    )
    frozen = monitor.update(event.y, event.q)
    storage().upsert_eprocess(event.cluster, monitor.log_e, monitor.threshold, frozen)
    return {"cluster": event.cluster, **monitor.state()}


@app.get("/clusters/{cluster}")
def cluster(cluster: str) -> dict:
    """
    Текущее состояние e-процесса кластера
    """
    row = storage().get_eprocess(cluster)
    if row is None:
        raise HTTPException(404, "кластер не наблюдается")
    return {
        "cluster": cluster,
        "log_e": row[1],
        "threshold": row[2],
        "frozen": bool(row[3]),
    }
