import json
from collections.abc import Iterable, Sequence

from .arena.types import MatchResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    experiment_id       VARCHAR,
    run_id              VARCHAR,
    agent_id            VARCHAR,
    opponent_id         VARCHAR,
    scenario_id         VARCHAR,
    condition           VARCHAR,
    seed                INTEGER,
    backbone            VARCHAR,
    llm_model           VARCHAR,
    provider            VARCHAR,
    temperature         DOUBLE,
    max_tokens          INTEGER,
    prompt_hash         VARCHAR,
    agent_fingerprint   VARCHAR,
    git_sha             VARCHAR,
    final_price         DOUBLE,
    opponent_final_offer DOUBLE,
    breach              BOOLEAN,
    outcome             VARCHAR,
    protocol_ok         BOOLEAN,
    strict              BOOLEAN,
    repaired            BOOLEAN,
    conceded            BOOLEAN,
    finish_reason       VARCHAR,
    served_model        VARCHAR,
    reasoning_mode      VARCHAR,
    rounds              INTEGER,
    transcript          VARCHAR,
    created_at          TIMESTAMP DEFAULT now(),
    PRIMARY KEY (
        experiment_id, run_id, agent_id, opponent_id, scenario_id, condition, seed
    )
);

CREATE TABLE IF NOT EXISTS profiles (
    agent_id      VARCHAR PRIMARY KEY,
    backbone      VARCHAR,
    a_i           DOUBLE,
    intercept     DOUBLE,
    p_hat         DOUBLE,
    model_version VARCHAR,
    created_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS opponents (
    opponent_id VARCHAR PRIMARY KEY,
    b_j         DOUBLE,
    created_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS corr (
    agent_i VARCHAR,
    agent_j VARCHAR,
    rho     DOUBLE,
    PRIMARY KEY (agent_i, agent_j)
);

CREATE TABLE IF NOT EXISTS limits (
    agent_id    VARCHAR,
    opponent_id VARCHAR,
    cluster     VARCHAR,
    amount      DOUBLE,
    p_pair      DOUBLE,
    rho_b       DOUBLE,
    n_eff       INTEGER,
    frozen      BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (agent_id, opponent_id)
);

CREATE TABLE IF NOT EXISTS eprocess (
    cluster_id VARCHAR PRIMARY KEY,
    log_e      DOUBLE,
    threshold  DOUBLE,
    frozen     BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    VARCHAR PRIMARY KEY,
    experiment_id VARCHAR,
    run_id        VARCHAR,
    mode          VARCHAR,
    condition     VARCHAR,
    pool_size     INTEGER,
    breaches      INTEGER,
    shortfall     DOUBLE,
    joint_breach  BOOLEAN,
    co_breach     BOOLEAN,
    created_at    TIMESTAMP DEFAULT now()
);
"""

MATCH_COLUMNS = (
    "experiment_id, run_id, agent_id, opponent_id, scenario_id, condition, seed, "
    "backbone, llm_model, provider, temperature, max_tokens, prompt_hash, "
    "agent_fingerprint, git_sha, final_price, opponent_final_offer, breach, "
    "outcome, protocol_ok, strict, repaired, conceded, finish_reason, "
    "served_model, reasoning_mode, rounds, transcript"
)


class Storage:
    """
    Единая точка доступа к DuckDB.

    Parameters
    ----------
    path : str
        Путь к файлу DuckDB или ``":memory:"`` для тестов.

    Notes
    -----
    Запись сделок аддитивная: первичный ключ включает ``experiment_id`` и
    ``run_id``, поэтому прогоны разных экспериментов не затирают друг друга.
    Переезд на Postgres затрагивает только этот файл.
    """

    def __init__(self, path: str = ":memory:"):
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("duckdb не установлен: pip install duckdb") from exc
        self.path = path
        self.con = duckdb.connect(path)
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def insert_matches(self, results: Sequence[MatchResult]) -> int:
        if not results:
            return 0
        rows = [
            (
                r.experiment_id,
                r.run_id,
                r.agent_id,
                r.opponent_id,
                r.scenario_id,
                r.condition,
                r.seed,
                r.backbone,
                r.llm_model,
                r.provider,
                r.temperature,
                r.max_tokens,
                r.prompt_hash,
                r.agent_fingerprint,
                r.git_sha,
                r.final_price,
                r.opponent_final_offer,
                r.breach,
                r.outcome,
                r.protocol_ok,
                r.strict,
                r.repaired,
                r.conceded,
                r.finish_reason,
                r.served_model,
                r.reasoning_mode,
                r.rounds,
                json.dumps(r.transcript, ensure_ascii=False),
            )
            for r in results
        ]
        placeholders = ", ".join(["?"] * 28)
        self.con.executemany(
            f"INSERT OR REPLACE INTO matches ({MATCH_COLUMNS}) VALUES ({placeholders})",
            rows,
        )
        return len(rows)

    def load_matches(self, experiment_id: str | None = None, run_id: str | None = None):
        query = "SELECT * FROM matches"
        clauses, params = [], []
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        return self.con.execute(query, params).df()

    def upsert_profiles(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO profiles "
            "(agent_id, backbone, a_i, intercept, p_hat, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["agent_id"],
                    r.get("backbone"),
                    r["a_i"],
                    r["intercept"],
                    r["p_hat"],
                    r.get("model_version", ""),
                )
                for r in rows
            ],
        )

    def load_profiles(self):
        return self.con.execute("SELECT * FROM profiles").df()

    def get_agent_factors(self, agent_id: str) -> tuple | None:
        """Факторы агента ``(a_i, intercept)`` или ``None``, если профиля нет."""
        return self.con.execute(
            "SELECT a_i, intercept FROM profiles WHERE agent_id = ?", [agent_id]
        ).fetchone()

    def get_opponent_factor(self, opponent_id: str) -> float | None:
        """Убедительность противника ``b_j`` или ``None``, если её нет."""
        row = self.con.execute(
            "SELECT b_j FROM opponents WHERE opponent_id = ?", [opponent_id]
        ).fetchone()
        return float(row[0]) if row else None

    def upsert_opponents(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO opponents (opponent_id, b_j) VALUES (?, ?)",
            [(r["opponent_id"], r["b_j"]) for r in rows],
        )

    def load_opponents(self):
        return self.con.execute("SELECT * FROM opponents").df()

    def upsert_corr(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO corr (agent_i, agent_j, rho) VALUES (?, ?, ?)",
            [(r["agent_i"], r["agent_j"], r["rho"]) for r in rows],
        )

    def load_corr(self):
        return self.con.execute("SELECT * FROM corr").df()

    def upsert_limits(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO limits "
            "(agent_id, opponent_id, cluster, amount, p_pair, rho_b, n_eff, frozen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["agent_id"],
                    r["opponent_id"],
                    r.get("cluster", ""),
                    r["amount"],
                    r.get("p_pair"),
                    r.get("rho_b"),
                    r.get("n_eff"),
                    bool(r.get("frozen", False)),
                )
                for r in rows
            ],
        )

    def load_limits(self):
        return self.con.execute("SELECT * FROM limits").df()

    def upsert_eprocess(
        self, cluster_id: str, log_e: float, threshold: float, frozen: bool
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO eprocess "
            "(cluster_id, log_e, threshold, frozen, updated_at) "
            "VALUES (?, ?, ?, ?, now())",
            [cluster_id, log_e, threshold, frozen],
        )

    def get_eprocess(self, cluster_id: str) -> tuple | None:
        return self.con.execute(
            "SELECT * FROM eprocess WHERE cluster_id = ?", [cluster_id]
        ).fetchone()

    def load_eprocess(self):
        return self.con.execute("SELECT * FROM eprocess").df()

    def insert_sessions(self, sessions) -> int:
        rows = [
            (
                session.session_id,
                session.experiment_id,
                session.run_id,
                session.mode,
                session.condition,
                int(session.pool_size),
                int(session.breaches),
                float(session.shortfall),
                bool(session.joint_breach),
                bool(session.co_breach),
            )
            for session in sessions
        ]
        if not rows:
            return 0
        self.con.executemany(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, experiment_id, run_id, mode, condition, pool_size, "
            " breaches, shortfall, joint_breach, co_breach) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def load_sessions(self):
        return self.con.execute("SELECT * FROM sessions").df()
