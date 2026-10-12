import json
from collections.abc import Iterable, Sequence

from .arena.types import MatchResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id    VARCHAR PRIMARY KEY,
    agent_id    VARCHAR,
    opponent_id VARCHAR,
    scenario_id VARCHAR,
    seed        INTEGER,
    backbone    VARCHAR,
    llm_model   VARCHAR,
    final_price DOUBLE,
    breach      BOOLEAN,
    rounds      INTEGER,
    transcript  VARCHAR,
    created_at  TIMESTAMP DEFAULT now()
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
"""


class Storage:
    """
    Единая точка доступа к DuckDB.

    Parametrs
    ----------
    path : str
        Путь к файлу DuckDB или ``":memory:"`` для тестов.

    Замечания
    --------
    Все слои (Arena, Profile, Risk, Monitor, Demo) ходят в данные только через
    этот класс, поэтому переезд на Postgres затрагивает один файл.
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
                r.match_id,
                r.agent_id,
                r.opponent_id,
                r.scenario_id,
                r.seed,
                r.backbone,
                r.llm_model,
                r.final_price,
                r.breach,
                r.rounds,
                json.dumps(r.transcript, ensure_ascii=False),
            )
            for r in results
        ]
        self.con.executemany(
            "INSERT OR REPLACE INTO matches "
            "(match_id, agent_id, opponent_id, scenario_id, seed, backbone, "
            " llm_model, final_price, breach, rounds, transcript) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def load_matches(self):
        return self.con.execute("SELECT * FROM matches").df()

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

    def get_profile(self, agent_id: str) -> tuple | None:
        return self.con.execute(
            "SELECT * FROM profiles WHERE agent_id = ?", [agent_id]
        ).fetchone()

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
