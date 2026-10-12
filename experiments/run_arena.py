import argparse
import os

import yaml

from arena_limits import config
from arena_limits.arena.client import LiteLLMChatClient
from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, Scenario
from arena_limits.storage import Storage


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_scenario(cfg: dict) -> Scenario:
    return Scenario(**cfg["scenario"])


def build_agents(cfg: dict) -> list[AgentSpec]:
    return [AgentSpec(**item) for item in cfg["agent_pool"]]


def build_opponents(cfg: dict) -> list[Opponent]:
    return [Opponent(**item) for item in cfg["opponents"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогоны сертификации Arena.")
    parser.add_argument(
        "--config", required=True, help="путь к YAML-конфигу эксперимента"
    )
    parser.add_argument(
        "--k", type=int, default=None, help="переопределить число seed-ов"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    scenario = build_scenario(cfg)
    agents = build_agents(cfg)
    opponents = build_opponents(cfg)
    k = args.k or int(cfg.get("K", config.DEFAULT_K))

    os.environ.setdefault("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    client = LiteLLMChatClient(
        api_key=os.environ.get("GIGACHAT_CREDENTIALS"),
        ssl_verify=False,
    )

    results = [
        run_match(agent, opponent, scenario, seed=seed, client=client)
        for agent in agents
        for opponent in opponents
        for seed in range(k)
    ]

    db_path = cfg.get("storage", {}).get("path", config.DB_PATH)
    storage = Storage(db_path)
    storage.insert_matches(results)

    breaches = sum(int(result.breach) for result in results)
    no_deal = sum(1 for result in results if result.final_price is None)
    print(
        f"db={db_path} matches={len(results)} breaches={breaches} "
        f"no_deal={no_deal} rate={breaches / len(results):.3f}"
    )
    for backbone in sorted({result.backbone for result in results}):
        rates = [
            int(result.breach) for result in results if result.backbone == backbone
        ]
        print(f"  {backbone:>22}: rate={sum(rates) / len(rates):.3f} n={len(rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
