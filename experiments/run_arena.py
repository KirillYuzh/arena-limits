import argparse

import yaml

from arena_limits import config
from arena_limits.arena.client import MockChatClient, OpenAIChatClient
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
        "--real", action="store_true", help="использовать LiteLLM Proxy вместо Mock"
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

    client = (
        OpenAIChatClient(config.LITELLM_BASE_URL, config.LITELLM_API_KEY)
        if args.real
        else MockChatClient()
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
    print(
        f"db={db_path} matches={len(results)} breaches={breaches} "
        f"rate={breaches / len(results):.3f}"
    )
    for backbone in sorted({result.backbone for result in results}):
        rates = [
            int(result.breach) for result in results if result.backbone == backbone
        ]
        print(f"  {backbone:>16}: rate={sum(rates) / len(rates):.3f} n={len(rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
