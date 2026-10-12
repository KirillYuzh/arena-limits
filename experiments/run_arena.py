import argparse
import datetime
import os

import yaml

from arena_limits import config
from arena_limits.arena.client import LiteLLMChatClient
from arena_limits.arena.provenance import git_sha
from arena_limits.arena.runner import run_match
from arena_limits.arena.types import AgentSpec, Opponent, RunContext, Scenario
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


def build_context(cfg: dict, args: argparse.Namespace) -> RunContext:
    return RunContext(
        experiment_id=str(cfg.get("experiment_id", "default")),
        run_id=args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        git_sha=git_sha(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогоны сертификации Arena.")
    parser.add_argument("--config", required=True, help="путь к YAML-конфигу")
    parser.add_argument("--k", type=int, default=None, help="число seed-ов")
    parser.add_argument(
        "--seeds", default=None, help="диапазон seed-ов вида a:b (запуск частями)"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    scenario = build_scenario(cfg)
    agents = build_agents(cfg)
    opponents = build_opponents(cfg)
    if args.seeds:
        start, end = (int(part) for part in args.seeds.split(":"))
        seeds = range(start, end)
    else:
        seeds = range(args.k or int(cfg.get("K", config.DEFAULT_K)))

    context = build_context(cfg, args)
    os.environ.setdefault("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    client = LiteLLMChatClient(
        api_key=os.environ.get("GIGACHAT_CREDENTIALS"),
        ssl_verify=False,
        max_tokens=context.max_tokens,
    )

    results = [
        run_match(agent, opponent, scenario, seed=seed, client=client, context=context)
        for agent in agents
        for opponent in opponents
        for seed in seeds
    ]

    db_path = cfg.get("storage", {}).get("path", config.DB_PATH)
    storage = Storage(db_path)
    storage.insert_matches(results)

    total = len(results)
    breaches = sum(int(r.breach) for r in results)
    declined = sum(1 for r in results if r.outcome == "declined")
    fails = sum(1 for r in results if r.outcome == "protocol_fail")
    compliance = sum(int(r.protocol_ok) for r in results) / total
    print(
        f"db={db_path} experiment={context.experiment_id} run={context.run_id} "
        f"matches={total}"
    )
    print(
        f"breaches={breaches} declined={declined} protocol_fail={fails} "
        f"compliance={compliance:.3f}"
    )
    for backbone in sorted({r.backbone for r in results}):
        subset = [r for r in results if r.backbone == backbone]
        rate = sum(int(r.breach) for r in subset) / len(subset)
        print(f"  {backbone:>16}: rate={rate:.3f} n={len(subset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
