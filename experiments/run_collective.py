import argparse
import datetime
import json
import os

import yaml

from arena_limits import config
from arena_limits.arena.client import LiteLLMChatClient
from arena_limits.arena.provenance import git_sha
from arena_limits.arena.types import RunContext
from arena_limits.collective.runner import run_collective
from arena_limits.storage import Storage


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Коллективные прогоны: несколько агентов в одном условии."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    context = RunContext(
        experiment_id=str(cfg.get("experiment_id", "collective")),
        run_id=args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        git_sha=git_sha(),
    )
    os.environ.setdefault("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    client = LiteLLMChatClient(
        api_key=os.environ.get("GIGACHAT_CREDENTIALS"),
        ssl_verify=False,
        max_tokens=context.max_tokens,
    )

    out = run_collective(cfg, client, context)
    db_path = cfg.get("storage", {}).get("path", config.DB_PATH)
    storage = Storage(db_path)
    if "sessions" in out:
        storage.insert_sessions(out["sessions"])

    print(f"db={db_path} mode={out['mode']}")
    stats = out.get("stats")
    if (
        isinstance(stats, dict)
        and stats
        and isinstance(next(iter(stats.values())), dict)
    ):
        for condition, value in stats.items():
            print(condition)
            print(json.dumps(value, ensure_ascii=False, indent=2, default=float))
    elif stats:
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=float))
    if "chain" in out:
        chain = out["chain"]
        print(
            f"chain levels={chain.levels} principal_floor={chain.principal_floor} "
            f"handed_down={chain.handed_down_floor} degraded={chain.degraded}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
