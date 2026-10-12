import argparse
import json

from arena_limits import config
from arena_limits.evaluate import compliance_by_model
from arena_limits.storage import Storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Аудит протокола: доля валидных тегов и нестрогие разборы."
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument(
        "--dump", type=int, default=20, help="сколько разборов выгрузить"
    )
    args = parser.parse_args(argv)

    frame = Storage(args.db).load_matches()
    if frame.empty:
        print("нет прогонов")
        return 1

    print(
        json.dumps(
            compliance_by_model(frame), ensure_ascii=False, indent=2, default=float
        )
    )
    print(f"--- нестрогие разборы (первые {args.dump}) ---")
    for _, row in frame[~frame["strict"].astype(bool)].head(args.dump).iterrows():
        turns = json.loads(row["transcript"])
        last_agent = [t["content"] for t in turns if t["role"] == "agent"]
        print(f"=== {row['agent_id']} | {row['opponent_id']} | {row['condition']}")
        print((last_agent[-1] if last_agent else "")[-300:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
