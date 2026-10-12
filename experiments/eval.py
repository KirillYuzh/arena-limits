import argparse
import json

from arena_limits import config
from arena_limits.evaluate import build_report
from arena_limits.storage import Storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Оценка: протокол, калибровка, монокультура, заморозка."
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--out", default=None, help="куда сохранить JSON-отчёт")
    args = parser.parse_args(argv)

    df = Storage(args.db).load_matches()
    if df.empty:
        print("нет прогонов")
        return 1
    report = build_report(df, bootstrap=args.bootstrap)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=float)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
