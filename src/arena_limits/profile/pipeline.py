import argparse

import pandas as pd

from arena_limits import config
from arena_limits.profile.correlation import (
    between_backbone_gap,
    bootstrap_ci,
    correlation_matrix,
    degenerate_pairs,
    effective_n,
    model_vs_persona_effect,
    rho_by_group,
    wilson_interval,
    within_backbone_variance,
)
from arena_limits.profile.irt import fit_irt
from arena_limits.storage import Storage


def _rate_rows(df: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for key, group in df.groupby(column):
        k, n = int(group["breach"].sum()), len(group)
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "key": str(key),
                "n": n,
                "breaches": k,
                "rate": k / n if n else float("nan"),
                "ci": (lo, hi),
            }
        )
    return sorted(rows, key=lambda row: row["rate"])


def profile_matches(
    df: pd.DataFrame,
    model_version: str = "irt-logit-v1",
    bootstrap: int = 200,
    seed: int = 0,
) -> dict:
    """
    Полный проход профилирования прогонов Arena.

    Считает логит-профили агентов и противников, тетрахорические корреляции
    (``nan`` при вырождении), ``N_eff``, уровневые метрики монокультуры и
    бутстрэп-интервалы, а также долю соблюдения протокола и доли пробоев с
    интервалами Уилсона.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов из :meth:`Storage.load_matches`.
    model_version : str
        Метка версии профилировщика.
    bootstrap : int
        Число бутстрэп-выборок; ``0`` отключает интервалы.
    seed : int
        Seed генератора бутстрэпа.

    Returns
    -------
    dict
        Строки ``profiles``, ``opponents`` и ``corr``, метрики ``n_eff``,
        ``rho``, ``ci``, ``monoculture``, ``compliance``, ``rates`` и ``irt``.
    """
    irt = fit_irt(df)
    corr, agents = correlation_matrix(df)
    backbone = df.groupby("agent_id")["backbone"].first().to_dict()
    rho = rho_by_group(corr, agents, backbone)
    n_eff = effective_n(corr)
    ci = (
        bootstrap_ci(df, agents, backbone, n_boot=bootstrap, seed=seed)
        if bootstrap
        else {}
    )

    profiles = [
        {
            "agent_id": agent,
            "backbone": backbone.get(agent),
            "a_i": irt.a[agent],
            "intercept": irt.intercept,
            "p_hat": irt.p_hat[agent],
            "model_version": model_version,
        }
        for agent in irt.agents
    ]
    opponents = [
        {"opponent_id": opponent, "b_j": float(b)} for opponent, b in irt.b.items()
    ]
    corr_rows = [
        {"agent_i": agents[i], "agent_j": agents[j], "rho": float(corr[i, j])}
        for i in range(len(agents))
        for j in range(i + 1, len(agents))
    ]
    profiles_df = pd.DataFrame(profiles)
    monoculture = {
        "variance": within_backbone_variance(profiles_df),
        "gap": between_backbone_gap(profiles_df),
        "model_vs_persona": model_vs_persona_effect(profiles_df),
    }

    return {
        "profiles": profiles,
        "opponents": opponents,
        "corr": corr_rows,
        "agents": agents,
        "n_eff": n_eff,
        "degenerate_pairs": degenerate_pairs(corr),
        "rho": rho,
        "ci": ci,
        "monoculture": monoculture,
        "compliance": (
            float(df["protocol_ok"].mean()) if "protocol_ok" in df.columns else 0.0
        ),
        "rates": {
            "backbone": _rate_rows(df, "backbone"),
            "agent": _rate_rows(df, "agent_id"),
            "opponent": _rate_rows(df, "opponent_id"),
        },
        "irt": irt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Профилировать прогоны Arena в риск-профили."
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)

    storage = Storage(args.db)
    df = storage.load_matches()
    if df.empty:
        print("в хранилище нет прогонов: сначала запусти experiments/run_arena.py")
        return 1

    out = profile_matches(df, bootstrap=args.bootstrap)
    storage.upsert_profiles(out["profiles"])
    storage.upsert_opponents(out["opponents"])
    storage.upsert_corr(out["corr"])

    print(f"прогонов={len(df)} соблюдение протокола={out['compliance']:.3f}")
    if "outcome" in df.columns:
        print("исходы:", df["outcome"].value_counts().to_dict())
    print(
        f"agents={len(out['agents'])} N_eff={out['n_eff']:.2f} "
        f"(неопределённых пар: {out['degenerate_pairs']})"
    )
    print("доля пробоев по бэкбонам (CI Уилсона):")
    for row in out["rates"]["backbone"]:
        lo, hi = row["ci"]
        print(
            f"  {row['key']:>16}: {row['rate']:.3f} [{lo:.2f}, {hi:.2f}] n={row['n']}"
        )
    print("монокультура (уровневая):")
    for backbone, stats in sorted(out["monoculture"]["variance"].items()):
        print(f"  {backbone:>16}: std={stats['std']:.3f} n={stats['n_agents']}")
    print(f"  разрыв между бэкбонами: {out['monoculture']['gap']:.3f}")
    effect = out["monoculture"]["model_vs_persona"]
    print(
        f"  модель против персоны: ratio={effect['ratio']:.1f} "
        f"F={effect['anova_f']:.1f} p={effect['anova_p']:.2e}"
    )
    print(
        f"корреляции (вспомогательно): rho_within={out['rho']['within']:+.3f} "
        f"rho_between={out['rho']['between']:+.3f}"
    )
    if out["ci"]:
        print(
            f"  within CI={out['ci']['within_ci']} between CI={out['ci']['between_ci']}"
        )
        print(f"  N_eff CI={out['ci']['n_eff_ci']}")
    print("p_hat:")
    for agent in out["agents"]:
        print(
            f"  {agent:>14}: {out['irt'].p_hat[agent]:.3f} "
            f"a_i={out['irt'].a[agent]:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
