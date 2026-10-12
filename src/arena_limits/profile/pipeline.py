import argparse

from arena_limits import config
from arena_limits.profile.correlation import (
    bootstrap_ci,
    correlation_matrix,
    effective_n,
    rho_by_group,
)
from arena_limits.profile.irt import fit_irt
from arena_limits.storage import Storage


def profile_matches(
    df,
    model_version: str = "irt-logit-v1",
    bootstrap: int = 200,
    seed: int = 0,
) -> dict:
    """
    Полный проход профилирования прогонов Arena.

    Считает логит-профили агентов и противников, тетрахорические корреляции
    агентов, ``N_eff`` и бутстрэп-интервалы для ``rho_within``, ``rho_between``
    и ``N_eff``.

    Parametrs
    ----------
    df : pandas.DataFrame
        Таблица прогонов из :meth:`Storage.load_matches`.
    model_version : str
        Метка версии профилировщика для воспроизводимости.
    bootstrap : int
        Число бутстрэп-выборок; ``0`` отключает интервалы.
    seed : int
        Seed генератора бутстрэпа.

    Returns
    -------
    dict
        Строки ``profiles``, ``opponents`` и ``corr``, готовые к записи, плюс
        метрики ``agents``, ``n_eff``, ``rho``, ``ci`` и объект ``irt``.
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

    return {
        "profiles": profiles,
        "opponents": opponents,
        "corr": corr_rows,
        "agents": agents,
        "n_eff": n_eff,
        "rho": rho,
        "ci": ci,
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

    print(f"agents={len(out['agents'])} N_eff={out['n_eff']:.2f}")
    print(
        f"rho_within={out['rho']['within']:+.3f} "
        f"rho_between={out['rho']['between']:+.3f}"
    )
    if out["ci"]:
        print(
            f"  within CI={out['ci']['within_ci']}  between CI={out['ci']['between_ci']}"
        )
        print(f"  N_eff CI={out['ci']['n_eff_ci']}")
    print("p_hat:")
    for agent in out["agents"]:
        print(
            f"  {agent:>12}: {out['irt'].p_hat[agent]:.3f}  "
            f"a_i={out['irt'].a[agent]:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
