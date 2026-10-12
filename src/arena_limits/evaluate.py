import math
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from arena_limits import config
from arena_limits.monitor.eprocess import EProcessMonitor
from arena_limits.monitor.streams import SyntheticStream
from arena_limits.profile.correlation import wilson_interval
from arena_limits.profile.irt import fit_irt, pair_probability
from arena_limits.profile.pipeline import profile_matches


def calibration(df) -> dict:
    """
    Калибровка ``p_hat``: Brier и ECE против фактических долей пробоев
    """
    if "p_hat" not in df.columns:
        return {"brier": float("nan"), "ece": float("nan"), "n_bins": 0}
    per_agent = df.groupby("agent_id").agg(p=("p_hat", "first"), rate=("breach", "mean"))
    if per_agent.empty:
        return {"brier": float("nan"), "ece": float("nan"), "n_bins": 0}
    brier = float(np.mean((per_agent["p"] - per_agent["rate"]) ** 2))
    bins = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    for lo, hi in pairwise(bins):
        mask = (per_agent["p"] >= lo) & (per_agent["p"] < hi)
        if mask.sum():
            ece += (
                mask.sum()
                / len(per_agent)
                * abs(per_agent.loc[mask, "p"].mean() - per_agent.loc[mask, "rate"].mean())
            )
    return {"brier": brier, "ece": float(ece), "n_bins": len(per_agent)}


def compliance_by_model(df) -> dict:
    """
    Дисциплина протокола по бэкбонам, включая причину остановки.

    Нужна, чтобы отличать «модель держится» от «модель не умеет отвечать по
    формату»: пропуск тега несёт информацию и не должен растворяться в `declined`.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.

    Returns
    -------
    dict
        По бэкбонам: доля валидных тегов, строгих разборов, повторов и обрывов.
    """
    out = {}
    for backbone, group in df.groupby("backbone"):
        n = len(group)
        finish_length = (
            int((group["finish_reason"] == "length").sum())
            if "finish_reason" in group.columns
            else 0
        )
        out[str(backbone)] = {
            "n": n,
            "protocol_ok": float(group["protocol_ok"].mean()),
            "strict": float(group["strict"].mean()),
            "repaired": float(group["repaired"].mean()),
            "finish_length": finish_length,
        }
    return out


def manski_bounds(df) -> dict:
    """
    Границы Мански для доли пробоев при информативных пропусках.

    Пропуск протокола мог бы быть и пробоем, и отказом. Нижняя граница считает
    все ``protocol_fail`` отказами, верхняя — пробоями; истина внутри.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.

    Returns
    -------
    dict
        ``lower``, ``upper`` и число пропусков протокола.
    """
    n = len(df)
    breaches = int(df["breach"].sum())
    fails = int((~df["protocol_ok"].astype(bool)).sum())
    return {
        "lower": breaches / n,
        "upper": (breaches + fails) / n,
        "n_protocol_fail": fails,
    }


def clustered_bootstrap_rate(
    df, cluster_col: str = "backbone", n_boot: int = 2000, seed: int = 0
) -> dict:
    """
    Интервал доли пробоев при кластерном бутстрэпе по бэкбонам.

    Единица независимости — бэкбон, а не сделка: сделки одной модели не
    независимы, поэтому обычный бутстрэп по сделкам занижает интервал.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.
    cluster_col : str
        Столбец кластера (по умолчанию бэкбон).
    n_boot : int
        Число бутстрэп-выборок.
    seed : int
        Seed генератора.

    Returns
    -------
    dict
        Точечная доля пробоев и её интервал.
    """
    clusters = list(df[cluster_col].unique())
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([df[df[cluster_col] == c] for c in chosen])
        rates.append(float(sample["breach"].mean()))
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return {"point": float(df["breach"].mean()), "ci": (float(lo), float(hi))}


def _design(frame, agents, opponents, pairs, mode: str) -> np.ndarray:
    if mode == "pair":
        key = frame["agent_id"].astype(str) + "|" + frame["opponent_id"].astype(str)
        return pd.get_dummies(key).reindex(columns=pairs, fill_value=0).to_numpy(float)
    a = pd.get_dummies(frame["agent_id"]).reindex(columns=agents, fill_value=0).to_numpy(float)
    o = (
        pd.get_dummies(frame["opponent_id"])
        .reindex(columns=opponents, fill_value=0)
        .to_numpy(float)
    )
    return np.hstack([a, o])


def _fit_predict(train, test, mode: str = "additive", C: float = 1.0) -> np.ndarray:
    if train["breach"].nunique() < 2:
        return np.full(len(test), float(train["breach"].mean()))
    agents = sorted(train["agent_id"].unique())
    opponents = sorted(train["opponent_id"].unique())
    pairs = sorted(
        (train["agent_id"].astype(str) + "|" + train["opponent_id"].astype(str)).unique()
    )
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(_design(train, agents, opponents, pairs, mode), train["breach"].astype(int))
    return model.predict_proba(_design(test, agents, opponents, pairs, mode))[:, 1]


def cross_validate(
    df, leave_out: str = "backbone", mode: str = "additive", C: float = 1.0
) -> list[dict]:
    """
    Внешняя проверка: обучить без одной группы, проверить на ней.

    ``leave_out`` ∈ {``backbone``, ``opponent``, ``scenario_id``} — так проверяется
    холодный старт, который заявлен в продукте.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.
    leave_out : str
        Столбец, по которому исключается группа.
    mode : str
        ``additive`` — агент + противник; ``pair`` — взаимодействие.
    C : float
        Обратная сила L2-регуляризации.

    Returns
    -------
    list of dict
        По фолдам: размер, log-loss и Brier.
    """
    folds = []
    for value in sorted(df[leave_out].astype(str).unique()):
        mask = df[leave_out].astype(str) == value
        if mask.all() or not mask.any():
            continue
        test, train = df[mask], df[~mask]
        p = _fit_predict(train, test, mode, C)
        y = test["breach"].astype(int).to_numpy()
        folds.append(
            {
                "fold": value,
                "n": len(test),
                "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
            }
        )
    return folds


def interaction_test(df, n_splits: int = 5, seed: int = 0, C: float = 1.0) -> dict:
    """
    Проверка тезиса «риск у пары»: даёт ли взаимодействие агент×противник
    выигрыш out-of-sample по сравнению с аддитивной моделью.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.
    n_splits : int
        Число кросс-валидационных фолдов.
    seed : int
        Seed перемешивания.
    C : float
        Обратная сила L2-регуляризации.

    Returns
    -------
    dict
        Log-loss аддитивной и парной моделей и их разность ``delta``.
    """
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(df)), n_splits)
    scores = {"additive": [], "pair": []}
    for k in range(n_splits):
        test = df.iloc[folds[k]]
        train = df.iloc[np.concatenate([folds[j] for j in range(n_splits) if j != k])]
        if test.empty or train.empty or test["breach"].nunique() < 2:
            continue
        y = test["breach"].astype(int).to_numpy()
        for mode, values in scores.items():
            p = _fit_predict(train, test, mode, C)
            values.append(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])))
    if not scores["additive"]:
        return {
            "additive_log_loss": float("nan"),
            "pair_log_loss": float("nan"),
            "delta": float("nan"),
        }
    additive = float(np.mean(scores["additive"]))
    pair = float(np.mean(scores["pair"]))
    return {
        "additive_log_loss": additive,
        "pair_log_loss": pair,
        "delta": additive - pair,
    }


def eprocess_oos(df, delta: float, kappa: float, holdout_frac: float = 0.5) -> dict:
    """
    Честная проверка e-процесса: ``q`` оценивается только по обучающим условиям.

    Иначе проверка круговая: мониторинг на тех же сделках, по которым построен
    прогноз, ничего не говорит о валидности.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица прогонов.
    delta : float
        Целевая доля ложных заморозок.
    kappa : float
        Допуск недооценки риска.
    holdout_frac : float
        Доля условий, отложенных под мониторинг.

    Returns
    -------
    dict
        По бэкбонам: ``q``, ``log_e`` и признак заморозки.
    """
    conditions = sorted(df["condition"].astype(str).unique())
    if len(conditions) < 2:
        return {}
    n_train = max(1, int(len(conditions) * (1.0 - holdout_frac)))
    train_mask = df["condition"].astype(str).isin(conditions[:n_train])
    test = df[~train_mask]
    p_hat = fit_irt(df[train_mask]).p_hat if df[train_mask]["breach"].nunique() > 1 else {}
    out = {}
    for backbone, group in test.groupby("backbone"):
        rates = [p_hat[a] for a in group["agent_id"].unique() if a in p_hat]
        if not rates:
            continue
        q = float(np.mean(rates))
        monitor = EProcessMonitor(delta=delta, kappa=kappa)
        ordered = group.sort_values(["condition", "seed"])
        for _, row in ordered.iterrows():
            monitor.update(int(row["breach"]), q)
        out[str(backbone)] = {"q": q, "log_e": monitor.log_e, "frozen": monitor.frozen}
    return out


def false_freeze_rate(
    q: float,
    delta: float,
    kappa: float,
    n: int,
    trials: int,
    seed: int = 0,
    rho: float = 0.0,
) -> float:
    """
    Доля ложных заморозок при калиброванном ``q`` (нулевая гипотеза).

    ``rho > 0`` вводит положительную корреляцию исходов внутри прогона: e-процесс
    выведен в предположении предсказуемых ставок, и при зависимых исходах доля
    ложных заморозок может превысить ``delta`` — это надо измерять, а не считать
    нулём.

    Parameters
    ----------
    q : float
        Калиброванная вероятность пробоя.
    delta, kappa : float
        Параметры монитора.
    n : int
        Длина потока в одном прогоне.
    trials : int
        Число независимых прогонов.
    seed : int
        Seed генератора.
    rho : float
        Внутрипрогонная корреляция исходов.

    Returns
    -------
    float
        Доля прогонов с ложной заморозкой.
    """
    rng = np.random.default_rng(seed)
    threshold = norm.ppf(q)
    freezes = 0
    for _ in range(trials):
        if rho > 0.0:
            shock = float(rng.standard_normal())
            latents = math.sqrt(rho) * shock + math.sqrt(1.0 - rho) * rng.standard_normal(n)
        else:
            latents = rng.standard_normal(n)
        ys = latents < threshold
        monitor = EProcessMonitor(delta=delta, kappa=kappa)
        if any(monitor.update(int(y), q) for y in ys):
            freezes += 1
    return freezes / trials


def detection_delay(q: float, rate: float, delta: float, kappa: float, seed: int = 0) -> int | None:
    """
    Число сделок до заморозки при недооценке риска (``rate > q``)
    """
    monitor = EProcessMonitor(delta=delta, kappa=kappa)
    stream = SyntheticStream("drift", q=q, n=5000, breach_rate=rate, seed=seed)
    for index, obs in enumerate(stream, start=1):
        if monitor.update(obs.y, obs.q):
            return index
    return None


def detection_delay_distribution(
    q: float, rate: float, delta: float, kappa: float, trials: int = 100, seed: int = 0
) -> dict:
    """
    Распределение задержки заморозки при дрейфе, а не один прогон
    """
    delays = [detection_delay(q, rate, delta, kappa, seed + trial) for trial in range(trials)]
    finite = [d for d in delays if d is not None]
    if not finite:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def limit_baselines(df) -> dict:
    """
    Ранговая корреляция трёх скоров с фактической долей пробоев по парам
    """
    irt = fit_irt(df)
    per_pair = df.groupby(["agent_id", "opponent_id"])["breach"].mean().reset_index()
    pair_scores = [
        pair_probability(irt, row["agent_id"], row["opponent_id"]) for _, row in per_pair.iterrows()
    ]
    agent_scores = [irt.p_hat[row["agent_id"]] for _, row in per_pair.iterrows()]
    realized = per_pair["breach"].to_numpy()
    return {
        "agent_spearman": float(spearmanr(agent_scores, realized).statistic),
        "pair_spearman": float(spearmanr(pair_scores, realized).statistic),
        "n_pairs": len(per_pair),
    }


def build_report(df, bootstrap: int = 100, seed: int = 0) -> dict:
    """
    Свести честностные метрики: протокол, границы, калибровка, заморозка, лимиты
    """
    out = profile_matches(df, bootstrap=bootstrap, seed=seed)
    delta, kappa = config.DEFAULT_DELTA, config.DEFAULT_KAPPA
    return {
        "matches": len(df),
        "compliance": float(df["protocol_ok"].mean()),
        "outcomes": df["outcome"].value_counts().to_dict(),
        "breach_rate": float(df["breach"].mean()),
        "breach_rate_ci": wilson_interval(int(df["breach"].sum()), len(df)),
        "breach_rate_ci_clustered": clustered_bootstrap_rate(df, seed=seed),
        "manski": manski_bounds(df),
        "compliance_by_model": compliance_by_model(df),
        "monoculture_heterogeneity": {
            "n_eff": out["n_eff"],
            "n_eff_ci": out["ci"].get("n_eff_ci"),
            "gap": out["monoculture"]["gap"],
            "model_vs_persona": out["monoculture"]["model_vs_persona"],
            "degenerate_pairs": out["degenerate_pairs"],
        },
        "calibration": calibration(df.assign(p_hat=df["agent_id"].map(out["irt"].p_hat))),
        "cross_validation": {
            "leave_one_backbone_out": cross_validate(df, "backbone"),
            "leave_one_opponent_out": cross_validate(df, "opponent_id"),
        },
        "interaction": interaction_test(df, seed=seed),
        "eprocess_oos": eprocess_oos(df, delta, kappa),
        "eprocess_false_freeze": {
            "independent": false_freeze_rate(0.2, delta, kappa, n=200, trials=300, seed=seed),
            "correlated_rho_0.3": false_freeze_rate(
                0.2, delta, kappa, n=200, trials=300, seed=seed, rho=0.3
            ),
        },
        "detection_delay": detection_delay_distribution(0.05, 0.5, delta, kappa, seed=seed),
        "baselines": limit_baselines(df),
    }
