"""Parte C: diagnóstico de integridad (SOLO LECTURA; no limpia datos ni crea splits).

C1. HIKARI, 81 predictores de S0 + Label: duplicados exactos, patrones con etiquetas
    contradictorias, desglose por traffic_category, simulación de la regla de limpieza
    de CICIDS (sin guardar nada) y conflictos ignorando originp.
C2. Para S0, S1 y S2 en ambos datasets: métricas en el test completo vs. el subconjunto
    del test sin copia exacta en train (copia definida con las columnas del escenario).

Utilidades de hash y métricas por subconjunto: src/integrity.py.
"""
import json
import time

import numpy as np
import pandas as pd

from _common import add_datasets_arg, prefix, setup
from src.data import load_raw, make_predictors, make_target
from src.integrity import conflict_mask, group_stats, row_hash, subset_metrics, test_has_train_copy
from src.scenarios import build_scenario
from src.splits import load_split


def by_category(df, mask, name):
    sub = df.loc[mask, ["traffic_category", "Label"]]
    out = sub.groupby(["traffic_category", "Label"]).size().rename(name).reset_index()
    total = df.groupby(["traffic_category", "Label"]).size().rename("total_rows").reset_index()
    out = total.merge(out, on=["traffic_category", "Label"], how="left").fillna({name: 0})
    out[name] = out[name].astype(int)
    out[f"{name}_pct_of_category"] = out[name] / out["total_rows"] * 100
    return out


def c1_hikari(cfg, out):
    df = load_raw(cfg, "HIKARI2021")
    X = make_predictors(df, cfg, "HIKARI2021")
    y = make_target(df, "HIKARI2021", cfg).to_numpy()
    feats = list(X.columns)
    summary = {"rows": int(len(df)), "attack_rate": float(y.mean()), "predictors": len(feats)}

    k_feat = row_hash(X)
    k_full = row_hash(pd.concat([X, pd.Series(y, name="Label", index=X.index)], axis=1))

    # Duplicados exactos (81 predictores + Label)
    n_dup_groups, in_dup = group_stats(k_full)
    removable_dups = int(pd.Series(k_full).duplicated(keep="first").sum())
    summary.update({
        "duplicate_groups_predictors_plus_label": n_dup_groups,
        "rows_in_duplicate_groups": int(in_dup.sum()),
        "rows_removable_keep_one": removable_dups,
    })
    # Patrones con etiquetas contradictorias (mismos 81 predictores, Label distinto)
    conf = conflict_mask(k_feat, y)
    summary.update({
        "conflicting_patterns": int(pd.Series(k_feat[conf]).nunique()),
        "rows_in_conflicting_patterns": int(conf.sum()),
        "attack_rows_in_conflicts": int(y[conf].sum()),
        "benign_rows_in_conflicts": int((y[conf] == 0).sum()),
    })

    cat = by_category(df, in_dup, "rows_in_duplicate_groups").merge(
        by_category(df, conf, "rows_in_conflicts").drop(columns="total_rows"),
        on=["traffic_category", "Label"])

    # Simulación de la regla de CICIDS (en memoria): (1) quitar conflictos, (2) dedup dejando uno
    keep1 = ~conf
    k_full_after = k_full[keep1]
    keep2_local = ~pd.Series(k_full_after).duplicated(keep="first").to_numpy()
    final_mask = np.zeros(len(df), dtype=bool)
    final_mask[np.flatnonzero(keep1)[keep2_local]] = True
    summary.update({
        "sim_rows_after_conflict_removal": int(keep1.sum()),
        "sim_rows_after_dedup": int(final_mask.sum()),
        "sim_rows_removed_total": int(len(df) - final_mask.sum()),
        "sim_attack_rows_final": int(y[final_mask].sum()),
        "sim_attack_rate_final": float(y[final_mask].mean()),
    })
    sim_cat = df.groupby(["traffic_category", "Label"]).size().rename("original").reset_index().merge(
        df.loc[final_mask].groupby(["traffic_category", "Label"]).size().rename("after_cicids_rule").reset_index(),
        on=["traffic_category", "Label"], how="left").fillna({"after_cicids_rule": 0})
    sim_cat["after_cicids_rule"] = sim_cat["after_cicids_rule"].astype(int)
    sim_cat["removed"] = sim_cat["original"] - sim_cat["after_cicids_rule"]
    sim_cat["removed_pct"] = sim_cat["removed"] / sim_cat["original"] * 100

    # Conflictos ignorando originp (R11)
    k_no_port = row_hash(X.drop(columns=["originp"]))
    conf_np = conflict_mask(k_no_port, y)
    summary.update({
        "no_originp_conflicting_patterns": int(pd.Series(k_no_port[conf_np]).nunique()),
        "no_originp_rows_in_conflicts": int(conf_np.sum()),
        "no_originp_attack_rows_in_conflicts": int(y[conf_np].sum()),
    })
    cat_np = by_category(df, conf_np, "rows_in_conflicts_without_originp")
    cat = cat.merge(cat_np.drop(columns="total_rows"), on=["traffic_category", "Label"])

    # Pares de categorías implicadas en los conflictos (sobre 81 predictores)
    pairs = (df.loc[conf, ["traffic_category"]].assign(k=k_feat[conf])
             .groupby("k")["traffic_category"].agg(lambda s: " | ".join(sorted(set(s))))
             .value_counts().rename_axis("categories_in_pattern").reset_index(name="patterns"))

    # Cruce BENIGN x categoría de ataque: por cada patrón conflictivo, qué categorías
    # benignas (Label 0) y de ataque (Label 1) contiene, y cuántas filas aporta cada lado
    cf = df.loc[conf, ["traffic_category"]].assign(k=k_feat[conf], y=y[conf])
    benign_cats = cf[cf.y == 0].groupby("k")["traffic_category"].agg(lambda s: " + ".join(sorted(set(s))))
    attack_side = cf[cf.y == 1].groupby(["k", "traffic_category"]).size().rename("attack_rows").reset_index()
    benign_rows = cf[cf.y == 0].groupby("k").size().rename("benign_rows")
    attack_side = attack_side.join(benign_cats.rename("benign_categories"), on="k").join(benign_rows, on="k")
    benign_vs_attack = (attack_side.groupby(["traffic_category", "benign_categories"])
                        .agg(patterns=("k", "nunique"), attack_rows=("attack_rows", "sum"),
                             benign_rows=("benign_rows", "sum"))
                        .reset_index().rename(columns={"traffic_category": "attack_category"})
                        .sort_values("attack_rows", ascending=False))
    attack_totals = df[df["Label"].astype(int) == 1].groupby("traffic_category").size()
    benign_vs_attack["attack_rows_pct_of_attack_category"] = (
        benign_vs_attack["attack_rows"] / benign_vs_attack["attack_category"].map(attack_totals) * 100)
    multi_attack = int((attack_side.groupby("k")["traffic_category"].nunique() > 1).sum())
    summary["conflicting_patterns_with_more_than_one_attack_category"] = multi_attack

    # Patrones de un mismo traffic_category con Label distinto (no debería ocurrir)
    summary["patterns_conflicting_within_same_category"] = int(
        (cf.groupby(["k", "traffic_category"])["y"].nunique() > 1).sum())

    d = out / "integrity"
    d.mkdir(exist_ok=True)
    (d / "C1_hikari_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cat.to_csv(d / "C1_hikari_by_category.csv", index=False)
    sim_cat.to_csv(d / "C1_hikari_cicids_rule_simulation_by_category.csv", index=False)
    pairs.to_csv(d / "C1_hikari_conflict_category_pairs.csv", index=False)
    benign_vs_attack.to_csv(d / "C1_hikari_conflicts_benign_vs_attack_category.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(benign_vs_attack.to_string(index=False))
    print(cat.to_string(index=False))
    print(sim_cat.to_string(index=False))
    print(pairs.head(15).to_string(index=False))


def c2_memorization(cfg, out, datasets):
    rows = []
    for dataset in datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        del df
        train_idx, test_idx = load_split(cfg, dataset)
        y_test = y[test_idx]
        for scenario in ["S0", "S1", "S2"]:
            Xs = build_scenario(scenario, X, cfg, dataset)
            k = row_hash(Xs)
            has_copy = test_has_train_copy(k, train_idx, test_idx)
            pdir = out / "predictions"
            pred = np.load(pdir / f"{prefix(cfg, dataset)}_{scenario}_predictions.npy")
            proba = np.load(pdir / f"{prefix(cfg, dataset)}_{scenario}_probabilities.npy")
            for subset, mask in [("full_test", np.ones(len(test_idx), bool)),
                                 ("test_without_train_copy", ~has_copy),
                                 ("test_with_train_copy", has_copy)]:
                rows.append({"dataset": dataset, "scenario": scenario, "subset": subset,
                             "pct_of_test": float(mask.mean() * 100),
                             **subset_metrics(y_test, pred, proba, mask)})
            del Xs
    res = pd.DataFrame(rows)
    res.to_csv(out / "integrity" / "C2_memorization_metrics.csv", index=False)
    print(res.to_string(index=False))


def peak_memory_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 1e9
    except ImportError:
        return float("nan")


def main():
    args, cfg, out = setup("Diagnóstico de integridad (solo lectura)", add_datasets_arg)
    t0 = time.perf_counter()
    c1_hikari(cfg, out)
    t_c1 = time.perf_counter() - t0
    mem_c1 = peak_memory_gb()
    t0 = time.perf_counter()
    c2_memorization(cfg, out, args.datasets)
    cost = {"c1_hikari_seconds": t_c1, "peak_memory_gb_after_c1": mem_c1,
            "c2_seconds": time.perf_counter() - t0, "peak_memory_gb_after_c2": peak_memory_gb()}
    (out / "integrity" / "runtime_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
