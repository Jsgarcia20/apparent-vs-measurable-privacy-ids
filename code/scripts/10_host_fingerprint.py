"""E2 — Diagnóstico de huella del host en HIKARI v2 (no reentrena escenarios ni toca el test salvo en a).

a) FPR por host de origen en el TEST, con las predicciones ya guardadas (S0/S1/S2): flujos Label 0
   de cada IP atacante frente al resto de hosts. n, FP, FPR e IC de Wilson al 95 %.
b) Importancia por permutación en S0, calculada en la VALIDACIÓN INTERNA (mismo subtrain/validación
   que el protocolo 9A–9C). Se mide la caída de PR-AUC (average precision) y de F1 al umbral del
   dataset; 5 repeticiones, semilla 42. Por variable y por familia; la familia se permuta
   conjuntamente (las mismas filas para todas sus columnas).
c) Ablación por familia en S0 (validación interna): reentrenar sin la familia y medir ΔF1 y ΔPR-AUC.
d) Avance de la Fase 11 (diagnóstico): RF que predice, entre los flujos Label 0 del TRAIN, si el
   host de origen es una IP atacante. Variables de S0, S1 y S2. Validación 80/20 estratificada,
   semilla 42. Se reportan macro-F1 y balanced accuracy.

Config del RF: hiperparámetros congelados, y class_weight y umbral validados del dataset
(en d, class_weight="balanced" porque la tarea es distinta).
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from _common import add_datasets_arg, path_of, prefix, setup
from src.data import load_raw, make_predictors, make_target
from src.families import family_of  # reglas de HIKARI por defecto
from src.models import class_weight_for, create_random_forest
from src.scenarios import build_scenario
from src.splits import load_split

N_REPEATS = 5
SEED = 42


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def scores(model, X, y, threshold):
    p = model.predict_proba(X)[:, 1]
    return average_precision_score(y, p), f1_score(y, (p >= threshold).astype(np.int8), zero_division=0)


def part_a(cfg, dataset, df, y, test_idx, attackers, source_run):
    pdir = path_of(cfg, "results_dir") / source_run / "predictions"
    origin = df["originh"].to_numpy()[test_idx]
    y_test = y[test_idx]
    rows = []
    for scenario in ["S0", "S1", "S2"]:
        pred = np.load(pdir / f"{prefix(cfg, dataset)}_{scenario}_predictions.npy")
        groups = {ip: origin == ip for ip in attackers}
        groups["other_hosts"] = ~np.isin(origin, list(attackers))
        for g, m in groups.items():
            m = m & (y_test == 0)
            n, fp = int(m.sum()), int(pred[m].sum())
            lo, hi = wilson(fp, n)
            rows.append({"dataset": dataset, "scenario": scenario, "origin_group": g, "n_label0": n,
                         "fp": fp, "fpr": fp / n if n else np.nan, "wilson95_low": lo, "wilson95_high": hi})
    return pd.DataFrame(rows)


def internal_split(cfg, X, y, train_idx):
    v = cfg["validation"]
    X_tr, y_tr = X.iloc[train_idx], pd.Series(y[train_idx])
    sub, val = train_test_split(np.arange(len(X_tr)), test_size=v["internal_validation_size"],
                                random_state=v["internal_validation_seed"], stratify=y_tr)
    return X_tr.iloc[sub], y_tr.iloc[sub].to_numpy(), X_tr.iloc[val], y_tr.iloc[val].to_numpy()


def part_bc(cfg, dataset, X, y, train_idx):
    threshold = cfg["datasets"][dataset]["decision_threshold"]
    X_sub, y_sub, X_val, y_val = internal_split(cfg, X, y, train_idx)
    fams = {c: family_of(c) for c in X.columns}
    model = create_random_forest(cfg, dataset)
    model.fit(X_sub, y_sub)
    base_ap, base_f1 = scores(model, X_val, y_val, threshold)

    rng = np.random.default_rng(SEED)
    units = [("feature", c, [c]) for c in X.columns] + \
            [("family", f, [c for c in X.columns if fams[c] == f]) for f in dict.fromkeys(fams.values())]
    perm_rows = []
    for kind, name, cols in units:
        drops_ap, drops_f1 = [], []
        for _ in range(N_REPEATS):
            Xp = X_val.copy()
            order = rng.permutation(len(Xp))
            Xp[cols] = X_val[cols].to_numpy()[order]
            ap, f1 = scores(model, Xp, y_val, threshold)
            drops_ap.append(base_ap - ap)
            drops_f1.append(base_f1 - f1)
        perm_rows.append({"dataset": dataset, "kind": kind, "name": name,
                          "family": fams[name] if kind == "feature" else name, "n_columns": len(cols),
                          "importance_pr_auc_mean": float(np.mean(drops_ap)), "importance_pr_auc_std": float(np.std(drops_ap)),
                          "importance_f1_mean": float(np.mean(drops_f1)), "importance_f1_std": float(np.std(drops_f1))})
    perm = pd.DataFrame(perm_rows)
    fam_sum = (perm[perm.kind == "feature"].groupby("family")[["importance_pr_auc_mean", "importance_f1_mean"]]
               .sum().add_suffix("_sum_of_features").reset_index())
    del model

    abl_rows = [{"dataset": dataset, "removed_family": "(none)", "n_features": X.shape[1],
                 "pr_auc": base_ap, "f1": base_f1, "delta_pr_auc": 0.0, "delta_f1": 0.0}]
    for f in dict.fromkeys(fams.values()):
        keep = [c for c in X.columns if fams[c] != f]
        m = create_random_forest(cfg, dataset)
        m.fit(X_sub[keep], y_sub)
        ap, f1 = scores(m, X_val[keep], y_val, threshold)
        abl_rows.append({"dataset": dataset, "removed_family": f, "n_features": len(keep), "pr_auc": ap, "f1": f1,
                         "delta_pr_auc": ap - base_ap, "delta_f1": f1 - base_f1})
        del m
    return perm, fam_sum, pd.DataFrame(abl_rows), {"base_pr_auc": base_ap, "base_f1": base_f1,
                                                   "validation_rows": int(len(y_val)), "threshold": threshold}


def part_d(cfg, dataset, X, y, df, train_idx, attackers):
    tr0 = train_idx[y[train_idx] == 0]                       # solo flujos Label 0 del train
    host = df["originh"].to_numpy()[tr0]
    target = np.isin(host, list(attackers)).astype(np.int8)
    sub, val = train_test_split(np.arange(len(tr0)), test_size=0.2, random_state=SEED, stratify=target)
    rows = []
    for scenario in ["S0", "S1", "S2"]:
        Xs = build_scenario(scenario, X, cfg, dataset).iloc[tr0]
        m = create_random_forest(cfg, dataset, class_weight="balanced")
        t0 = time.perf_counter()
        m.fit(Xs.iloc[sub], target[sub])
        pred = m.predict(Xs.iloc[val])
        rows.append({"dataset": dataset, "scenario": scenario, "features": Xs.shape[1],
                     "label0_train_rows": int(len(tr0)), "attacker_host_share": float(target.mean()),
                     "validation_rows": int(len(val)),
                     "macro_f1": float(f1_score(target[val], pred, average="macro")),
                     "balanced_accuracy": float(balanced_accuracy_score(target[val], pred)),
                     "majority_baseline_balanced_accuracy": 0.5, "training_seconds": time.perf_counter() - t0})
        print(rows[-1], flush=True)
        del m, Xs
    return pd.DataFrame(rows)


def extra(parser):
    add_datasets_arg(parser, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"])
    parser.add_argument("--predictions-run", default="2026-10-06_D4")


def main():
    args, cfg, out = setup("E2: huella del host", extra)
    out = out / "E2_host_fingerprint"
    out.mkdir(exist_ok=True)
    outputs = {k: [] for k in ["a", "perm", "fam", "abl", "d"]}
    meta = {}
    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        train_idx, test_idx = load_split(cfg, dataset)
        attackers = sorted(set(df.loc[y == 1, "originh"]))
        meta[dataset] = {"attacker_ips": attackers,
                         "family_sizes": pd.Series({c: family_of(c) for c in X.columns}).value_counts().to_dict()}
        outputs["a"].append(part_a(cfg, dataset, df, y, test_idx, attackers, args.predictions_run))
        print(outputs["a"][-1].to_string(index=False), flush=True)
        perm, fam, abl, base = part_bc(cfg, dataset, X, y, train_idx)
        meta[dataset].update(base)
        outputs["perm"].append(perm)
        outputs["fam"].append(fam.assign(dataset=dataset))
        outputs["abl"].append(abl)
        print(abl.to_string(index=False), flush=True)
        outputs["d"].append(part_d(cfg, dataset, X, y, df, train_idx, attackers))
        del df, X
    names = {"a": "a_fpr_by_origin_host_test.csv", "perm": "b_permutation_importance_validation.csv",
             "fam": "b_family_importance_sum_of_features.csv", "abl": "c_family_ablation_validation.csv",
             "d": "d_attacker_host_identification.csv"}
    for k, fname in names.items():
        pd.concat(outputs[k], ignore_index=True).to_csv(out / fname, index=False)
    (out / "E2_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
