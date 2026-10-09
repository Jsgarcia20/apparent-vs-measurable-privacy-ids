"""A3 — Reidentificación entre dos hosts con el mismo rol (HIKARI v2_main y v2_pairs).

Definición adoptada:
- Solo flujos Label 0 de 103.255.15.23 y 103.255.15.150 con responp = 53 (A3-DNS) o 443 (A3-443).
- Clase positiva: 103.255.15.150 (solo afecta al sentido del ROC-AUC).
- Escenarios S0, S1a, S1b y S2; atacante de src/attack.py con semillas 42, 43 y 44;
  aux = train del split vigente, víctima = test.
- Métrica principal: balanced accuracy (azar = 0,5). También macro-F1, ROC-AUC, accuracy,
  baseline de clase mayoritaria y proporción de cada host en train y test.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import add_datasets_arg, setup
from src.attack import ATTACKER_SEEDS, attack_metrics, attacker
from src.data import load_raw, make_predictors, make_target
from src.scenarios import build_scenario
from src.splits import load_split

HOSTS = ["103.255.15.23", "103.255.15.150"]
POSITIVE = "103.255.15.150"
PORTS = {"A3_DNS": 53, "A3_443": 443}
SCENARIOS = [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]


def main():
    args, cfg, out = setup("A3: dos hosts con el mismo rol", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]))
    out = out / "A3_same_role_host"
    out.mkdir(exist_ok=True)
    rows, sizes = [], []
    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        host = df["originh"].astype(str).to_numpy()
        train_idx, test_idx = load_split(cfg, dataset)
        in_train = np.zeros(len(df), bool)
        in_train[train_idx] = True
        in_test = np.zeros(len(df), bool)
        in_test[test_idx] = True
        for objective, port in PORTS.items():
            base = (y == 0) & (X["responp"].to_numpy() == port) & np.isin(host, HOSTS)
            tr, te = np.flatnonzero(base & in_train), np.flatnonzero(base & in_test)
            target = (host == POSITIVE).astype(np.int8)
            for part, idx in [("train", tr), ("test", te)]:
                for h in HOSTS:
                    sizes.append({"dataset": dataset, "objective": objective, "partition": part, "host": h,
                                  "rows": int((host[idx] == h).sum()), "share": float((host[idx] == h).mean())})
            majority = int(pd.Series(target[tr]).value_counts().index[0])
            for name, internal in SCENARIOS:
                Xs = build_scenario(internal, X, cfg, dataset)
                for seed in ATTACKER_SEEDS:
                    m = attacker(cfg, seed)
                    t0 = time.perf_counter()
                    m.fit(Xs.iloc[tr], target[tr])
                    proba_all = m.predict_proba(Xs.iloc[te])
                    proba = proba_all[:, list(m.classes_).index(1)]
                    pred = m.classes_[np.argmax(proba_all, axis=1)]   # equivale a m.predict
                    r = {"dataset": dataset, "objective": objective, "responp": port, "scenario": name, "seed": seed,
                         "n_features": Xs.shape[1], "n_train": len(tr), "n_test": len(te),
                         **attack_metrics(target[te], pred, [0, 1], majority),
                         "roc_auc": float(roc_auc_score(target[te], proba)), "fit_seconds": time.perf_counter() - t0}
                    rows.append(r)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                    del m
                del Xs
                pd.DataFrame(rows).to_csv(out / "a3_runs.csv", index=False)
        del df, X
    pd.DataFrame(sizes).to_csv(out / "a3_host_proportions.csv", index=False)
    summary = (pd.DataFrame(rows).groupby(["dataset", "objective", "scenario"], sort=False)
               .agg(balanced_accuracy=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
                    macro_f1=("macro_f1", "mean"), roc_auc=("roc_auc", "mean"), accuracy=("accuracy", "mean"),
                    majority_baseline=("majority_baseline", "first"), n_train=("n_train", "first"),
                    n_test=("n_test", "first")).reset_index())
    summary.to_csv(out / "a3_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(sizes).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
