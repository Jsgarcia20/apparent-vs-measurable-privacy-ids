"""Sensibilidad al conocimiento auxiliar del atacante.

A2 (host de origen, top-10 + other, flujos Label 0) y A3 (.15.23 frente a .15.150, Label 0, responp 53 y
443), en HIKARI v2_main y v2_pairs, escenarios S0, S1a, S1b y S2.
- Conocimiento auxiliar = submuestra estratificada (por la clase objetivo) del train del 1 %, 5 % y 20 %
  (el 100 % ya está en F2 y A3). Víctimas = test completo, como antes.
- 3 semillas de submuestreo (42, 43, 44) × 3 semillas del atacante (42, 43, 44).
- Las clases objetivo y el baseline de clase mayoritaria se definen con el train completo, igual que en
  F2/A3, para que las métricas sean comparables entre fracciones.
- Métricas: balanced accuracy, macro-F1, ventaja (y ROC-AUC en A3).
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _common import add_datasets_arg, setup
from src.attack import ATTACKER_SEEDS, attack_metrics, attacker
from src.data import load_raw, make_predictors, make_target
from src.s3_common import A3_HOSTS
from src.scenarios import build_scenario
from src.splits import load_split

FRACTIONS = [0.01, 0.05, 0.20]
SUBSAMPLE_SEEDS = [42, 43, 44]
SCENARIOS = [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]


def tasks(df, X, y, tr, te):
    host = df["originh"].astype(str).to_numpy()
    l0 = y == 0
    tr0, te0 = tr[l0[tr]], te[l0[te]]
    top = pd.Series(host[tr0]).value_counts().index[:10]
    out = [("A2_source_host", np.where(np.isin(host, top), host, "other"), tr0, te0)]
    for name, port in [("A3_DNS", 53), ("A3_443", 443)]:
        keep = l0 & (X["responp"].to_numpy() == port) & np.isin(host, A3_HOSTS)
        out.append((name, host, tr[keep[tr]], te[keep[te]]))
    return out


def main():
    args, cfg, out = setup("Sensibilidad al conocimiento auxiliar", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]))
    out = out / "aux_knowledge"
    out.mkdir(exist_ok=True)
    path = out / "aux_knowledge_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["objective"], r["scenario"], r["fraction"], r["subsample_seed"], r["attacker_seed"]) for r in rows}
    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        tr, te = load_split(cfg, dataset)
        task_list = tasks(df, X, y, tr, te)
        del df
        for name, internal in SCENARIOS:
            Xs = build_scenario(internal, X, cfg, dataset)
            for obj, labels, a_tr, a_te in task_list:
                classes = pd.Series(labels[a_tr]).value_counts().index.tolist()
                majority = classes[0]
                Xte = Xs.iloc[a_te]
                for frac in FRACTIONS:
                    for ss in SUBSAMPLE_SEEDS:
                        aux, _ = train_test_split(a_tr, train_size=frac, random_state=ss, stratify=labels[a_tr])
                        for seed in ATTACKER_SEEDS:
                            key = (dataset, obj, name, frac, ss, seed)
                            if key in done:
                                continue
                            m = attacker(cfg, seed)
                            t0 = time.perf_counter()
                            m.fit(Xs.iloc[aux], labels[aux])
                            proba = m.predict_proba(Xte)
                            pred = m.classes_[np.argmax(proba, axis=1)]
                            r = {"dataset": dataset, "objective": obj, "scenario": name, "fraction": frac,
                                 "subsample_seed": ss, "attacker_seed": seed, "n_aux": len(aux), "n_victims": len(a_te),
                                 "n_classes_in_aux": int(len(np.unique(labels[aux]))),
                                 **attack_metrics(labels[a_te], pred, classes, majority),
                                 "fit_seconds": time.perf_counter() - t0}
                            if obj.startswith("A3"):
                                pos = A3_HOSTS[1]
                                r["roc_auc"] = float(roc_auc_score(labels[a_te] == pos, proba[:, list(m.classes_).index(pos)]))
                            rows.append(r)
                            del m
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"{dataset} {name} {obj}: {len(rows)} filas acumuladas", flush=True)
            del Xs


if __name__ == "__main__":
    main()
