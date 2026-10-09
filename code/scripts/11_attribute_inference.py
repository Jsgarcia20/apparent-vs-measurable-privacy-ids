"""F2 — Fase 11A: inferencia de atributos (amenaza T1: publicación de datos).

Protocolo común:
- Atacante: RandomForest con los hiperparámetros congelados, pero class_weight=None y
  random_state ∈ {42, 43, 44}. Se reportan la media y la desviación de las 3 semillas.
- Conocimiento auxiliar = train del split vigente; víctima = test.
- Variables del atacante = columnas publicadas en el escenario (S0, S1a, S1b, S2), sin la columna
  objetivo. Si el escenario publica el objetivo tal cual, se registra una fila "exposed_directly"
  (éxito = 1 por construcción) y se mide además la inferencia sin esa columna.
- Las clases del objetivo se definen con frecuencias del TRAIN (top-K + "other"); el test nunca
  define clases.
- Métricas: accuracy, macro-F1 (sobre las clases del train), balanced accuracy, baseline de clase
  mayoritaria (clase más frecuente en train, evaluada en test) y
  ventaja = (acc − baseline) / (1 − baseline).

Objetivos:
- A1: puerto de destino (top-20 + other). CICIDS: Destination Port; HIKARI: responp. Todas las filas.
  En S1b se añade como referencia el baseline "puerto más frecuente del train dentro del rango IANA
  observado".
- A2: host de origen (originh), solo HIKARI y solo flujos Label 0; top-10 hosts del train + other.
- A3: host de origen con el mismo rol, solo HIKARI y solo flujos Label 0. Se restringe al responp
  dominante del train y a hosts con ≥ 500 flujos de train en ese subconjunto; clasificación entre
  hosts individuales, sin "other". Si quedan menos de 3 hosts, se omite y se informa.

Extras por objetivo y dataset (escenario S2, semilla 42):
- matriz de confusión en el test;
- importancia por permutación de familias de variables en una partición de validación del TRAIN
  (80/20, estratificada por la clase, semilla 42; 5 repeticiones por familia).
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from _common import add_datasets_arg, setup
from src.attack import ATTACKER_SEEDS as SEEDS, attack_metrics, attacker
from src.data import load_raw, make_predictors, make_target
from src.families import families
from src.scenarios import build_scenario, iana_port_range
from src.splits import load_split

SCENARIOS = [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]
PERM_REPEATS = 5
A3_MIN_FLOWS = 500
PORT_COLUMN = {"CICIDS2017": "Destination Port"}   # HIKARI: responp


def topk_target(values, train_rows, k):
    """Clases = top-k del train (por frecuencia) + "other". Devuelve etiquetas str y orden de clases."""
    v = pd.Series(values).astype(str).to_numpy()
    top = pd.Series(v[train_rows]).value_counts().index[:k]
    labels = np.where(np.isin(v, top), v, "other")
    order = pd.Series(labels[train_rows]).value_counts().index.tolist()
    return labels, order


def build_tasks(dataset, df, X, y, train_idx, test_idx):
    """Cada tarea: objetivo, filas de train/test (posiciones en el dataset), etiquetas y columna objetivo."""
    n = len(df)
    is_train = np.zeros(n, bool)
    is_train[train_idx] = True
    is_test = np.zeros(n, bool)
    is_test[test_idx] = True
    tasks, notes = [], {}

    port_col = PORT_COLUMN.get(dataset, "responp")
    labels, order = topk_target(X[port_col].to_numpy(), train_idx, 20)
    tasks.append({"objective": "A1_destination_port", "target_col": port_col, "labels": labels, "classes": order,
                  "train": train_idx, "test": test_idx})

    if dataset.startswith("HIKARI"):
        l0 = y == 0
        tr0, te0 = np.flatnonzero(is_train & l0), np.flatnonzero(is_test & l0)
        host = df["originh"].astype(str).to_numpy()
        labels, order = topk_target(host, tr0, 10)
        tasks.append({"objective": "A2_source_host", "target_col": None, "labels": labels, "classes": order,
                      "train": tr0, "test": te0})

        dominant = int(pd.Series(X["responp"].to_numpy()[tr0]).value_counts().index[0])
        same_role = l0 & (X["responp"].to_numpy() == dominant)
        counts = pd.Series(host[is_train & same_role]).value_counts()
        hosts = counts[counts >= A3_MIN_FLOWS].index.tolist()
        notes["A3"] = {"dominant_responp": dominant, "hosts_with_min_flows": len(hosts),
                       "host_train_flows": counts[counts >= A3_MIN_FLOWS].to_dict(),
                       "hosts_below_threshold": int((counts < A3_MIN_FLOWS).sum())}
        if len(hosts) >= 3:
            keep = same_role & np.isin(host, hosts)
            tasks.append({"objective": "A3_source_host_same_role", "target_col": None, "labels": host,
                          "classes": counts[hosts].sort_values(ascending=False).index.tolist(),
                          "train": np.flatnonzero(is_train & keep), "test": np.flatnonzero(is_test & keep)})
        else:
            notes["A3"]["skipped"] = "menos de 3 hosts con >= 500 flujos: requiere alternativa"
    return tasks, notes


def permutation_by_family(cfg, dataset, Xf, labels, classes, rows):
    """Importancia de familias en una partición de validación del train (no usa el test)."""
    y_rows = labels[rows]
    counts = pd.Series(y_rows).value_counts()
    strat = y_rows if counts.min() >= 2 else None
    fit_pos, val_pos = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42, stratify=strat)
    m = attacker(cfg, 42)
    m.fit(Xf.iloc[rows[fit_pos]], y_rows[fit_pos])
    Xv, yv = Xf.iloc[rows[val_pos]], y_rows[val_pos]
    base = f1_score(yv, m.predict(Xv), labels=classes, average="macro", zero_division=0)
    base_acc = accuracy_score(yv, m.predict(Xv))
    rng = np.random.default_rng(42)
    out = []
    for fam, cols in families(list(Xf.columns), dataset).items():
        d_f1, d_acc = [], []
        for _ in range(PERM_REPEATS):
            Xp = Xv.copy()
            Xp[cols] = Xv[cols].to_numpy()[rng.permutation(len(Xv))]
            p = m.predict(Xp)
            d_f1.append(base - f1_score(yv, p, labels=classes, average="macro", zero_division=0))
            d_acc.append(base_acc - accuracy_score(yv, p))
        out.append({"family": fam, "n_columns": len(cols), "macro_f1_drop_mean": float(np.mean(d_f1)),
                    "macro_f1_drop_std": float(np.std(d_f1)), "accuracy_drop_mean": float(np.mean(d_acc)),
                    "validation_macro_f1": base, "validation_accuracy": base_acc, "validation_rows": len(val_pos)})
    return out


def main():
    args, cfg, out = setup("F2: inferencia de atributos", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]))
    out = out / "F2_attribute_inference"
    out.mkdir(exist_ok=True)
    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        train_idx, test_idx = load_split(cfg, dataset)
        tasks, notes = build_tasks(dataset, df, X, y, train_idx, test_idx)
        (out / f"{dataset}_notes.json").write_text(json.dumps(notes, indent=2, default=int), encoding="utf-8")
        print(dataset, json.dumps(notes, default=int), flush=True)
        rows, perm_rows, cache = [], [], {}
        for task in tasks:
            obj, labels, classes = task["objective"], task["labels"], task["classes"]
            tr, te = task["train"], task["test"]
            majority = classes[0]
            y_te = labels[te]
            for name, internal in SCENARIOS:
                Xs = build_scenario(internal, X, cfg, dataset)
                published = list(Xs.columns)
                common = {"dataset": dataset, "objective": obj, "scenario": name, "n_train": len(tr),
                          "n_test": len(te), "n_classes": len(classes)}
                if task["target_col"] is not None and task["target_col"] in published:
                    rows.append({**common, "attack": "exposed_directly", "seed": "-", "n_features": len(published),
                                 "accuracy": 1.0, "macro_f1": 1.0, "balanced_accuracy": 1.0,
                                 "majority_baseline": float(np.mean(y_te == majority)), "advantage": 1.0})
                feats = [c for c in published if c != task["target_col"]]
                key = (obj, tuple(feats))
                if key in cache:
                    for r in cache[key]["rows"]:
                        rows.append({**r, **common, "same_feature_set_as": cache[key]["scenario"]})
                    continue
                Xf = Xs[feats]
                del Xs
                these = []
                for seed in SEEDS:
                    m = attacker(cfg, seed)
                    t0 = time.perf_counter()
                    m.fit(Xf.iloc[tr], labels[tr])
                    pred = m.predict(Xf.iloc[te])
                    r = {**common, "attack": "inference", "seed": seed, "n_features": len(feats),
                         **attack_metrics(y_te, pred, classes, majority), "fit_seconds": time.perf_counter() - t0}
                    if obj.startswith("A1") and name == "S1b":
                        # Referencia: puerto más frecuente del train dentro de cada rango IANA
                        port = X[task["target_col"]]
                        rng_all = iana_port_range(port).to_numpy()
                        best = pd.Series(labels[tr]).groupby(rng_all[tr]).agg(lambda s: s.value_counts().index[0])
                        r["iana_range_only_baseline_accuracy"] = float(np.mean(y_te == best.reindex(rng_all[te]).to_numpy()))
                    these.append(r)
                    rows.append(r)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                    if seed == 42 and name == "S2":
                        cm = confusion_matrix(y_te, pred, labels=classes)
                        pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                                     columns=[f"pred_{c}" for c in classes]).to_csv(
                            out / f"{dataset}_{obj}_S2_confusion_seed42.csv")
                    del m
                cache[key] = {"scenario": name, "rows": [{k: v for k, v in r.items()
                                                          if k not in common} for r in these]}
                if name == "S2":
                    for p in permutation_by_family(cfg, dataset, Xf, labels, classes, tr):
                        perm_rows.append({"dataset": dataset, "objective": obj, "scenario": "S2", **p})
                del Xf
                pd.DataFrame(rows).to_csv(out / f"{dataset}_attribute_inference_runs.csv", index=False)
            pd.DataFrame(perm_rows).to_csv(out / f"{dataset}_S2_family_permutation.csv", index=False)
        del df, X


if __name__ == "__main__":
    main()
