"""Utilidades compartidas por los escenarios S3 (S3-B, S3c, M1): submuestra fija de CICIDS, preparación
de datos y variables por variante, métricas de utilidad y tareas de ataque. Trasladadas sin cambios
desde scripts/14_s3_dp.py.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import path_of
from src.data import load_raw, make_predictors, make_target
from src.metrics import calculate_binary_metrics
from src.scenarios import build_scenario
from src.splits import load_split

SUBSAMPLE = {"CICIDS2017": 500_000}
VARIANTS = {"S3a": "S0", "S3a-min": "S2"}
A3_HOSTS = ["103.255.15.23", "103.255.15.150"]


def training_rows(cfg, dataset, y):
    """Train completo o submuestra fija (CICIDS). La submuestra se guarda una vez y luego solo se verifica."""
    train_idx, test_idx = load_split(cfg, dataset)
    if dataset not in SUBSAMPLE:
        return train_idx, test_idx
    path = path_of(cfg, "splits_dir") / f"{dataset}_train_sub{SUBSAMPLE[dataset] // 1000}k_indices.npy"
    sub, _ = train_test_split(train_idx, train_size=SUBSAMPLE[dataset], random_state=42, stratify=y[train_idx])
    if path.exists():
        assert np.array_equal(np.load(path), sub), f"La submuestra guardada no coincide: {path}"
    else:
        np.save(path, sub)
    return sub, test_idx


def prepare(cfg, dataset):
    df = load_raw(cfg, dataset)
    X = make_predictors(df, cfg, dataset)
    y = make_target(df, dataset, cfg).to_numpy()
    tr, te = training_rows(cfg, dataset, y)
    cols = {v: list(build_scenario(s, X.iloc[:1], cfg, dataset).columns) for v, s in VARIANTS.items()}
    return df, X, y, tr, te, cols


def utility_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(np.int8)
    m = calculate_binary_metrics(y_true, pred)
    return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"], "fpr": m["fpr"],
            "roc_auc": float(roc_auc_score(y_true, proba)), "pr_auc": float(average_precision_score(y_true, proba))}


def attack_tasks(dataset, df, X, y, tr, te):
    port_col = "Destination Port" if dataset.startswith("CICIDS") else "responp"
    v = X[port_col].astype(str).to_numpy()
    top = pd.Series(v[tr]).value_counts().index[:20]
    lab = np.where(np.isin(v, top), v, "other")
    tasks = [("A1_destination_port", port_col, lab, tr, te)]
    if dataset.startswith("HIKARI"):
        host = df["originh"].astype(str).to_numpy()
        l0 = y == 0
        tr0, te0 = tr[l0[tr]], te[l0[te]]
        top = pd.Series(host[tr0]).value_counts().index[:10]
        tasks.append(("A2_source_host", None, np.where(np.isin(host, top), host, "other"), tr0, te0))
        if dataset == "HIKARI2021_v2_main":
            for name, port in [("A3_DNS", 53), ("A3_443", 443)]:
                keep = l0 & (X["responp"].to_numpy() == port) & np.isin(host, A3_HOSTS)
                tasks.append((name, None, host, tr[keep[tr]], te[keep[te]]))
    return tasks
