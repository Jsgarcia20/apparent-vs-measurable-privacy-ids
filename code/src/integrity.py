"""Utilidades de integridad: hash de filas, conflictos de etiqueta, copias train/test y métricas por subconjunto.

Las filas se identifican por un hash de 64 bits de sus valores (pandas.util.hash_pandas_object);
la probabilidad de colisión con 2,5 M de filas es del orden de 1e-7.
"""
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.metrics import average_precision_score

from src.metrics import calculate_binary_metrics


def row_hash(df):
    return hash_pandas_object(df, index=False).to_numpy()


def group_stats(keys):
    """Número de grupos con más de una fila y máscara de las filas que pertenecen a esos grupos."""
    counts = pd.Series(keys).map(pd.Series(keys).value_counts())
    in_group = counts.to_numpy() > 1
    n_groups = int(pd.Series(keys[in_group]).nunique())
    return n_groups, in_group


def conflict_mask(feature_keys, labels):
    nunique = pd.DataFrame({"k": feature_keys, "y": labels}).groupby("k")["y"].transform("nunique")
    return nunique.to_numpy() > 1


def test_has_train_copy(keys, train_idx, test_idx):
    train_keys = set(keys[train_idx].tolist())
    return np.fromiter((v in train_keys for v in keys[test_idx]), dtype=bool, count=len(test_idx))


def subset_metrics(y, pred, proba, mask):
    m = calculate_binary_metrics(y[mask], pred[mask])
    pr_auc = float(average_precision_score(y[mask], proba[mask].astype(np.float32).astype(np.float64))) \
        if 0 < y[mask].sum() < mask.sum() else np.nan
    return {"rows": int(mask.sum()), "attacks": int(y[mask].sum()), "f1": m["f1"], "recall": m["recall"],
            "precision": m["precision"], "fpr": m["fpr"], "pr_auc": pr_auc,
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]}
