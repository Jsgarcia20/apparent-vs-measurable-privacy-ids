"""Métricas binarias (celda 23) y discriminación independiente del umbral (celdas 30–32)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    fnr = fn / (fn + tp) if fn + tp > 0 else np.nan
    return {
        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def discrimination_metrics(y_true, probabilities):
    """ROC-AUC y PR-AUC. Como en las celdas 30–32, las probabilidades se pasan a float32
    antes del cálculo; también se devuelve el valor con float64 sin redondear."""
    y_true = np.asarray(y_true)
    p32 = np.asarray(probabilities).astype(np.float32).astype(np.float64)
    p64 = np.asarray(probabilities, dtype=np.float64)
    return {
        "roc_auc": float(roc_auc_score(y_true, p32)),
        "pr_auc": float(average_precision_score(y_true, p32)),
        "roc_auc_float64": float(roc_auc_score(y_true, p64)),
        "pr_auc_float64": float(average_precision_score(y_true, p64)),
    }
