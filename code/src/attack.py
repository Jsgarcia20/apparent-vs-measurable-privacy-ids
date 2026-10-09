"""Atacante común de la Fase 11 (inferencia de atributos) y sus métricas."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ATTACKER_SEEDS = [42, 43, 44]


def attacker(cfg, seed):
    """RF con los hiperparámetros congelados, pero class_weight=None y la semilla del atacante."""
    rf = cfg["random_forest"]
    return RandomForestClassifier(n_estimators=100, max_depth=rf["max_depth"],
                                  min_samples_split=rf["min_samples_split"], min_samples_leaf=rf["min_samples_leaf"],
                                  max_features=rf["max_features"], bootstrap=rf["bootstrap"], class_weight=None,
                                  random_state=seed, n_jobs=rf["n_jobs"])


def attack_metrics(y_true, y_pred, classes, majority):
    acc = accuracy_score(y_true, y_pred)
    base = float(np.mean(y_true == majority))
    return {"accuracy": acc,
            "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "majority_baseline": base,
            "advantage": (acc - base) / (1 - base) if base < 1 else np.nan}
