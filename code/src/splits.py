"""Splits fijos train/test.

REGLA: los splits NUNCA se regeneran. Este módulo solo carga los .npy existentes
y los verifica. `regenerated_matches_saved` recalcula train_test_split en memoria
únicamente para comprobar que los archivos coinciden con el protocolo
(80/20 estratificado, random_state=42); no escribe nada.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import path_of


def split_paths(cfg, dataset):
    files = cfg["datasets"][dataset]["split_files"]
    d = path_of(cfg, "splits_dir")
    return d / files["train"], d / files["test"]


def load_split(cfg, dataset):
    train_path, test_path = split_paths(cfg, dataset)
    return np.load(train_path), np.load(test_path)


def verify_split(train_idx, test_idx, y, expected_train, expected_test):
    y = np.asarray(y)
    n = len(y)
    report = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "expected_train": int(expected_train),
        "expected_test": int(expected_test),
        "sizes_ok": len(train_idx) == expected_train and len(test_idx) == expected_test,
        "overlap": int(len(np.intersect1d(train_idx, test_idx))),
        "train_unique": bool(len(np.unique(train_idx)) == len(train_idx)),
        "test_unique": bool(len(np.unique(test_idx)) == len(test_idx)),
        "full_coverage": bool(len(np.union1d(train_idx, test_idx)) == n and n == len(train_idx) + len(test_idx)),
        "train_attack_rate": float(y[train_idx].mean()),
        "test_attack_rate": float(y[test_idx].mean()),
        "total_attack_rate": float(y.mean()),
    }
    report["stratification_ok"] = bool(
        abs(report["train_attack_rate"] - report["test_attack_rate"]) < 1e-4
    )
    return report


def regenerated_matches_saved(train_idx, test_idx, y, test_size, random_state):
    """Comprueba (sin escribir nada) que el split guardado es el de train_test_split(42), incluido el orden."""
    tr, te = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=y
    )
    return bool(np.array_equal(tr, train_idx)), bool(np.array_equal(te, test_idx))
