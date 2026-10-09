"""Escenarios de privacidad sobre las features (celdas 19 y S2 reconstruida).

Orden canónico de columnas: el del dataset. Ningún escenario reordena columnas.
"""
import numpy as np
import pandas as pd

from src.taxonomy import features_with_relevance, load_taxonomy


def s0(X, cfg, dataset):
    return X


def s1(X, cfg, dataset):
    """Supresión de metadatos de conexión P1."""
    drop = cfg["datasets"][dataset]["s1_drop"]
    missing = [c for c in drop if c not in X.columns]
    assert not missing, f"Features P1 ausentes en S0: {missing}"
    return X.drop(columns=drop).copy()


IANA_SUFFIX = "_iana"


def iana_port_range(ports):
    """Rango IANA del puerto, definido sin usar los datos: 0 = 0–1023, 1 = 1024–49151, 2 = 49152–65535."""
    p = ports.to_numpy()
    assert ((p >= 0) & (p <= 65535)).all(), "Puerto fuera de 0–65535"
    return pd.Series(np.select([p <= 1023, p <= 49151], [0, 1], default=2).astype(np.int8), index=ports.index)


def s1b(X, cfg, dataset):
    """Generalización de puertos: cada columna de s1_drop se sustituye, en su misma posición,
    por su rango IANA (columna renombrada con el sufijo _iana)."""
    ports = cfg["datasets"][dataset]["s1_drop"]
    missing = [c for c in ports if c not in X.columns]
    assert not missing, f"Columnas de puerto ausentes en S0: {missing}"
    Xg = X.copy()
    for c in ports:
        Xg[c] = iana_port_range(X[c])
    return Xg.rename(columns={c: c + IANA_SUFFIX for c in ports})


def s2(X, cfg, dataset):
    """Minimización guiada por taxonomía: conserva SOLO privacy_relevance == LOW.

    Se filtra respetando el orden de columnas de X (no el de la taxonomía).
    """
    taxonomy = load_taxonomy(cfg, dataset)
    keep = features_with_relevance(taxonomy, cfg["scenarios"]["S2"]["keep_privacy_relevance"])
    columns = [c for c in X.columns if c in keep]
    return X[columns].copy()


SCENARIOS = {
    "S0": s0,
    "S1": s1,
    "S2": s2,
    "S1b": s1b,
    # "S3": ver src/dp.py (no se ejecuta en este paso)
}

def build_scenario(name, X, cfg, dataset):
    Xs = SCENARIOS[name](X, cfg, dataset)
    expected = cfg["datasets"][dataset]["expected_features"][name]
    assert Xs.shape[1] == expected, f"{dataset} {name}: {Xs.shape[1]} features, se esperaban {expected}"
    assert Xs.index.equals(X.index)
    return Xs
