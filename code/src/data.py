"""Carga de parquets limpios y definición de X / y (celda 14 del notebook, sin cambios de lógica)."""
import pandas as pd

from src.config import path_of


def load_raw(cfg, dataset):
    ds = cfg["datasets"][dataset]
    return pd.read_parquet(path_of(cfg, ds["parquet_key"]))


def make_target(df, dataset, cfg=None):
    if cfg is not None:
        kind = cfg["datasets"][dataset]["target"]
    else:
        kind = {"CICIDS2017": "benign_string", "HIKARI2021": "binary"}.get(dataset)
    if kind == "benign_string":
        # BENIGN = 0, cualquier ataque = 1
        return df["Label"].astype(str).str.strip().str.upper().ne("BENIGN").astype(int)
    if kind == "binary":
        # HIKARI ya trae Label binario 0/1
        return df["Label"].astype(int)
    raise ValueError(f"Dataset desconocido: {dataset}")


def make_predictors(df, cfg, dataset):
    # Se conserva el orden de columnas del parquet (orden canónico)
    return df.drop(columns=cfg["datasets"][dataset]["exclude_columns"])


def load_xy(cfg, dataset):
    df = load_raw(cfg, dataset)
    return make_predictors(df, cfg, dataset), make_target(df, dataset, cfg)
