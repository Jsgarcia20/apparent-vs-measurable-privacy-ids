"""Lectura de las taxonomías de privacidad vigentes (generadas en la Fase 3)."""
import pandas as pd

from src.config import path_of


def load_taxonomy(cfg, dataset):
    return pd.read_csv(path_of(cfg, cfg["datasets"][dataset]["taxonomy_key"]))


def features_with_relevance(taxonomy, relevances):
    """Conjunto de features cuya privacy_relevance está en `relevances`.

    Solo usa la taxonomía: no usa target, importancia, SHAP ni rendimiento.
    """
    return set(taxonomy.loc[taxonomy["privacy_relevance"].isin(relevances), "feature"])
