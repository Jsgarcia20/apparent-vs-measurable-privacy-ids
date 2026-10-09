"""Escenario S3 (privacidad diferencial a nivel de registro).

Código TRASLADADO SIN CAMBIOS desde las celdas 20, 21 y 22 del notebook
`analysis/fase 1.ipynb` (extraído por AST). No se ejecuta en esta etapa.
Riesgos conocidos (R2–R5): límites y C calculados
con todo el dataset; train con ruido vs. test sin ruido.
"""
import numpy as np
import pandas as pd
from diffprivlib.mechanisms import GaussianAnalytic

DP_RANDOM_SEED = 42


def create_public_benchmark_bounds(X):
    rows = []
    for feature in X.columns:
        lower = float(X[feature].min())
        upper = float(X[feature].max())
        rows.append({'feature': feature, 'lower_bound': lower, 'upper_bound': upper, 'is_constant': lower == upper, 'bound_source': 'public_benchmark_metadata'})
    return pd.DataFrame(rows)


def normalize_with_bounds(X, bounds):
    X_norm = X.astype(np.float32).copy()
    for row in bounds.itertuples():
        feature = row.feature
        lower = row.lower_bound
        upper = row.upper_bound
        if upper == lower:
            X_norm[feature] = 0.0
        else:
            X_norm[feature] = (X_norm[feature] - lower) / (upper - lower)
            X_norm[feature] = X_norm[feature].clip(lower=0.0, upper=1.0)
    return X_norm


def compute_norm_percentiles(X, bounds, dataset_name):
    X_norm = normalize_with_bounds(X, bounds)
    norms = np.linalg.norm(X_norm.to_numpy(dtype=np.float32), axis=1)
    percentiles = [50, 75, 90, 95, 97.5, 99, 99.5, 100]
    rows = []
    for p in percentiles:
        value = float(np.percentile(norms, p))
        rows.append({'dataset': dataset_name, 'percentile': p, 'l2_norm': value, 'candidate_C': value, 'replace_one_sensitivity': 2.0 * value, 'percentage_above_C': float(100.0 * np.mean(norms > value))})
    return (pd.DataFrame(rows), norms)


def l2_clip_records(X, clip_bound):
    if clip_bound <= 0:
        raise ValueError('clip_bound must be > 0')
    X_array = X.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(X_array, axis=1)
    safe_norms = np.maximum(norms, 1e-12)
    scale_factors = np.minimum(1.0, clip_bound / safe_norms).astype(np.float32)
    X_clipped_array = X_array * scale_factors[:, np.newaxis]
    norms_after = np.linalg.norm(X_clipped_array, axis=1)
    X_clipped = pd.DataFrame(X_clipped_array, columns=X.columns, index=X.index)
    metadata = {'rows': int(len(X)), 'clip_bound': float(clip_bound), 'rows_clipped': int(np.sum(norms > clip_bound)), 'fraction_clipped': float(np.mean(norms > clip_bound)), 'percentage_clipped': float(100.0 * np.mean(norms > clip_bound)), 'mean_norm_before': float(norms.mean()), 'median_norm_before': float(np.median(norms)), 'max_norm_before': float(norms.max()), 'mean_norm_after': float(norms_after.mean()), 'max_norm_after': float(norms_after.max())}
    return (X_clipped, metadata)


def get_analytic_gaussian_sigma(epsilon, delta, sensitivity):
    mechanism = GaussianAnalytic(epsilon=float(epsilon), delta=float(delta), sensitivity=float(sensitivity), random_state=DP_RANDOM_SEED)
    sigma = float(mechanism._scale)
    return sigma


def create_dp_training_data_gaussian(X, train_idx, bounds, epsilon, delta, clip_bound, random_seed=42):
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0')
    if not 0 < delta < 1:
        raise ValueError('delta must be between 0 and 1')
    X_train = X.iloc[train_idx].copy()
    X_train_norm = normalize_with_bounds(X_train, bounds)
    (X_train_clipped, clipping_metadata) = l2_clip_records(X_train_norm, clip_bound=clip_bound)
    sensitivity = 2.0 * clip_bound
    sigma = get_analytic_gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=X_train_clipped.shape).astype(np.float32)
    X_train_dp_array = X_train_clipped.to_numpy(dtype=np.float32) + noise
    X_train_dp_array = np.clip(X_train_dp_array, 0.0, 1.0)
    X_train_dp = pd.DataFrame(X_train_dp_array, columns=X_train_clipped.columns, index=X_train_clipped.index)
    zero_fraction = float(np.mean(X_train_dp_array == 0.0))
    one_fraction = float(np.mean(X_train_dp_array == 1.0))
    interior_fraction = float(np.mean((X_train_dp_array > 0.0) & (X_train_dp_array < 1.0)))
    metadata = {'epsilon': float(epsilon), 'delta': float(delta), 'features': int(X_train_dp.shape[1]), 'clip_bound': float(clip_bound), 'l2_sensitivity': float(sensitivity), 'gaussian_sigma': float(sigma), 'rows': int(len(X_train_dp)), 'rows_clipped': clipping_metadata['rows_clipped'], 'fraction_clipped': clipping_metadata['fraction_clipped'], 'percentage_clipped': clipping_metadata['percentage_clipped'], 'mean_norm_before': clipping_metadata['mean_norm_before'], 'median_norm_before': clipping_metadata['median_norm_before'], 'max_norm_before': clipping_metadata['max_norm_before'], 'max_norm_after': clipping_metadata['max_norm_after'], 'zero_fraction_after_dp': zero_fraction, 'one_fraction_after_dp': one_fraction, 'interior_fraction_after_dp': interior_fraction, 'privacy_unit': 'one_network_flow_vector'}
    return (X_train_dp, metadata)
