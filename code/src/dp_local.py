"""S3 (diseño 2026-10-07): normalización por dominio y mecanismo gaussiano analítico local.

- Normalización con los límites congelados de S3-A:
      x' = log1p(clip(x - lo, 0, hi - lo)) / log1p(hi - lo)  ∈ [0, 1]
- DP local por registro (adyacencia replace-one): cada vector normalizado está en [0, 1]^d, así que
  la sensibilidad L2 es Δ₂ = √d sin necesidad de clipping L2.
- Mecanismo gaussiano analítico (Balle y Wang, ICML 2018, Teorema 8): M(x) = x + N(0, σ² I) es
  (ε, δ)-DP si y solo si
      Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ) ≤ δ.
  σ es el menor valor que cumple la desigualdad; se busca por bisección en log σ y se evalúa en
  escala logarítmica para que sea estable con ε grande (hasta 1000).
- Sin recorte a [0, 1] después del ruido. Los labels no se perturban.
"""
import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from src.config import path_of


def analytic_gaussian_log_delta(sigma, epsilon, sensitivity):
    """log δ(σ) = log[Φ(a − b) − e^ε Φ(−a − b)], con a = Δ/(2σ) y b = εσ/Δ."""
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    l1 = log_ndtr(a - b)
    l2 = epsilon + log_ndtr(-a - b)
    if l2 >= l1:            # no ocurre en teoría (δ(σ) ≥ 0); protección numérica
        return -np.inf
    return l1 + np.log1p(-np.exp(l2 - l1))


def analytic_gaussian_sigma(epsilon, delta, sensitivity, iters=300):
    """Menor σ tal que δ(σ) ≤ δ (δ(σ) es decreciente en σ)."""
    if epsilon <= 0 or not (0 < delta < 1) or sensitivity <= 0:
        raise ValueError("epsilon > 0, 0 < delta < 1 y sensibilidad > 0")
    log_target = np.log(delta)
    lo, hi = np.log(1e-12 * sensitivity), np.log(1e12 * sensitivity)
    assert analytic_gaussian_log_delta(np.exp(hi), epsilon, sensitivity) <= log_target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if analytic_gaussian_log_delta(np.exp(mid), epsilon, sensitivity) <= log_target:
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def classical_gaussian_sigma(epsilon, delta, sensitivity):
    """Cota clásica de Dwork y Roth (válida para ε < 1); solo se usa como comprobación (σ_analítico ≤ σ_clásico)."""
    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def load_bounds(cfg, dataset):
    """Límites congelados. Las versiones de HIKARI comparten límites (mismas definiciones)."""
    d = path_of(cfg, "domain_bounds_dir")
    b = pd.read_csv(d / f"{dataset}_domain_bounds.csv")
    return b.set_index("feature")[["lo", "hi"]]


def domain_normalize(X, bounds):
    """Normalización por dominio columna a columna; conserva el orden y los nombres de las columnas."""
    out = {}
    for c in X.columns:
        lo, hi = bounds.loc[c, "lo"], bounds.loc[c, "hi"]
        v = X[c].to_numpy(dtype=np.float64)
        out[c] = np.log1p(np.clip(v - lo, 0.0, hi - lo)) / np.log1p(hi - lo)
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def add_gaussian_noise(Xn, sigma, seed):
    """Ruido gaussiano isótropo por registro (sin recorte posterior)."""
    rng = np.random.default_rng(seed)
    noisy = Xn.to_numpy(dtype=np.float64) + rng.normal(0.0, sigma, size=Xn.shape)
    return pd.DataFrame(noisy, index=Xn.index, columns=Xn.columns)
