"""Ajuste de LR con DP (diffprivlib) según la decisión 5b (2026-10-07): max_iter = 2000 y, si aún hay
ConvergenceWarning, 5000 y luego 10000. Devuelve el modelo y el registro de convergencia.
Un ajuste que no converge ni con 10000 se conserva y se marca como "no convergido"; nunca se descarta.
"""
import time
import warnings

import numpy as np
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.exceptions import ConvergenceWarning

MAX_ITER_SEQUENCE = [2000, 5000, 10000]
LR_FIXED = dict(C=1.0, tol=1e-4, fit_intercept=True)


def fit_dp_lr_converged(X, y, epsilon, data_norm, seed):
    log = {}
    total = 0.0
    for max_iter in MAX_ITER_SEQUENCE:
        m = DPLogisticRegression(epsilon=epsilon, data_norm=float(data_norm), random_state=seed,
                                 max_iter=max_iter, **LR_FIXED)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            m.fit(X, y)
            total += time.perf_counter() - t0
        conv_warn = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        other = sorted({f"{x.category.__name__}: {str(x.message)[:120]}" for x in w if x not in conv_warn})
        log = {"max_iter_used": max_iter, "n_iter": int(np.max(m.n_iter_)), "converged": len(conv_warn) == 0,
               "convergence_status": "convergido" if not conv_warn else "no convergido",
               "other_warnings": " | ".join(other), "fit_seconds_total": total}
        if not conv_warn:
            break
    return m, log
