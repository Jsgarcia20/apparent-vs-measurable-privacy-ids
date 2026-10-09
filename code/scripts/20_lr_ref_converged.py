"""Decisión 5 (2026-10-07): referencia LR-ref convergida para S3c.

sklearn LogisticRegression sin DP, mismos datos normalizados por dominio y mismos hiperparámetros que la
LR-ref original (C = 1,0; tol = 1e-4; lbfgs; L2; fit_intercept = True; class_weight = None), salvo
max_iter = 2000. Si aun así hubiera ConvergenceWarning, se registra y se prueba 5000 y 10000 hasta que
converja sin aviso (se informa el valor usado). Variantes S3c (variables de S0) y S3c-min (de S2);
CICIDS sobre la submuestra de 500.000 filas.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from _common import add_datasets_arg, setup
from src.dp_local import domain_normalize, load_bounds
from src.s3_common import prepare, utility_metrics

MAX_ITER_SEQUENCE = [2000, 5000, 10000]
VARIANTS = {"S3c": "S3a", "S3c-min": "S3a-min"}


def main():
    args, cfg, out = setup("LR-ref convergida (decisión 5)", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]))
    out = out / "S3c"
    out.mkdir(exist_ok=True)
    rows = []
    for dataset in args.datasets:
        df, X, y, tr, te, cols = prepare(cfg, dataset)
        del df
        thr = cfg["datasets"][dataset]["decision_threshold"]
        for v, key in VARIANTS.items():
            c = cols[key]
            Xn = domain_normalize(X.iloc[np.concatenate([tr, te])][c], load_bounds(cfg, dataset))
            Xtr, Xte = Xn.iloc[:len(tr)].to_numpy(), Xn.iloc[len(tr):].to_numpy()
            for max_iter in MAX_ITER_SEQUENCE:
                m = LogisticRegression(C=1.0, tol=1e-4, max_iter=max_iter, fit_intercept=True,
                                       solver="lbfgs", penalty="l2", class_weight=None)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter()
                    m.fit(Xtr, y[tr])
                    secs = time.perf_counter() - t0
                conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
                r = {"dataset": dataset, "variant": f"LR-ref ({v})", "max_iter": max_iter,
                     "n_iter": int(np.max(m.n_iter_)), "converged_without_warning": len(conv) == 0,
                     "warnings": " | ".join(sorted({f"{x.category.__name__}: {str(x.message)[:120]}" for x in w})),
                     **utility_metrics(y[te], m.predict_proba(Xte)[:, 1], thr), "fit_seconds": secs,
                     "n_features": len(c)}
                rows.append(r)
                print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
                pd.DataFrame(rows).to_csv(out / "lr_ref_converged.csv", index=False)
                if not conv:
                    break


if __name__ == "__main__":
    main()
