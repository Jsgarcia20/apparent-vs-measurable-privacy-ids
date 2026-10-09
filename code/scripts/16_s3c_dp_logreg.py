"""S3c — DP central con regresión logística (perturbación del objetivo; Chaudhuri et al., 2011).

Diseño registrado antes de ejecutar.
- diffprivlib.models.LogisticRegression sobre los datos normalizados por dominio, data_norm = √d,
  resto de hiperparámetros por defecto (C = 1, tol = 1e-4, max_iter = 100, fit_intercept = True).
- LR-ref: sklearn LogisticRegression sin DP con los hiperparámetros equivalentes.
- Variantes S3c (variables de S0) y S3c-min (variables de S2); ε ∈ {0,1; 0,5; 1; 5; 10; 50; 100};
  semillas 42–46 (HIKARI) y 42–44 (CICIDS, submuestra de 500.000 filas); test limpio normalizado.
- Se registran todos los avisos que emitan diffprivlib o sklearn (data_norm, convergencia…).
"""
import time
import warnings

import numpy as np
import pandas as pd
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.linear_model import LogisticRegression

from _common import add_datasets_arg, setup
from src.dp_local import domain_normalize, load_bounds
from src.s3_common import prepare, utility_metrics

EPS = [0.1, 0.5, 1, 5, 10, 50, 100]
SEEDS = {"CICIDS2017": [42, 43, 44]}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
VARIANTS = {"S3c": "S3a", "S3c-min": "S3a-min"}   # columnas de S0 y de S2 (claves de prepare())
LR_PARAMS = dict(C=1.0, tol=1e-4, max_iter=100, fit_intercept=True)


def fit_with_warnings(model, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(X, y)
        secs = time.perf_counter() - t0
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in w})
    return secs, msgs


def main():
    args, cfg, out = setup("S3c: regresión logística con DP", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]))
    out = out / "S3c"
    out.mkdir(exist_ok=True)
    for dataset in args.datasets:
        df, X, y, tr, te, cols = prepare(cfg, dataset)
        del df
        thr = cfg["datasets"][dataset]["decision_threshold"]
        rows, warn_rows = [], []
        for v, key in VARIANTS.items():
            c = cols[key]
            d = len(c)
            Xn = domain_normalize(X.iloc[np.concatenate([tr, te])][c], load_bounds(cfg, dataset))
            Xtr, Xte = Xn.iloc[:len(tr)].to_numpy(), Xn.iloc[len(tr):].to_numpy()
            ref = LogisticRegression(solver="lbfgs", penalty="l2", class_weight=None, **LR_PARAMS)
            secs, msgs = fit_with_warnings(ref, Xtr, y[tr])
            ref_m = utility_metrics(y[te], ref.predict_proba(Xte)[:, 1], thr)
            rows.append({"dataset": dataset, "variant": f"LR-ref ({v})", "epsilon": np.nan, "seed": -1,
                         "n_features": d, "data_norm": np.nan, **ref_m, "fit_seconds": secs,
                         "n_iter": int(np.max(ref.n_iter_)), "warnings": " | ".join(msgs)})
            warn_rows += [{"dataset": dataset, "variant": f"LR-ref ({v})", "epsilon": np.nan, "seed": -1, "warning": m} for m in msgs]
            for eps in EPS:
                for seed in SEEDS.get(dataset, DEFAULT_SEEDS):
                    m = DPLogisticRegression(epsilon=eps, data_norm=float(np.sqrt(d)), random_state=seed, **LR_PARAMS)
                    secs, msgs = fit_with_warnings(m, Xtr, y[tr])
                    met = utility_metrics(y[te], m.predict_proba(Xte)[:, list(m.classes_).index(1)], thr)
                    rel = {f"rel_loss_{k}": (ref_m[k] - met[k]) / ref_m[k] if ref_m[k] else np.nan
                           for k in ["f1", "pr_auc", "roc_auc"]}
                    r = {"dataset": dataset, "variant": v, "epsilon": eps, "seed": seed, "n_features": d,
                         "data_norm": float(np.sqrt(d)), **met, **rel, "fit_seconds": secs,
                         "n_iter": int(np.max(m.n_iter_)), "warnings": " | ".join(msgs)}
                    rows.append(r)
                    warn_rows += [{"dataset": dataset, "variant": v, "epsilon": eps, "seed": seed, "warning": w} for w in msgs]
                    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k != "warnings"},
                          ("AVISOS: " + " | ".join(msgs)) if msgs else "", flush=True)
                    pd.DataFrame(rows).to_csv(out / f"{dataset}_s3c_runs.csv", index=False)
        pd.DataFrame(warn_rows, columns=["dataset", "variant", "epsilon", "seed", "warning"]).to_csv(
            out / f"{dataset}_s3c_warnings.csv", index=False)


if __name__ == "__main__":
    main()
