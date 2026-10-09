"""M1 — Inferencia de membresía por umbral de confianza (Yeom et al., 2018); amenaza T2.

Diseño:
- Escenarios: S0, S1a, S1b, S2 (RF con la configuración validada, random_state = 42); S3a y S3a-min con
  ε ∈ {10, 100, 1000} (semilla de ruido 42); S3b (RF de diffprivlib) y S3c (LR de diffprivlib) con todos
  sus ε (semilla 42).
- Datos de entrenamiento: HIKARI v2_main y v2_pairs, train completo. CICIDS: train completo para S0–S2 y
  la submuestra fija de 500.000 filas para S3a/S3a-min/S3b/S3c.
- Miembros: 10.000 flujos al azar de las filas de entrenamiento del modelo.
- No miembros: 10.000 flujos del test, EXCLUYENDO los que tienen copia exacta en el train (de ese modelo)
  con las columnas del escenario (columnas originales sin ruido). Se reporta cuántos se excluyeron.
- En S3a/S3a-min el atacante consulta con el registro original (limpio, normalizado por dominio).
- Puntuación: probabilidad asignada a la clase verdadera. Métricas: ROC-AUC (miembro = positivo) y
  TPR con FPR = 1 % (Carlini et al., 2022). 3 semillas de muestreo (42, 43, 44) por modelo.
- Con --s3c-converged (decisión 5b): solo S3c y S3c-min, con las LR con DP CONVERGIDAS
  (max_iter 2000 → 5000 → 10000), escrito en m1_s3c_converged.csv. La ejecución por defecto (S3c con
  max_iter = 100) queda como sensibilidad.
"""
import time

import numpy as np
import pandas as pd
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from diffprivlib.models import RandomForestClassifier as DPRandomForest
from sklearn.metrics import roc_auc_score, roc_curve

from _common import add_datasets_arg, setup
from src.data import load_raw, make_predictors, make_target
from src.dp_local import add_gaussian_noise, analytic_gaussian_sigma, domain_normalize, load_bounds
from src.dp_lr import fit_dp_lr_converged
from src.integrity import row_hash
from src.models import create_random_forest
from src.s3_common import SUBSAMPLE, training_rows
from src.scenarios import build_scenario
from src.splits import load_split

N_SAMPLE = 10_000
SAMPLE_SEEDS = [42, 43, 44]
EPS_S3A = [10, 100, 1000]
EPS_S3B = [0.1, 0.5, 1, 5, 10]
EPS_S3C = [0.1, 0.5, 1, 5, 10, 50, 100]
NOISE_SEED = 42
LR_PARAMS = dict(C=1.0, tol=1e-4, max_iter=100, fit_intercept=True)


def tpr_at_fpr(y_member, score, target=0.01):
    fpr, tpr, _ = roc_curve(y_member, score)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def nonmember_pool(X_cols, train_rows, test_rows):
    """Test sin copia exacta en el train (con las columnas dadas). Devuelve (pool, nº excluidos)."""
    k = row_hash(X_cols)
    train_keys = set(k[train_rows].tolist())
    has_copy = np.fromiter((v in train_keys for v in k[test_rows]), bool, len(test_rows))
    return test_rows[~has_copy], int(has_copy.sum())


def evaluate(model_proba, Xq_member, Xq_non, y_member, y_non):
    """Puntuación = probabilidad de la clase verdadera."""
    pm = model_proba(Xq_member)
    pn = model_proba(Xq_non)
    s_m = np.where(y_member == 1, pm, 1 - pm)
    s_n = np.where(y_non == 1, pn, 1 - pn)
    score = np.concatenate([s_m, s_n])
    lab = np.concatenate([np.ones(len(s_m)), np.zeros(len(s_n))])
    return {"roc_auc": float(roc_auc_score(lab, score)), "tpr_at_fpr_1pct": tpr_at_fpr(lab, score),
            "mean_score_members": float(s_m.mean()), "mean_score_nonmembers": float(s_n.mean())}


def main():
    def extra(p):
        add_datasets_arg(p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"])
        p.add_argument("--s3c-converged", action="store_true",
                       help="solo S3c y S3c-min con LR con DP convergidas (decisión 5b)")
    args, cfg, out = setup("M1: inferencia de membresía", extra)
    out = out / "M1"
    out.mkdir(exist_ok=True)
    path = out / ("m1_s3c_converged.csv" if args.s3c_converged else "m1_runs.csv")
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["scenario"], str(r["epsilon"])) for r in rows}

    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        del df
        tr_full, te = load_split(cfg, dataset)
        tr_s3, _ = training_rows(cfg, dataset, y) if dataset in SUBSAMPLE else (tr_full, te)
        bounds = load_bounds(cfg, dataset)
        cols = {s: list(build_scenario(s, X.iloc[:1], cfg, dataset).columns) for s in ["S0", "S1", "S1b", "S2"]}

        def record(scenario, eps, train_rows, copy_cols, proba_fn, query_fn, fit_secs, extra_cols=None):
            pool, excluded = nonmember_pool(copy_cols, train_rows, te)
            for ss in SAMPLE_SEEDS:
                rng = np.random.default_rng(ss)
                mem = rng.choice(train_rows, size=N_SAMPLE, replace=False)
                non = rng.choice(pool, size=N_SAMPLE, replace=False)
                r = {"dataset": dataset, "scenario": scenario, "epsilon": eps, "sample_seed": ss,
                     "n_train_model": len(train_rows), "nonmember_pool": len(pool),
                     "nonmembers_excluded_exact_copy": excluded, "fit_seconds": fit_secs, **(extra_cols or {}),
                     **evaluate(proba_fn, query_fn(mem), query_fn(non), y[mem], y[non])}
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

        if args.s3c_converged:
            Xn_c = {k: domain_normalize(X[cols[s]], bounds) for k, s in [("S3c", "S0"), ("S3c-min", "S2")]}
            for name in ["S3c", "S3c-min"]:
                base = "S0" if name == "S3c" else "S2"
                for eps in EPS_S3C:
                    if (dataset, name, str(float(eps))) in done:
                        continue
                    m, log = fit_dp_lr_converged(Xn_c[name].iloc[tr_s3].to_numpy(), y[tr_s3], eps,
                                                 np.sqrt(len(cols[base])), NOISE_SEED)
                    pos = list(m.classes_).index(1)
                    record(name, eps, tr_s3, X[cols[base]], lambda Q, m=m, pos=pos: m.predict_proba(Q)[:, pos],
                           lambda idx, nm=name: Xn_c[nm].iloc[idx].to_numpy(), log["fit_seconds_total"],
                           {k: log[k] for k in ["max_iter_used", "n_iter", "converged", "convergence_status"]})
            del X, Xn_c
            continue

        # S0, S1a, S1b, S2: RF validado sobre el train completo y las variables originales
        for name, s in [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]:
            if (dataset, name, "nan") in done:
                continue
            Xs = build_scenario(s, X, cfg, dataset)
            m = create_random_forest(cfg, dataset)
            t0 = time.perf_counter()
            m.fit(Xs.iloc[tr_full], y[tr_full])
            secs = time.perf_counter() - t0
            record(name, np.nan, tr_full, Xs, lambda Q, m=m: m.predict_proba(Q)[:, 1],
                   lambda idx, Xs=Xs: Xs.iloc[idx], secs)
            del m, Xs

        # Variantes S3: datos normalizados por dominio sobre las filas de entrenamiento de S3
        delta = 1.0 / len(tr_s3) ** 2
        Xn = {k: domain_normalize(X[cols[s]], bounds) for k, s in [("S0", "S0"), ("S2", "S2")]}
        for name, base in [("S3a", "S0"), ("S3a-min", "S2")]:
            c = cols[base]
            sigma_of = {eps: analytic_gaussian_sigma(eps, delta, np.sqrt(len(c))) for eps in EPS_S3A}
            for eps in EPS_S3A:
                if (dataset, name, str(float(eps))) in done or (dataset, name, str(eps)) in done:
                    continue
                Xtr = add_gaussian_noise(Xn[base].iloc[tr_s3], sigma_of[eps], NOISE_SEED)
                m = create_random_forest(cfg, dataset)
                t0 = time.perf_counter()
                m.fit(Xtr, y[tr_s3])
                secs = time.perf_counter() - t0
                record(name, eps, tr_s3, X[c], lambda Q, m=m: m.predict_proba(Q)[:, 1],
                       lambda idx, b=base: Xn[b].iloc[idx], secs)
                del m, Xtr
        Xn0 = Xn["S0"]
        for eps in EPS_S3B:
            if (dataset, "S3b", str(float(eps))) in done or (dataset, "S3b", str(eps)) in done:
                continue
            m = DPRandomForest(epsilon=eps, bounds=(0, 1), classes=[0, 1], random_state=NOISE_SEED)
            t0 = time.perf_counter()
            m.fit(Xn0.iloc[tr_s3].to_numpy(), y[tr_s3])
            secs = time.perf_counter() - t0
            pos = list(m.classes_).index(1)
            record("S3b", eps, tr_s3, X[cols["S0"]], lambda Q, m=m, pos=pos: m.predict_proba(Q)[:, pos],
                   lambda idx: Xn0.iloc[idx].to_numpy(), secs)
        d = len(cols["S0"])
        for eps in EPS_S3C:
            if (dataset, "S3c", str(float(eps))) in done or (dataset, "S3c", str(eps)) in done:
                continue
            m = DPLogisticRegression(epsilon=eps, data_norm=float(np.sqrt(d)), random_state=NOISE_SEED, **LR_PARAMS)
            t0 = time.perf_counter()
            m.fit(Xn0.iloc[tr_s3].to_numpy(), y[tr_s3])
            secs = time.perf_counter() - t0
            pos = list(m.classes_).index(1)
            record("S3c", eps, tr_s3, X[cols["S0"]], lambda Q, m=m, pos=pos: m.predict_proba(Q)[:, pos],
                   lambda idx: Xn0.iloc[idx].to_numpy(), secs)
        del X, Xn, Xn0


if __name__ == "__main__":
    main()
