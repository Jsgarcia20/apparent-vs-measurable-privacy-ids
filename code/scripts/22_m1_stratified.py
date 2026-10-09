"""M1 estratificado por clase (corrección del sesgo de composición).

Igual que 19_membership_inference.py (mismos escenarios, ε, modelos, semillas y regla de exclusión), pero:
- miembros y no miembros se muestrean ESTRATIFICADOS por clase con la misma proporción de ataques, la
  del train del escenario (n_ataque = round(10.000 × prevalencia) en cada grupo);
- se reporta M1 global y por clase (solo benignos y solo ataques);
- S3c y S3c-min usan LR con DP convergidas (decisión 5b).
"""
import time

import numpy as np
import pandas as pd
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


def tpr_at_fpr(lab, score, target=0.01):
    fpr, tpr, _ = roc_curve(lab, score)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def m1_metrics(s_m, s_n):
    score = np.concatenate([s_m, s_n])
    lab = np.concatenate([np.ones(len(s_m)), np.zeros(len(s_n))])
    return float(roc_auc_score(lab, score)), tpr_at_fpr(lab, score)


def stratified_sample(rng, rows, y, n_att):
    att, ben = rows[y[rows] == 1], rows[y[rows] == 0]
    return np.concatenate([rng.choice(att, n_att, replace=False), rng.choice(ben, N_SAMPLE - n_att, replace=False)])


def main():
    args, cfg, out = setup("M1 estratificado por clase", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]))
    out = out / "M1"
    out.mkdir(exist_ok=True)
    path = out / "m1_stratified.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["scenario"], str(r["epsilon"])) for r in rows}

    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        del df
        tr_full, te = load_split(cfg, dataset)
        tr_s3 = training_rows(cfg, dataset, y)[0] if dataset in SUBSAMPLE else tr_full
        bounds = load_bounds(cfg, dataset)
        cols = {s: list(build_scenario(s, X.iloc[:1], cfg, dataset).columns) for s in ["S0", "S1", "S1b", "S2"]}

        def record(scenario, eps, train_rows, copy_cols, proba_fn, query_fn, fit_secs, extra=None):
            k = row_hash(copy_cols)
            keys = set(k[train_rows].tolist())
            has = np.fromiter((v in keys for v in k[te]), bool, len(te))
            pool = te[~has]
            prev = float(y[train_rows].mean())
            n_att = int(round(N_SAMPLE * prev))
            for ss in SAMPLE_SEEDS:
                rng = np.random.default_rng(ss)
                mem = stratified_sample(rng, train_rows, y, n_att)
                non = stratified_sample(rng, pool, y, n_att)
                pm, pn = proba_fn(query_fn(mem)), proba_fn(query_fn(non))
                s_m = np.where(y[mem] == 1, pm, 1 - pm)
                s_n = np.where(y[non] == 1, pn, 1 - pn)
                auc, tpr = m1_metrics(s_m, s_n)
                auc_b, tpr_b = m1_metrics(s_m[y[mem] == 0], s_n[y[non] == 0])
                auc_a, tpr_a = m1_metrics(s_m[y[mem] == 1], s_n[y[non] == 1])
                r = {"dataset": dataset, "scenario": scenario, "epsilon": eps, "sample_seed": ss,
                     "train_attack_rate": prev, "n_attack_per_group": n_att,
                     "members_attack_rate": float(y[mem].mean()), "nonmembers_attack_rate": float(y[non].mean()),
                     "nonmembers_excluded_exact_copy": int(has.sum()), "nonmember_pool": len(pool),
                     "roc_auc": auc, "tpr_at_fpr_1pct": tpr, "roc_auc_benign": auc_b, "tpr_at_fpr_1pct_benign": tpr_b,
                     "roc_auc_attack": auc_a, "tpr_at_fpr_1pct_attack": tpr_a, "fit_seconds": fit_secs, **(extra or {})}
                rows.append(r)
                print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)

        for name, s in [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]:
            if (dataset, name, "nan") in done:
                continue
            Xs = build_scenario(s, X, cfg, dataset)
            m = create_random_forest(cfg, dataset)
            t0 = time.perf_counter()
            m.fit(Xs.iloc[tr_full], y[tr_full])
            record(name, np.nan, tr_full, Xs, lambda Q, m=m: m.predict_proba(Q)[:, 1],
                   lambda idx, Xs=Xs: Xs.iloc[idx], time.perf_counter() - t0)
            del m, Xs

        delta = 1.0 / len(tr_s3) ** 2
        Xn = {b: domain_normalize(X[cols[b]], bounds) for b in ["S0", "S2"]}
        for name, base in [("S3a", "S0"), ("S3a-min", "S2")]:
            for eps in EPS_S3A:
                if (dataset, name, str(float(eps))) in done:
                    continue
                sigma = analytic_gaussian_sigma(eps, delta, np.sqrt(len(cols[base])))
                Xtr = add_gaussian_noise(Xn[base].iloc[tr_s3], sigma, NOISE_SEED)
                m = create_random_forest(cfg, dataset)
                t0 = time.perf_counter()
                m.fit(Xtr, y[tr_s3])
                record(name, eps, tr_s3, X[cols[base]], lambda Q, m=m: m.predict_proba(Q)[:, 1],
                       lambda idx, b=base: Xn[b].iloc[idx], time.perf_counter() - t0)
                del m, Xtr
        for eps in EPS_S3B:
            if (dataset, "S3b", str(float(eps))) in done:
                continue
            m = DPRandomForest(epsilon=eps, bounds=(0, 1), classes=[0, 1], random_state=NOISE_SEED)
            t0 = time.perf_counter()
            m.fit(Xn["S0"].iloc[tr_s3].to_numpy(), y[tr_s3])
            pos = list(m.classes_).index(1)
            record("S3b", eps, tr_s3, X[cols["S0"]], lambda Q, m=m, pos=pos: m.predict_proba(Q)[:, pos],
                   lambda idx: Xn["S0"].iloc[idx].to_numpy(), time.perf_counter() - t0)
        for name, base in [("S3c", "S0"), ("S3c-min", "S2")]:
            for eps in EPS_S3C:
                if (dataset, name, str(float(eps))) in done:
                    continue
                m, log = fit_dp_lr_converged(Xn[base].iloc[tr_s3].to_numpy(), y[tr_s3], eps,
                                             np.sqrt(len(cols[base])), NOISE_SEED)
                pos = list(m.classes_).index(1)
                record(name, eps, tr_s3, X[cols[base]], lambda Q, m=m, pos=pos: m.predict_proba(Q)[:, pos],
                       lambda idx, b=base: Xn[b].iloc[idx].to_numpy(), log["fit_seconds_total"],
                       {k: log[k] for k in ["max_iter_used", "n_iter", "converged", "convergence_status"]})
        del X, Xn


if __name__ == "__main__":
    main()
