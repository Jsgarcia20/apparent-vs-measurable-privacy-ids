"""Fase 10 — Estadística (protocolo registrado antes de ejecutar).

A) Escenarios estructurales (S1a, S1b y S2 frente a S0), por dataset, con las predicciones de la
   semilla 42 del RF (corrida 2026-10-06_F1):
   - bootstrap pareado (1000) de ΔF1, Δprecisión, Δrecall y ΔFPR (multinomial, src/stats.py) y de
     ΔPR-AUC y ΔROC-AUC (remuestreo de filas pareado), con IC del 95 %;
   - semilla = 42 + offset (CICIDS 0, v2_main 100, v2_pairs 200) + índice (S1a 0, S1b 1, S2 2);
   - McNemar exacto y Holm por dataset (familia = 3);
   - TOST: equivalencia si el IC del 95 % de ΔF1 ⊂ [−margen, +margen], con margen = 0,005 (o
     3 × sd de F1 entre semillas en S0, si fuera mayor que 0,005/3);
   - degradación relativa y rango de F1 entre las 5 semillas del RF.
B) Escenarios con DP: media, sd e IC del 95 % (t) entre semillas y diferencia frente a su referencia
   (S0 —o S0-sub en CICIDS— para los RF; LR-ref convergida para S3c/S3c-min). Sin McNemar.
Las AUC se calculan en float32, como en el resto del proyecto.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as st
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import path_of, setup
from src import stats as S
from src.data import load_raw, make_target
from src.splits import load_split

DATASETS = {"CICIDS2017": ("CICIDS", 0), "HIKARI2021_v2_main": ("HIKARI_v2_main", 100),
            "HIKARI2021_v2_pairs": ("HIKARI_v2_pairs", 200)}
COMPARISONS = [("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]
N_BOOT = 1000
METRICS = ["f1", "precision", "recall", "fpr", "pr_auc", "roc_auc"]


def extra(parser):
    parser.add_argument("--f1-run", default="2026-10-06_F1")
    parser.add_argument("--s3b-run", default="2026-10-07_S3B")
    parser.add_argument("--batch2-run", default="2026-10-07_batch2")


def auc_pair(y, p0, p1, idx):
    yy = y[idx]
    return (roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx]),
            average_precision_score(yy, p1[idx]) - average_precision_score(yy, p0[idx]))


def auc_bootstrap(y, p0, p1, seed):
    rng = np.random.default_rng(seed)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    res = np.array(Parallel(n_jobs=-1, batch_size=8)(delayed(auc_pair)(y, p0, p1, i) for i in idxs))
    return res[:, 0], res[:, 1]


def ci_t(v):
    v = np.asarray(v, float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = float(v.std(ddof=1))
    h = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, sd, m - h, m + h


def structural(cfg, args, root, out):
    preds = root / args.f1_run / "predictions"
    seeds = pd.read_csv(root / args.batch2_run / "rf_seeds" / "rf_seed_runs.csv")
    f1run = pd.read_csv(root / args.f1_run / "utility_report" / "utility_full_test.csv")
    f1run = f1run.assign(rf_seed=42)[["dataset", "scenario", "rf_seed", "f1"]]
    allseeds = pd.concat([seeds[["dataset", "scenario", "rf_seed", "f1"]], f1run])
    rows = []
    for dataset, (pfx, offset) in DATASETS.items():
        y = make_target(load_raw(cfg, dataset), dataset, cfg).to_numpy()
        _, te = load_split(cfg, dataset)
        yt = y[te].astype(np.int8)
        base_pred = np.load(preds / f"{pfx}_S0_predictions.npy").astype(np.int8)
        base_p = np.load(preds / f"{pfx}_S0_probabilities.npy").astype(np.float32)
        s0_sd = allseeds[(allseeds.dataset == dataset) & (allseeds.scenario == "S0")].f1.std(ddof=1)
        margin = max(0.005, 3 * s0_sd) if s0_sd > 0.005 / 3 else 0.005
        tmp = []
        for idx, (label, internal) in enumerate(COMPARISONS):
            seed = 42 + offset + idx
            pred = np.load(preds / f"{pfx}_{internal}_predictions.npy").astype(np.int8)
            p = np.load(preds / f"{pfx}_{internal}_probabilities.npy").astype(np.float32)
            thr = {r["metric"]: r for r in S.paired_bootstrap_all_binary_metrics_fast(
                yt, base_pred, pred, n_bootstrap=N_BOOT, random_seed=seed)}
            d_roc, d_pr = auc_bootstrap(yt, base_p, p, seed)
            mc = S.exact_mcnemar_phase10(yt, base_pred, pred)
            f1s = allseeds[(allseeds.dataset == dataset) & (allseeds.scenario == internal)].f1
            r = {"dataset": dataset, "comparison": f"{label} vs S0", "bootstrap_seed": seed, "equivalence_margin": margin,
                 "mcnemar_b": mc["baseline_correct_scenario_wrong"], "mcnemar_c": mc["baseline_wrong_scenario_correct"],
                 "mcnemar_p": mc["p_value"], "f1_rf_seeds_min": float(f1s.min()), "f1_rf_seeds_max": float(f1s.max()),
                 "f1_rf_seeds_range": float(f1s.max() - f1s.min()), "n_rf_seeds": int(len(f1s))}
            for mtr in ["f1", "precision", "recall", "fpr"]:
                t = thr[mtr]
                r.update({f"{mtr}_s0": t["baseline_value"], f"{mtr}_scenario": t["scenario_value"],
                          f"delta_{mtr}": t["difference"], f"delta_{mtr}_ci_low": t["ci_lower"],
                          f"delta_{mtr}_ci_high": t["ci_upper"], f"rel_degradation_{mtr}_pct": t["relative_degradation_percent"]})
            for mtr, dist in [("roc_auc", d_roc), ("pr_auc", d_pr)]:
                b = roc_auc_score(yt, base_p) if mtr == "roc_auc" else average_precision_score(yt, base_p)
                s = roc_auc_score(yt, p) if mtr == "roc_auc" else average_precision_score(yt, p)
                r.update({f"{mtr}_s0": b, f"{mtr}_scenario": s, f"delta_{mtr}": s - b,
                          f"delta_{mtr}_ci_low": float(np.quantile(dist, 0.025)),
                          f"delta_{mtr}_ci_high": float(np.quantile(dist, 0.975)),
                          f"rel_degradation_{mtr}_pct": (b - s) / b * 100})
            r["tost_equivalent_f1"] = bool(r["delta_f1_ci_low"] >= -margin and r["delta_f1_ci_high"] <= margin)
            tmp.append(r)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        holm = S.holm_adjustment_phase10([r["mcnemar_p"] for r in tmp])
        for r, h in zip(tmp, holm):
            r["mcnemar_p_holm"] = float(h)
            r["significant_after_holm"] = bool(h < 0.05)
        rows += tmp
    res = pd.DataFrame(rows)
    res.to_csv(out / "phase10_structural.csv", index=False)
    return res


def dp_summary(cfg, args, root, out):
    s3 = root / args.s3b_run / "S3B"
    b2 = root / args.batch2_run / "S3c"
    f1run = pd.read_csv(root / args.f1_run / "utility_report" / "utility_full_test.csv")
    lrref = pd.read_csv(b2 / "lr_ref_converged.csv")
    lrref = lrref[lrref.converged_without_warning]
    rows = []
    for dataset in DATASETS:
        u = pd.read_csv(s3 / f"{dataset}_utility_runs.csv")
        c = pd.read_csv(s3 / f"{dataset}_central_runs.csv")
        lr = pd.read_csv(b2 / "s3c_converged_runs.csv")
        lr = lr[lr.dataset == dataset]
        if dataset == "CICIDS2017":
            ref_rf = u[u.variant == "S0-sub"].iloc[0]
            ref_name = "S0-sub"
        else:
            ref_rf = f1run[(f1run.dataset == dataset) & (f1run.scenario == "S0")].iloc[0]
            ref_name = "S0"
        groups = [("S3a", u[(u.variant == "S3a") & (u.evaluation == "clean_test")], ref_rf, ref_name),
                  ("S3a-min", u[(u.variant == "S3a-min") & (u.evaluation == "clean_test")], ref_rf, ref_name),
                  ("S3b", c, ref_rf, ref_name)]
        for v in ["S3c", "S3c-min"]:
            ref = lrref[(lrref.dataset == dataset) & (lrref.variant == f"LR-ref ({v})")].iloc[-1]
            groups.append((v, lr[lr.variant == v], ref, f"LR-ref ({v}), convergida"))
        for v, g, ref, ref_name_v in groups:
            for eps, ge in g.groupby("epsilon"):
                r = {"dataset": dataset, "variant": v, "epsilon": eps, "n_seeds": len(ge), "reference": ref_name_v}
                for mtr in METRICS:
                    m, sd, lo, hi = ci_t(ge[mtr])
                    rv = float(ref[mtr])
                    r.update({f"{mtr}_mean": m, f"{mtr}_sd": sd, f"{mtr}_ci95_low": lo, f"{mtr}_ci95_high": hi,
                              f"{mtr}_reference": rv, f"delta_{mtr}_vs_reference": m - rv,
                              f"delta_{mtr}_ci95_low": lo - rv, f"delta_{mtr}_ci95_high": hi - rv})
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "phase10_dp_vs_reference.csv", index=False)
    return res


def main():
    args, cfg, out = setup("Fase 10: estadística", extra)
    root = path_of(cfg, "results_dir")
    out = out / "phase10"
    out.mkdir(exist_ok=True)
    s = structural(cfg, args, root, out)
    d = dp_summary(cfg, args, root, out)
    pd.set_option("display.width", 250)
    cols = ["dataset", "comparison", "delta_f1", "delta_f1_ci_low", "delta_f1_ci_high", "tost_equivalent_f1",
            "delta_pr_auc", "delta_pr_auc_ci_low", "delta_pr_auc_ci_high", "mcnemar_p_holm", "significant_after_holm",
            "rel_degradation_f1_pct", "f1_rf_seeds_range"]
    print(s[cols].round(6).to_string(index=False))
    print(d[["dataset", "variant", "epsilon", "n_seeds", "pr_auc_mean", "pr_auc_ci95_low", "pr_auc_ci95_high",
             "delta_pr_auc_vs_reference", "f1_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
