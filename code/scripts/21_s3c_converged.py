"""Decisión 5b (2026-10-07): S3c y S3c-min con LR con DP convergidas (max_iter 2000 → 5000 → 10000).

Mismos datos, ε, semillas e hiperparámetros que 16_s3c_dp_logreg.py, salvo max_iter. Se registra por
ajuste el max_iter usado, las iteraciones reales y si convergió. La pérdida relativa se calcula
respecto de la LR-ref convergida (20_lr_ref_converged.py, referencia principal según la decisión 5).
"""
import numpy as np
import pandas as pd

from _common import add_datasets_arg, setup
from src.dp_local import domain_normalize, load_bounds
from src.dp_lr import fit_dp_lr_converged
from src.s3_common import prepare, utility_metrics

EPS = [0.1, 0.5, 1, 5, 10, 50, 100]
SEEDS = {"CICIDS2017": [42, 43, 44]}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
VARIANTS = {"S3c": "S3a", "S3c-min": "S3a-min"}


def main():
    args, cfg, out = setup("S3c con LR convergidas (decisión 5b)", lambda p: add_datasets_arg(
        p, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]))
    out = out / "S3c"
    out.mkdir(exist_ok=True)
    ref_path = out / "lr_ref_converged.csv"
    refs = pd.read_csv(ref_path) if ref_path.exists() else None
    path = out / "s3c_converged_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["variant"], float(r["epsilon"]), int(r["seed"])) for r in rows}
    for dataset in args.datasets:
        df, X, y, tr, te, cols = prepare(cfg, dataset)
        del df
        thr = cfg["datasets"][dataset]["decision_threshold"]
        for v, key in VARIANTS.items():
            c = cols[key]
            Xn = domain_normalize(X.iloc[np.concatenate([tr, te])][c], load_bounds(cfg, dataset))
            Xtr, Xte = Xn.iloc[:len(tr)].to_numpy(), Xn.iloc[len(tr):].to_numpy()
            ref = None
            if refs is not None:
                rr = refs[(refs.dataset == dataset) & (refs.variant == f"LR-ref ({v})") & refs.converged_without_warning]
                ref = rr.iloc[-1] if len(rr) else None
            for eps in EPS:
                for seed in SEEDS.get(dataset, DEFAULT_SEEDS):
                    if (dataset, v, float(eps), seed) in done:
                        continue
                    m, log = fit_dp_lr_converged(Xtr, y[tr], eps, np.sqrt(len(c)), seed)
                    met = utility_metrics(y[te], m.predict_proba(Xte)[:, list(m.classes_).index(1)], thr)
                    rel = {f"rel_loss_{k}": (float(ref[k]) - met[k]) / float(ref[k]) if ref is not None and ref[k] else np.nan
                           for k in ["f1", "pr_auc", "roc_auc"]}
                    r = {"dataset": dataset, "variant": v, "epsilon": eps, "seed": seed, "n_features": len(c),
                         "data_norm": float(np.sqrt(len(c))), **met, **rel, **log}
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(path, index=False)
                    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    nc = res[~res.converged]
    print(f"No convergidos ni con 10000: {len(nc)}", nc.groupby(["dataset", "variant", "epsilon"]).size().to_dict())


if __name__ == "__main__":
    main()
