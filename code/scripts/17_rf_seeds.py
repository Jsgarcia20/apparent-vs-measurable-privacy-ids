"""Variabilidad del detector: S0, S1a, S1b y S2 con varias semillas del RF (decisión 2026-10-07).

Misma configuración validada de cada dataset (class_weight, umbral) y train completo; solo cambia
random_state del RF. La semilla 42 ya está en la corrida F1 (2026-10-06_F1) y no se repite aquí;
el informe la incorpora. Se guardan métricas, predicciones y probabilidades por semilla.
"""
import copy
import gc

import numpy as np
import pandas as pd

from _common import add_datasets_arg, prefix, setup
from src.data import load_xy
from src.metrics import calculate_binary_metrics, discrimination_metrics
from src.models import train_predict
from src.scenarios import build_scenario
from src.splits import load_split

SCENARIOS = ["S0", "S1", "S1b", "S2"]     # S1 = "S1a" en las tablas


def extra(parser):
    add_datasets_arg(parser, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])


def main():
    args, cfg, out = setup("Semillas del RF", extra)
    out = out / "rf_seeds"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    path = out / "rf_seed_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["dataset"], r["scenario"], r["rf_seed"]) for r in rows}
    for dataset in args.datasets:
        X, y = load_xy(cfg, dataset)
        tr, te = load_split(cfg, dataset)
        thr = cfg["datasets"][dataset]["decision_threshold"]
        for scenario in SCENARIOS:
            Xs = build_scenario(scenario, X, cfg, dataset)
            for seed in args.seeds:
                if (dataset, scenario, seed) in done:
                    continue
                c = copy.deepcopy(cfg)
                c["seeds"]["rf"] = seed
                proba, pred, t_fit, _ = train_predict(c, Xs.iloc[tr], y.iloc[tr], Xs.iloc[te], thr, dataset)
                name = f"{prefix(cfg, dataset)}_{scenario}_rfseed{seed}"
                np.save(out / "predictions" / f"{name}_predictions.npy", pred)
                np.save(out / "predictions" / f"{name}_probabilities.npy", proba)
                m = calculate_binary_metrics(y.iloc[te], pred)
                r = {"dataset": dataset, "scenario": scenario, "rf_seed": seed, "decision_threshold": thr,
                     **{k: m[k] for k in ["f1", "precision", "recall", "fpr", "tn", "fp", "fn", "tp"]},
                     **{k: v for k, v in discrimination_metrics(y.iloc[te].to_numpy(), proba).items()
                        if k in ("roc_auc", "pr_auc")}, "fit_seconds": t_fit}
                rows.append(r)
                pd.DataFrame(rows).to_csv(path, index=False)
                print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                del proba, pred
                gc.collect()
            del Xs
            gc.collect()
        del X, y


if __name__ == "__main__":
    main()
