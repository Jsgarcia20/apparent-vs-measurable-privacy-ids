"""D4 — Informe de utilidad por versión de dataset a partir de predicciones ya guardadas (no entrena).

Para cada dataset y escenario (S0, S1, S2):
- F1, precisión, recall, FPR, ROC-AUC, PR-AUC en el test completo.
- Por categoría de tráfico del test (traffic_category en HIKARI, Label en CICIDS): recall (categorías de ataque) o FPR (categorías Label 0).
- Las mismas métricas en el subconjunto del test SIN copia exacta en train (copia definida con las
  columnas del escenario), como en C2.

Uso: py -3.10 scripts/09_utility_report.py --run-id <salida> \
        --sources HIKARI2021=2026-10-06_partB HIKARI2021_v2_pairs=2026-10-06_D4 HIKARI2021_v2_main=2026-10-06_D4
"""
import numpy as np
import pandas as pd

from _common import path_of, prefix, setup
from src.data import load_raw, make_predictors, make_target
from src.integrity import row_hash, subset_metrics, test_has_train_copy
from src.metrics import calculate_binary_metrics, discrimination_metrics
from src.scenarios import build_scenario
from src.splits import load_split

SCENARIOS = ["S0", "S1", "S1b", "S2"]   # S1 = "S1a" en las tablas


def extra(parser):
    parser.add_argument("--sources", nargs="+", required=True, help="DATASET=RUN_ID con las predicciones")


def main():
    args, cfg, out = setup("D4: informe de utilidad", extra)
    out = out / "utility_report"
    out.mkdir(exist_ok=True)
    full_rows, cat_rows, sub_rows = [], [], []
    for item in args.sources:
        dataset, run_id = item.split("=")
        pdir = path_of(cfg, "results_dir") / run_id / "predictions"
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        train_idx, test_idx = load_split(cfg, dataset)
        y_test = y[test_idx]
        cat_test = df[cfg["datasets"][dataset]["category_column"]].astype(str).str.strip().to_numpy()[test_idx]
        del df
        for scenario in SCENARIOS:
            pred = np.load(pdir / f"{prefix(cfg, dataset)}_{scenario}_predictions.npy")
            proba = np.load(pdir / f"{prefix(cfg, dataset)}_{scenario}_probabilities.npy")
            assert len(pred) == len(test_idx)
            m = calculate_binary_metrics(y_test, pred)
            full_rows.append({"dataset": dataset, "scenario": scenario, "test_rows": len(test_idx),
                              "test_attacks": int(y_test.sum()), "f1": m["f1"], "precision": m["precision"],
                              "recall": m["recall"], "fpr": m["fpr"], **discrimination_metrics(y_test, proba),
                              "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]})
            for cat in sorted(set(cat_test)):
                mask = cat_test == cat
                is_attack = bool(y_test[mask].mean() == 1.0)
                cat_rows.append({"dataset": dataset, "scenario": scenario, "category": cat,
                                 "label": int(is_attack), "test_rows": int(mask.sum()),
                                 "predicted_attack": int(pred[mask].sum()),
                                 "metric": "recall" if is_attack else "fpr",
                                 "value": float(pred[mask].mean())})
            Xs = build_scenario(scenario, X, cfg, dataset)
            has_copy = test_has_train_copy(row_hash(Xs), train_idx, test_idx)
            for subset, mask in [("full_test", np.ones(len(test_idx), bool)),
                                 ("test_without_train_copy", ~has_copy)]:
                sub_rows.append({"dataset": dataset, "scenario": scenario, "subset": subset,
                                 "pct_of_test": float(mask.mean() * 100), **subset_metrics(y_test, pred, proba, mask)})
            del Xs
    full = pd.DataFrame(full_rows)
    cats = pd.DataFrame(cat_rows)
    subs = pd.DataFrame(sub_rows)
    full.to_csv(out / "utility_full_test.csv", index=False)
    cats.to_csv(out / "utility_by_traffic_category.csv", index=False)
    subs.to_csv(out / "utility_without_train_copy.csv", index=False)
    pd.set_option("display.width", 250)
    print(full.to_string(index=False))
    print(cats.pivot_table(index=["category", "metric"], columns=["dataset", "scenario"], values="value").round(4).to_string())
    print(subs.to_string(index=False))


if __name__ == "__main__":
    main()
