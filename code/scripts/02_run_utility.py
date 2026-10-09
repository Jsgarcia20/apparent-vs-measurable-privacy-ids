"""Fase 9: entrena el RF final por dataset y escenario, guarda probabilidades, predicciones y métricas.

Uso: py -3.10 scripts/02_run_utility.py --run-id <id> --datasets CICIDS2017 HIKARI2021 --scenarios S0 S1 S2
"""
import gc

import numpy as np
import pandas as pd

from _common import path_of, prefix, setup
from src.data import load_xy
from src.metrics import calculate_binary_metrics
from src.models import class_weight_for, train_predict
from src.scenarios import build_scenario
from src.splits import load_split


def extra(parser):
    parser.add_argument("--datasets", nargs="+", default=["CICIDS2017", "HIKARI2021"])
    parser.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S2"])


def main():
    args, cfg, out = setup("Evaluación de utilidad RF", extra)
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    ref_dir = path_of(cfg, "reference_predictions_dir")
    results_path = out / "utility_results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    feature_rows = []

    for dataset in args.datasets:
        X, y = load_xy(cfg, dataset)
        train_idx, test_idx = load_split(cfg, dataset)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        threshold = cfg["datasets"][dataset]["decision_threshold"]
        assert threshold is not None, f"{dataset}: umbral no validado en config.yaml"

        for scenario in args.scenarios:
            Xs = build_scenario(scenario, X, cfg, dataset)
            feature_rows += [{"dataset": dataset, "scenario": scenario, "position": i, "feature": c}
                             for i, c in enumerate(Xs.columns)]
            print(f"--- {dataset} {scenario}: {Xs.shape[1]} features, umbral {threshold}", flush=True)
            proba, pred, t_train, t_pred = train_predict(
                cfg, Xs.iloc[train_idx], y_train, Xs.iloc[test_idx], threshold, dataset
            )
            metrics = calculate_binary_metrics(y_test, pred)
            name = f"{prefix(cfg, dataset)}_{scenario}"
            np.save(pred_dir / f"{name}_predictions.npy", pred)
            np.save(pred_dir / f"{name}_probabilities.npy", proba)

            ref_path = ref_dir / f"{name}_predictions.npy"
            if ref_path.exists():
                ref = np.load(ref_path)
                same = int(np.sum(ref.astype(np.int8) == pred))
                identical = bool(len(ref) == len(pred) and same == len(pred))
            else:
                same, identical = -1, False

            row = {"dataset": dataset, "scenario": scenario, "features": int(Xs.shape[1]),
                   "decision_threshold": threshold,
                   "class_weight": str(class_weight_for(cfg, dataset)), "train_rows": len(train_idx), "test_rows": len(test_idx),
                   **metrics, "training_seconds": t_train, "prediction_seconds": t_pred,
                   "predictions_identical_to_reference": identical, "predictions_equal_count": same}
            rows = [r for r in rows if not (r["dataset"] == dataset and r["scenario"] == scenario)] + [row]
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"F1={metrics['f1']:.6f} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} "
                  f"TP={metrics['tp']} | idéntico a referencia: {identical} | {t_train:.0f}s", flush=True)
            del Xs, proba, pred
            gc.collect()
        del X, y
        gc.collect()

    pd.DataFrame(feature_rows).to_csv(out / f"scenario_features_{'_'.join(args.datasets)}.csv", index=False)


if __name__ == "__main__":
    main()
