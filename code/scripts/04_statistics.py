"""Fase 10: bootstrap pareado, McNemar exacto y Holm sobre las predicciones de una corrida.

Lógica de la celda 43 (src/stats.py). Solo compara los escenarios cuyas predicciones
existen en la corrida. Holm se aplica por dataset sobre las comparaciones presentes.
NOTA: con menos escenarios que en el notebook (7 comparaciones), Holm no es comparable
con statistical_analysis/phase10.
"""
import numpy as np
import pandas as pd

from _common import add_datasets_arg, prefix, setup
from src import stats
from src.data import load_raw, make_target
from src.splits import load_split

DATASET_LABEL = {"CICIDS2017": "CICIDS2017", "HIKARI2021": "HIKARI-2021"}


def main():
    args, cfg, out = setup("Análisis estadístico pareado", add_datasets_arg)
    boot_rows, mc_rows = [], []
    for dataset in args.datasets:
        y = make_target(load_raw(cfg, dataset), dataset, cfg)
        _, test_idx = load_split(cfg, dataset)
        y_test = y.iloc[test_idx].to_numpy(dtype=np.int8)
        pdir = out / "predictions"
        base = np.load(pdir / f"{prefix(cfg, dataset)}_S0_predictions.npy").astype(np.int8)
        label = DATASET_LABEL.get(dataset, dataset)
        for idx, scenario in enumerate(stats.COMPARISON_SCENARIOS):
            path = pdir / f"{prefix(cfg, dataset)}_{scenario.replace('.', '_')}_predictions.npy"
            if not path.exists():
                continue
            pred = np.load(path).astype(np.int8)
            for r in stats.paired_bootstrap_all_binary_metrics_fast(
                y_test, base, pred, random_seed=stats.comparison_seed(label, idx)
            ):
                boot_rows.append({"dataset": label, "baseline": "S0", "scenario": scenario, **r})
            mc_rows.append({"dataset": label, "baseline": "S0", "scenario": scenario,
                            **stats.exact_mcnemar_phase10(y_test, base, pred)})
    boot = pd.DataFrame(boot_rows)
    mc = pd.DataFrame(mc_rows)
    mc["p_value_holm"] = np.nan
    for label in mc["dataset"].unique():
        mask = mc["dataset"] == label
        mc.loc[mask, "p_value_holm"] = stats.holm_adjustment_phase10(mc.loc[mask, "p_value"].to_numpy())
    mc["significant_after_holm"] = mc["p_value_holm"] < stats.ALPHA
    boot.to_csv(out / "paired_bootstrap_all_binary_metrics.csv", index=False)
    mc.to_csv(out / "exact_mcnemar_holm_results.csv", index=False)
    print(boot[boot.metric == "f1"].to_string(index=False))
    print(mc.to_string(index=False))


if __name__ == "__main__":
    main()
