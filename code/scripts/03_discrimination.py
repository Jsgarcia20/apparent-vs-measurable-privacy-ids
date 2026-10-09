"""ROC-AUC y PR-AUC a partir de las probabilidades guardadas por 02_run_utility.py (mismo entrenamiento)."""
import numpy as np
import pandas as pd

from _common import add_datasets_arg, prefix, setup
from src.data import load_raw, make_target
from src.metrics import discrimination_metrics
from src.splits import load_split


def main():
    args, cfg, out = setup("Discriminación independiente del umbral", add_datasets_arg)
    rows = []
    for dataset in args.datasets:
        y = make_target(load_raw(cfg, dataset), dataset, cfg)
        _, test_idx = load_split(cfg, dataset)
        y_test = y.iloc[test_idx].to_numpy()
        pfx = prefix(cfg, dataset)
        # "S*" evita que el prefijo HIKARI capture los archivos HIKARI_v2_*
        for path in sorted((out / "predictions").glob(f"{pfx}_S*_probabilities.npy")):
            scenario = path.name[len(pfx) + 1:-len("_probabilities.npy")]
            rows.append({"dataset": dataset, "scenario": scenario, **discrimination_metrics(y_test, np.load(path))})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"discrimination_results_{'_'.join(args.datasets)}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
