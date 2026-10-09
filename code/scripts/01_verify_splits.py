"""Verifica los splits fijos guardados en disco (no los regenera ni los modifica)."""
import json

from _common import add_datasets_arg, setup
from src.data import load_raw, make_target
from src.splits import load_split, regenerated_matches_saved, verify_split


def main():
    args, cfg, out = setup("Verificación de splits fijos", add_datasets_arg)
    reports = {}
    ok = True
    for dataset in args.datasets:
        y = make_target(load_raw(cfg, dataset), dataset, cfg).to_numpy()
        train_idx, test_idx = load_split(cfg, dataset)
        exp = cfg["datasets"][dataset]["expected_split"]
        if exp is None:  # splits nuevos (v2): el tamaño esperado es el de train_test_split
            exp = {"train": len(y) - len(test_idx), "test": len(test_idx)}
        rep = verify_split(train_idx, test_idx, y, exp["train"], exp["test"])
        rep["train_matches_train_test_split_42"], rep["test_matches_train_test_split_42"] = regenerated_matches_saved(
            train_idx, test_idx, y, cfg["split"]["test_size"], cfg["seeds"]["split"]
        )
        passed = (
            rep["sizes_ok"] and rep["overlap"] == 0 and rep["full_coverage"] and rep["stratification_ok"]
            and rep["train_unique"] and rep["test_unique"]
            and rep["train_matches_train_test_split_42"] and rep["test_matches_train_test_split_42"]
        )
        rep["passed"] = bool(passed)
        ok &= passed
        reports[dataset] = rep
        print(dataset, json.dumps(rep, indent=2))
    (out / f"split_verification_{'_'.join(args.datasets)}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not ok:
        raise SystemExit("VERIFICACIÓN DE SPLITS FALLIDA: detenerse e informar.")
    print("Splits verificados.")


if __name__ == "__main__":
    main()
