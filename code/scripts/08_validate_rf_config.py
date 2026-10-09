"""D3 — Revalidación de class_weight y umbral en validación interna (réplica de las fases 9A–9C).

El test NO interviene: todo se hace dentro del train del split fijo del dataset.
- Validación interna: train_test_split sobre las posiciones del train, 20 %, estratificado, semilla 42.
- class_weight ∈ {None, balanced, balanced_subsample}: se entrena en el subtrain y se predice con
  model.predict (umbral 0.50, como en la celda 27). Criterio: máximo F1; desempates: mayor recall
  y menor FPR.
- Umbral: rejilla 0.01–0.50 (paso 0.01) sobre las probabilidades de validación del class_weight
  elegido (predicción = proba >= umbral, como en las celdas 28–29); mismo criterio y desempates.
- Si el umbral óptimo queda en un extremo de la rejilla, el script lo marca y termina con error.
- Con --apply-tie-rule: si |ΔF1| < validation.tie_tolerance entre la mejor opción y la configuración
  congelada del dataset en config.yaml (class_weight y umbral), se conserva la congelada.

El script no modifica config.yaml: la configuración elegida se copia a mano tras revisarla.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _common import add_datasets_arg, setup
from src.data import load_xy
from src.metrics import calculate_binary_metrics
from src.models import class_weight_for, create_random_forest
from src.splits import load_split

RANKING = dict(by=["f1", "recall", "fpr"], ascending=[False, False, True])


def label(cw):
    return "None" if cw is None else cw


def extra(parser):
    add_datasets_arg(parser)
    parser.add_argument("--apply-tie-rule", action="store_true",
                        help="Conservar la configuración congelada si |ΔF1| < validation.tie_tolerance")


def tie_rule(table, column, best_value, frozen_value, tolerance):
    """Devuelve (valor final, ΔF1 mejor - congelada, si se conservó la congelada)."""
    best_f1 = float(table.loc[table[column] == best_value, "f1"].iloc[0])
    frozen_rows = table.loc[table[column] == frozen_value, "f1"]
    if frozen_rows.empty:
        return best_value, None, False
    delta = best_f1 - float(frozen_rows.iloc[0])
    keep_frozen = abs(delta) < tolerance
    return (frozen_value if keep_frozen else best_value), delta, keep_frozen


def main():
    args, cfg, out = setup("Revalidación de class_weight y umbral", extra)
    vcfg = cfg["validation"]
    g = vcfg["threshold_grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + g["step"] / 2, g["step"]), 2)
    summary, at_edge = {}, []

    for dataset in args.datasets:
        d = out / "D3_validation" / dataset
        d.mkdir(parents=True, exist_ok=True)
        X, y = load_xy(cfg, dataset)
        train_idx, _ = load_split(cfg, dataset)          # el test no se carga en ningún momento
        X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        del X
        pos = np.arange(len(X_tr))
        sub_pos, val_pos = train_test_split(pos, test_size=vcfg["internal_validation_size"],
                                            random_state=vcfg["internal_validation_seed"], stratify=y_tr)
        X_sub, y_sub = X_tr.iloc[sub_pos], y_tr.iloc[sub_pos]
        X_val, y_val = X_tr.iloc[val_pos], y_tr.iloc[val_pos]

        rows, probas = [], {}
        for cw in vcfg["class_weight_candidates"]:
            model = create_random_forest(cfg, class_weight=cw)
            t0 = time.perf_counter()
            model.fit(X_sub, y_sub)
            secs = time.perf_counter() - t0
            pred = model.predict(X_val).astype(np.int8)
            probas[label(cw)] = model.predict_proba(X_val)[:, 1]
            rows.append({"class_weight": label(cw), **calculate_binary_metrics(y_val, pred),
                         "predicted_attacks": int(pred.sum()), "training_seconds": secs})
            print(dataset, rows[-1], flush=True)
            del model
        cw_table = pd.DataFrame(rows)
        cw_rank = cw_table.sort_values(**RANKING).reset_index(drop=True)
        best_cw_raw = cw_rank.iloc[0]["class_weight"]
        best_cw, cw_delta, cw_kept = best_cw_raw, None, False
        if args.apply_tie_rule:
            best_cw, cw_delta, cw_kept = tie_rule(cw_table, "class_weight", best_cw_raw,
                                                  label(class_weight_for(cfg, dataset)), vcfg["tie_tolerance"])

        p = probas[best_cw]
        th_rows = [{"threshold": float(t), **calculate_binary_metrics(y_val, (p >= t).astype(np.int8))} for t in grid]
        th_table = pd.DataFrame(th_rows)
        th_rank = th_table.sort_values(**RANKING).reset_index(drop=True)
        best_t_raw = float(th_rank.iloc[0]["threshold"])
        edge = best_t_raw in (float(grid.min()), float(grid.max()))
        best_t, t_delta, t_kept = best_t_raw, None, False
        if args.apply_tie_rule:
            frozen_t = cfg["datasets"][dataset]["decision_threshold"]
            best_t, t_delta, t_kept = tie_rule(th_table.assign(threshold=th_table.threshold.round(2)), "threshold",
                                               round(best_t_raw, 2), round(float(frozen_t), 2), vcfg["tie_tolerance"])
            best_t = float(best_t)
        if edge:
            at_edge.append(dataset)

        cw_table.to_csv(d / "class_weight_validation.csv", index=False)
        th_table.to_csv(d / "threshold_validation.csv", index=False)
        np.save(d / f"validation_probabilities_{best_cw}.npy", p)
        summary[dataset] = {
            "train_rows": int(len(X_tr)), "internal_subtrain_rows": int(len(sub_pos)),
            "internal_validation_rows": int(len(val_pos)),
            "internal_validation_attack_rate": float(y_val.mean()),
            "best_class_weight_raw": best_cw_raw, "selected_class_weight": best_cw,
            "tie_rule_applied": bool(args.apply_tie_rule),
            "class_weight_delta_f1_best_minus_frozen": cw_delta, "class_weight_kept_frozen": cw_kept,
            "class_weight_validation_f1": {r["class_weight"]: r["f1"] for r in rows},
            "best_threshold_raw": best_t_raw, "selected_threshold": best_t, "threshold_at_grid_edge": edge,
            "threshold_delta_f1_best_minus_frozen": t_delta, "threshold_kept_frozen": t_kept,
            "validation_at_best_threshold_raw": th_rank.iloc[0].to_dict(),
            "validation_at_selected_threshold": th_table[np.isclose(th_table.threshold, best_t)].iloc[0].to_dict(),
            "validation_at_0.50": th_table[np.isclose(th_table.threshold, 0.5)].iloc[0].to_dict(),
            "top5_thresholds": th_rank.head(5)[["threshold", "f1", "recall", "precision", "fpr"]].to_dict("records"),
            "final_test_used_for_selection": False,
        }
        print(json.dumps(summary[dataset], indent=2, default=float), flush=True)

    (out / "D3_validation" / f"D3_summary_{'_'.join(args.datasets)}.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    if at_edge:
        raise SystemExit(f"UMBRAL ÓPTIMO EN EL EXTREMO DE LA REJILLA: {at_edge}. Detenerse e informar.")


if __name__ == "__main__":
    main()
