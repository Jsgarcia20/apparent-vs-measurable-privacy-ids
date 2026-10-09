"""DIAGNÓSTICO del ROC-AUC invertido en HIKARI v2_main, S3a-min, ε = 1 (no cambia ningún criterio).

1) Semillas de ruido 42–56 (15) con ε ∈ {0,5; 1; 5}, con el mismo mecanismo y RF que S3-B.
   Para cada semilla: ROC-AUC en test limpio y ruidoso, PR-AUC, probabilidad media predicha por
   clase verdadera y nº de probabilidades distintas en el test limpio. Las semillas 42–46 deben
   reproducir los valores guardados en S3-B (comprobación de consistencia).
2) ¿Depende la inversión de pocas variables? (ε = 1, mismas 15 semillas):
   - por variable de S2: dirección de la diferencia de medias atacante − benigno en el train RUIDOSO
     frente al train limpio, y ROC-AUC de la variable en el test LIMPIO orientada con la dirección
     aprendida del train ruidoso;
   - correlación de Spearman, en el test limpio, entre la probabilidad predicha por el RF y cada
     variable, comparada con la correlación entre la etiqueta y esa variable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from _common import path_of, setup  # noqa: E402
from src.data import load_xy  # noqa: E402
from src.dp_local import add_gaussian_noise, analytic_gaussian_sigma, domain_normalize, load_bounds  # noqa: E402
from src.models import create_random_forest  # noqa: E402
from src.scenarios import build_scenario  # noqa: E402
from src.splits import load_split  # noqa: E402

DATASET = "HIKARI2021_v2_main"
EPS = [0.5, 1.0, 5.0]
SEEDS = list(range(42, 57))
TEST_SEED_OFFSET = 1000


def main():
    args, cfg, out = setup("Diagnóstico de inversión del ROC-AUC (S3a-min, v2_main)")
    out = out / "S3_inversion_diagnostic"
    out.mkdir(exist_ok=True)
    X, y = load_xy(cfg, DATASET)
    y = y.to_numpy()
    tr, te = load_split(cfg, DATASET)
    cols = list(build_scenario("S2", X.iloc[:1], cfg, DATASET).columns)
    Xn = domain_normalize(X.iloc[np.concatenate([tr, te])][cols], load_bounds(cfg, DATASET))
    Xtr_c, Xte_c = Xn.iloc[:len(tr)], Xn.iloc[len(tr):]
    y_tr, y_te = y[tr], y[te]
    delta = 1.0 / len(tr) ** 2
    stored = pd.read_csv(path_of(cfg, "results_dir") / "2026-10-07_S3B" / "S3B" / f"{DATASET}_utility_runs.csv")
    stored = stored[(stored.variant == "S3a-min") & (stored.evaluation == "clean_test")]

    # Dirección "verdadera" por variable (train limpio) y correlación etiqueta–variable en el test limpio
    clean_dir = np.sign(Xtr_c[y_tr == 1].mean() - Xtr_c[y_tr == 0].mean())
    label_corr = pd.Series({c: spearmanr(Xte_c[c], y_te).correlation for c in cols})

    rows, var_rows, corr_rows = [], [], []
    for eps in EPS:
        sigma = analytic_gaussian_sigma(eps, delta, np.sqrt(len(cols)))
        for seed in SEEDS:
            Xtr = add_gaussian_noise(Xtr_c, sigma, seed)
            Xte_n = add_gaussian_noise(Xte_c, sigma, seed + TEST_SEED_OFFSET)
            m = create_random_forest(cfg, DATASET)
            m.fit(Xtr, y_tr)
            p_c = m.predict_proba(Xte_c)[:, 1]
            p_n = m.predict_proba(Xte_n)[:, 1]
            r = {"epsilon": eps, "seed": seed, "sigma": sigma,
                 "roc_auc_clean": roc_auc_score(y_te, p_c), "roc_auc_noisy": roc_auc_score(y_te, p_n),
                 "pr_auc_clean": average_precision_score(y_te, p_c),
                 "mean_p_benign_clean": float(p_c[y_te == 0].mean()), "mean_p_attack_clean": float(p_c[y_te == 1].mean()),
                 "mean_p_benign_noisy": float(p_n[y_te == 0].mean()), "mean_p_attack_noisy": float(p_n[y_te == 1].mean()),
                 "n_unique_proba_clean": int(len(np.unique(np.round(p_c, 6))))}
            s = stored[(stored.epsilon == eps) & (stored.seed == seed)]
            r["stored_roc_auc_clean"] = float(s.roc_auc.iloc[0]) if len(s) else np.nan
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if eps == 1.0:
                noisy_dir = np.sign(Xtr[y_tr == 1].mean() - Xtr[y_tr == 0].mean())
                for c in cols:
                    a = roc_auc_score(y_te, Xte_c[c] * noisy_dir[c]) if noisy_dir[c] != 0 else 0.5
                    var_rows.append({"seed": seed, "feature": c, "clean_direction": int(clean_dir[c]),
                                     "noisy_direction": int(noisy_dir[c]),
                                     "direction_agrees": bool(noisy_dir[c] == clean_dir[c]),
                                     "clean_test_auc_with_noisy_direction": a,
                                     "clean_test_auc_with_clean_direction": roc_auc_score(y_te, Xte_c[c] * clean_dir[c])
                                     if clean_dir[c] != 0 else 0.5})
                for c in cols:
                    corr_rows.append({"seed": seed, "feature": c,
                                      "spearman_proba_vs_feature": spearmanr(p_c, Xte_c[c]).correlation,
                                      "spearman_label_vs_feature": label_corr[c]})
            del m
            pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(var_rows).to_csv(out / "per_variable_direction_eps1.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "proba_feature_correlation_eps1.csv", index=False)

    res = pd.DataFrame(rows)
    print(res.groupby("epsilon")[["roc_auc_clean", "roc_auc_noisy"]].describe().round(3).to_string())
    consist = res.dropna(subset=["stored_roc_auc_clean"])
    print("Coincidencia con S3-B (semillas 42–46):",
          bool(np.allclose(consist.roc_auc_clean, consist.stored_roc_auc_clean, atol=1e-9)))


if __name__ == "__main__":
    main()
