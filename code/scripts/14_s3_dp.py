"""S3-B — Privacidad diferencial: utilidad (S3a local y S3b central) y ataques T1 sobre datos ruidosos.

Partes (--part):
  sigma    Tabla de σ por ε y variante (Gaussiano analítico; Δ₂ = √d; δ = 1/N_train²).
  utility  S3a (variables de S0), S3a-min (variables de S2) y controles sin ruido (S3-ctrl, S3-ctrl-min).
           Semillas de ruido 42–46 (CICIDS: 42–44).
           Train ruidoso → test limpio (principal) y train ruidoso → test ruidoso (diagnóstico; misma ε,
           semilla + 1000). RF con random_state = 42 y la configuración validada del dataset.
  central  S3b: diffprivlib.models.RandomForestClassifier con sus hiperparámetros por defecto,
           bounds = (0, 1), classes = [0, 1]; ε ∈ {0.1, 0.5, 1, 5, 10}, 5 semillas, test limpio normalizado.
  attacks  Amenaza T1 con datos publicados ruidosos (train y test perturbados con la misma ε y semillas
           distintas): A1 (puerto), A2 (host, HIKARI), A3-DNS y A3-443 (solo v2_main). Variantes S3a y S3a-min;
           ε ∈ {10, 100, 500, 1000}; semillas 42, 43, 44 (ruido del train = s, ruido del test = s + 1000, atacante = s).
           El atacante recibe TODAS las columnas publicadas con ruido, incluida la columna objetivo si se
           publica (A1 en S3a: el puerto con ruido). Referencia de A1-S3a: S0 publicado (ventaja = 1).
  checks   Comprobaciones automáticas (control frente a S0, mejora con ε, ROC-AUC < 0,45, σ decreciente).

CICIDS2017: submuestra estratificada FIJA de 500.000 filas del train (random_state = 42), guardada como
experimental_splits/CICIDS2017_train_sub500k_indices.npy; test completo. Referencias: S0-sub y S2-sub.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import add_datasets_arg, path_of, setup
from src.attack import attack_metrics, attacker
from src.data import load_raw, make_predictors, make_target
from src.dp_local import add_gaussian_noise, analytic_gaussian_sigma, domain_normalize, load_bounds
from src.models import create_random_forest
from src.s3_common import A3_HOSTS, SUBSAMPLE, VARIANTS, attack_tasks, prepare, training_rows, utility_metrics
from src.scenarios import build_scenario

EPS_LOCAL = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
EPS_CENTRAL = [0.1, 0.5, 1, 5, 10]
EPS_ATTACK = [10, 100, 500, 1000]          # decisión 2026-10-07: sin ε = 1 (azar), con ε = 500
NOISE_SEEDS = [42, 43, 44, 45, 46]
NOISE_SEEDS_BY_DATASET = {"CICIDS2017": [42, 43, 44]}   # recorte de tiempo aprobado (opción 1)
ATTACK_SEEDS = [42, 43, 44]
TEST_SEED_OFFSET = 1000


def extra(parser):
    add_datasets_arg(parser, ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"])
    parser.add_argument("--part", required=True, choices=["sigma", "utility", "central", "attacks", "checks"])
    parser.add_argument("--reference-run", default="2026-10-06_F1", help="corrida con S0/S2 completos (HIKARI)")
    parser.add_argument("--roc-alerts-nonblocking", action="store_true",
                        help="decisión 2026-10-07: las alertas de ROC-AUC < 0,45 se registran pero no detienen; "
                             "las otras tres comprobaciones sí detienen")


def sigma_table(datasets, cfg):
    rows = []
    for dataset in datasets:
        df = load_raw(cfg, dataset)
        y = make_target(df, dataset, cfg).to_numpy()
        tr, _ = training_rows(cfg, dataset, y)
        X0 = make_predictors(df.iloc[:1], cfg, dataset)
        delta = 1.0 / len(tr) ** 2
        for v, s in VARIANTS.items():
            d = build_scenario(s, X0, cfg, dataset).shape[1]
            for eps in EPS_LOCAL:
                rows.append({"dataset": dataset, "variant": v, "d": d, "sensitivity_l2": float(np.sqrt(d)),
                             "n_train": len(tr), "delta": delta, "epsilon": eps,
                             "sigma": analytic_gaussian_sigma(eps, delta, np.sqrt(d))})
    return pd.DataFrame(rows)


def part_utility(cfg, dataset, out):
    df, X, y, tr, te, cols = prepare(cfg, dataset)
    del df
    thr = cfg["datasets"][dataset]["decision_threshold"]
    Xn = domain_normalize(X.iloc[np.concatenate([tr, te])], load_bounds(cfg, dataset))
    Xn_tr, Xn_te = Xn.iloc[:len(tr)], Xn.iloc[len(tr):]
    y_tr, y_te = y[tr], y[te]
    delta = 1.0 / len(tr) ** 2
    path = out / f"{dataset}_utility_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["variant"], r["epsilon"], r["seed"]) for r in rows}

    def fit_eval(name, Xtr, Xte_list, eps=np.nan, seed=-1, sigma=0.0):
        m = create_random_forest(cfg, dataset)
        t0 = time.perf_counter()
        m.fit(Xtr, y_tr)
        secs = time.perf_counter() - t0
        for evaluation, Xte in Xte_list:
            r = {"dataset": dataset, "variant": name, "evaluation": evaluation, "epsilon": eps, "seed": seed,
                 "sigma": sigma, "n_features": Xtr.shape[1], "n_train": len(y_tr),
                 **utility_metrics(y_te, m.predict_proba(Xte)[:, 1], thr), "fit_seconds": secs}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    # Referencias: controles sin ruido (y, en CICIDS, S0-sub y S2-sub con las variables originales)
    for v, c in cols.items():
        ctrl = "S3-ctrl" if v == "S3a" else "S3-ctrl-min"
        if (ctrl, "nan", -1) not in {(r["variant"], str(r["epsilon"]), r["seed"]) for r in rows}:
            fit_eval(ctrl, Xn_tr[c], [("clean_test", Xn_te[c])])
        if dataset in SUBSAMPLE:
            ref = "S0-sub" if v == "S3a" else "S2-sub"
            if ref not in {r["variant"] for r in rows}:
                fit_eval(ref, X.iloc[tr][c], [("clean_test", X.iloc[te][c])])
    for v, c in cols.items():
        sens = np.sqrt(len(c))
        for eps in EPS_LOCAL:
            sigma = analytic_gaussian_sigma(eps, delta, sens)
            for seed in NOISE_SEEDS_BY_DATASET.get(dataset, NOISE_SEEDS):
                if (v, eps, seed) in done:
                    continue
                Xtr = add_gaussian_noise(Xn_tr[c], sigma, seed)
                Xte_noisy = add_gaussian_noise(Xn_te[c], sigma, seed + TEST_SEED_OFFSET)
                fit_eval(v, Xtr, [("clean_test", Xn_te[c]), ("noisy_test", Xte_noisy)], eps, seed, sigma)


def part_central(cfg, dataset, out):
    from diffprivlib.models import RandomForestClassifier as DPRandomForest
    df, X, y, tr, te, cols = prepare(cfg, dataset)
    del df
    thr = cfg["datasets"][dataset]["decision_threshold"]
    c = cols["S3a"]
    Xn = domain_normalize(X.iloc[np.concatenate([tr, te])][c], load_bounds(cfg, dataset))
    Xtr, Xte = Xn.iloc[:len(tr)].to_numpy(), Xn.iloc[len(tr):].to_numpy()
    rows = []
    for eps in EPS_CENTRAL:
        for seed in NOISE_SEEDS:
            m = DPRandomForest(epsilon=eps, bounds=(0, 1), classes=[0, 1], random_state=seed)
            t0 = time.perf_counter()
            m.fit(Xtr, y[tr])
            r = {"dataset": dataset, "variant": "S3b", "evaluation": "clean_test", "epsilon": eps, "seed": seed,
                 "n_features": Xtr.shape[1], "n_train": len(tr),
                 "n_estimators": m.n_estimators, "max_depth": m.max_depth,
                 **utility_metrics(y[te], m.predict_proba(Xte)[:, list(m.classes_).index(1)], thr),
                 "fit_seconds": time.perf_counter() - t0}
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(out / f"{dataset}_central_runs.csv", index=False)


def part_attacks(cfg, dataset, out):
    df, X, y, tr, te, cols = prepare(cfg, dataset)
    bounds = load_bounds(cfg, dataset)
    tasks = attack_tasks(dataset, df, X, y, tr, te)
    del df
    delta = 1.0 / len(tr) ** 2
    path = out / f"{dataset}_attack_runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["objective"], r["variant"], r["epsilon"], r["seed"]) for r in rows}
    for obj, target_col, labels, a_tr, a_te in tasks:
        classes = pd.Series(labels[a_tr]).value_counts().index.tolist()
        majority = classes[0]
        Xn_tr = domain_normalize(X.iloc[a_tr], bounds)
        Xn_te = domain_normalize(X.iloc[a_te], bounds)
        for v, c in cols.items():
            sens = np.sqrt(len(c))       # el ruido se calibra sobre todas las columnas publicadas
            feats = list(c)              # incluye la columna objetivo con ruido si se publica
            for eps in EPS_ATTACK:
                sigma = analytic_gaussian_sigma(eps, delta, sens)
                for seed in ATTACK_SEEDS:
                    if (obj, v, eps, seed) in done:
                        continue
                    Ptr = add_gaussian_noise(Xn_tr[c], sigma, seed)[feats]
                    Pte = add_gaussian_noise(Xn_te[c], sigma, seed + TEST_SEED_OFFSET)[feats]
                    m = attacker(cfg, seed)
                    t0 = time.perf_counter()
                    m.fit(Ptr, labels[a_tr])
                    proba = m.predict_proba(Pte)
                    pred = m.classes_[np.argmax(proba, axis=1)]
                    r = {"dataset": dataset, "objective": obj, "variant": v, "epsilon": eps, "seed": seed,
                         "sigma": sigma, "target_column_published_noisy": bool(target_col in c), "n_features": len(feats), "n_train": len(a_tr), "n_test": len(a_te),
                         "n_classes": len(classes), **attack_metrics(labels[a_te], pred, classes, majority),
                         "fit_seconds": time.perf_counter() - t0}
                    if len(classes) == 2:
                        pos = A3_HOSTS[1]
                        r["roc_auc"] = float(roc_auc_score(labels[a_te] == pos, proba[:, list(m.classes_).index(pos)]))
                    rows.append(r)
                    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
                    pd.DataFrame(rows).to_csv(path, index=False)
                    del m


def part_checks(cfg, datasets, out, reference_run):
    """Comprobaciones automáticas antes de continuar. Devuelve la lista de (tipo, mensaje)."""
    alerts, report = [], {}
    sig = pd.read_csv(out / "sigma_table.csv")
    for (d, v), g in sig.groupby(["dataset", "variant"]):
        if not (np.diff(g.sort_values("epsilon").sigma.to_numpy()) < 0).all():
            alerts.append(("sigma", f"{d} {v}: σ no es estrictamente decreciente en ε"))
    ref_dir = path_of(cfg, "results_dir") / reference_run
    for dataset in datasets:
        u = pd.read_csv(out / f"{dataset}_utility_runs.csv")
        if dataset in SUBSAMPLE:
            ref = u[u.variant.isin(["S0-sub", "S2-sub"])].set_index("variant")
            ref_s0, ref_s2 = ref.loc["S0-sub"], ref.loc["S2-sub"]
        else:
            full = pd.read_csv(ref_dir / "utility_report" / "utility_full_test.csv")
            full = full[full.dataset == dataset].set_index("scenario")
            ref_s0, ref_s2 = full.loc["S0"], full.loc["S2"]
        rep = {}
        for ctrl, ref in [("S3-ctrl", ref_s0), ("S3-ctrl-min", ref_s2)]:
            c = u[u.variant == ctrl].iloc[0]
            dif = {"f1": float(c.f1 - ref.f1), "pr_auc": float(c.pr_auc - ref.pr_auc)}
            rep[ctrl] = dif
            if max(abs(dif["f1"]), abs(dif["pr_auc"])) > 0.01:
                alerts.append(("control", f"{dataset} {ctrl}: difiere de la referencia en más de 0,01 ({dif})"))
        main = u[(u.evaluation == "clean_test") & u.variant.isin(VARIANTS)]
        means = main.groupby(["variant", "epsilon"])[["f1", "pr_auc", "roc_auc"]].mean().reset_index()
        for v, g in means.groupby("variant"):
            lo, hi = g[g.epsilon == min(EPS_LOCAL)].iloc[0], g[g.epsilon == max(EPS_LOCAL)].iloc[0]
            rep[f"{v}_eps_min_vs_max"] = {"pr_auc": [float(lo.pr_auc), float(hi.pr_auc)], "f1": [float(lo.f1), float(hi.f1)]}
            if not (hi.pr_auc > lo.pr_auc):
                alerts.append(("no_improvement", f"{dataset} {v}: la PR-AUC media no mejora de ε={min(EPS_LOCAL)} a ε={max(EPS_LOCAL)}"))
        low = means[means.roc_auc < 0.45]
        for _, r in low.iterrows():
            alerts.append(("roc_auc", f"{dataset} {r.variant} ε={r.epsilon}: ROC-AUC medio {r.roc_auc:.3f} < 0,45"))
        # ROC-AUC por semilla para ε ≤ 10 (entregable para el artículo)
        low_eps = main[main.epsilon <= 10].pivot_table(index=["variant", "epsilon"], columns="seed", values="roc_auc")
        low_eps.to_csv(out / f"{dataset}_roc_auc_by_seed_eps_le_10.csv")
        report[dataset] = rep
    report["alerts"] = [{"type": t, "message": m} for t, m in alerts]
    (out / f"checks_{'_'.join(datasets)}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return alerts


def main():
    args, cfg, out = setup("S3-B: privacidad diferencial", extra)
    out = out / "S3B"
    out.mkdir(exist_ok=True)
    if args.part == "sigma":
        t = sigma_table(args.datasets, cfg)
        path = out / "sigma_table.csv"
        if path.exists():
            t = pd.concat([pd.read_csv(path), t]).drop_duplicates(["dataset", "variant", "epsilon"], keep="last")
        t.to_csv(path, index=False)
        print(t.to_string(index=False))
        return
    if args.part == "checks":
        alerts = part_checks(cfg, args.datasets, out, args.reference_run)
        blocking = [m for t, m in alerts if not (args.roc_alerts_nonblocking and t == "roc_auc")]
        if blocking:
            raise SystemExit("ALERTAS EN LAS COMPROBACIONES: detenerse e informar.")
        if alerts:
            print("Solo alertas de ROC-AUC: registradas y aceptadas (decisión 2026-10-07); se continúa.")
        return
    for dataset in args.datasets:
        {"utility": part_utility, "central": part_central, "attacks": part_attacks}[args.part](cfg, dataset, out)


if __name__ == "__main__":
    main()
