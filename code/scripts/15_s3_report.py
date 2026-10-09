"""Informe de S3-B (no entrena): tablas de σ, utilidad frente a ε con IC 95 %, ROC-AUC por semilla
(ε ≤ 10), ventaja del atacante frente a ε y curvas de utilidad (PR-AUC y F1).

Referencias:
- Utilidad: S0 y S2 de la corrida F1 (HIKARI, train completo); en CICIDS, S0-sub y S2-sub entrenados
  sobre la misma submuestra de 500.000 filas que S3.
- Ataques: A1-S3a frente a "S0 publicado" (ventaja = 1) y, como contexto, "S0 sin la columna" de F2;
  A2 frente a S0/S2 de F2; A3 frente a S0/S2 de la corrida A3.
IC 95 % entre semillas: media ± t(n−1; 0,975) · sd / √n.
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from _common import path_of, setup

DATASETS = ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs", "CICIDS2017"]
LABEL = {"HIKARI2021_v2_main": "HIKARI v2_main (principal)", "HIKARI2021_v2_pairs": "HIKARI v2_pairs (sensibilidad)",
         "CICIDS2017": "CICIDS2017 (submuestra 500 k)"}
METRICS = ["f1", "precision", "recall", "fpr", "roc_auc", "pr_auc"]
# Paleta categórica (slots 1–3 validados con validate_palette.js, modo claro); referencias en tinta gris
COLORS = {"S3a": "#2a78d6", "S3a-min": "#eb6834", "S3b": "#1baf7a"}
MARKERS = {"S3a": "o", "S3a-min": "s", "S3b": "^"}
LINESTYLES = {"S3a": "-", "S3a-min": "--", "S3b": ":"}
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a64", "#e4e3dd", "#fcfcfb"


def extra(parser):
    parser.add_argument("--s3-run", default="2026-10-07_S3B")
    parser.add_argument("--f1-run", default="2026-10-06_F1")
    parser.add_argument("--f2-run", default="2026-10-06_F2")
    parser.add_argument("--a3-run", default="2026-10-07_A3")


def ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = float(v.std(ddof=1))
    h = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, sd, m - h, m + h


def summarize(df, keys, metrics):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_seeds"] = len(g)
        for mtr in metrics:
            m, sd, lo, hi = ci(g[mtr])
            r.update({f"{mtr}_mean": m, f"{mtr}_sd": sd, f"{mtr}_ci95_low": lo, f"{mtr}_ci95_high": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def utility_references(root, f1_run):
    full = pd.read_csv(root / f1_run / "utility_report" / "utility_full_test.csv")
    refs = {}
    for d in DATASETS:
        if d == "CICIDS2017":
            u = pd.read_csv(root / "2026-10-07_S3B" / "S3B" / f"{d}_utility_runs.csv")
            s0, s2 = u[u.variant == "S0-sub"].iloc[0], u[u.variant == "S2-sub"].iloc[0]
        else:
            f = full[full.dataset == d].set_index("scenario")
            s0, s2 = f.loc["S0"], f.loc["S2"]
        refs[d] = {"S0": {m: float(s0[m]) for m in ["f1", "pr_auc", "roc_auc"]},
                   "S2": {m: float(s2[m]) for m in ["f1", "pr_auc", "roc_auc"]}}
    return refs


def plot_utility(util, refs, base_rate, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 3.1 * len(DATASETS)), sharex=True)
    for i, d in enumerate(DATASETS):
        for j, (mtr, title) in enumerate([("pr_auc", "PR-AUC"), ("f1", "F1 (umbral validado)")]):
            ax = axes[i, j]
            sub = util[(util.dataset == d) & (util.evaluation == "clean_test")]
            for v in ["S3a", "S3a-min", "S3b"]:
                g = sub[sub.variant == v].sort_values("epsilon")
                if g.empty:
                    continue
                x = g.epsilon.to_numpy()
                ax.fill_between(x, g[f"{mtr}_ci95_low"].clip(0, 1), g[f"{mtr}_ci95_high"].clip(0, 1),
                                color=COLORS[v], alpha=0.15, linewidth=0)
                ax.plot(x, g[f"{mtr}_mean"], color=COLORS[v], lw=2, ls=LINESTYLES[v], marker=MARKERS[v], ms=5,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=v)
            for ref, ls in [("S0", "-"), ("S2", "--")]:
                ax.axhline(refs[d][ref][mtr], color=MUTED, lw=1.2, ls=ls, label=f"{ref} (sin ruido)")
            if mtr == "pr_auc":
                ax.axhline(base_rate[d], color=MUTED, lw=1, ls=":", label="Azar (prevalencia)")
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.set_title(f"{LABEL[d]} — {title}", color=INK, fontsize=9, loc="left")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("ε (escala logarítmica)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Utilidad frente a ε: train ruidoso → test limpio (media e IC 95 % entre semillas)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ["png", "svg"]:
        fig.savefig(out / f"utility_vs_epsilon.{ext}", dpi=200)
    plt.close(fig)


def main():
    args, cfg, out = setup("Informe de S3-B", extra)
    root = path_of(cfg, "results_dir")
    s3 = root / args.s3_run / "S3B"
    out = root / args.s3_run / "S3B_report"
    out.mkdir(exist_ok=True)

    # 1) σ
    sig = pd.read_csv(s3 / "sigma_table.csv")
    sig_w = sig.pivot_table(index=["dataset", "variant", "d", "n_train", "delta"], columns="epsilon", values="sigma").reset_index()
    sig_w.to_csv(out / "sigma_by_epsilon.csv", index=False)

    # 2) Utilidad
    runs = []
    for d in DATASETS:
        runs.append(pd.read_csv(s3 / f"{d}_utility_runs.csv"))
        c = pd.read_csv(s3 / f"{d}_central_runs.csv")
        runs.append(c)
    allu = pd.concat(runs, ignore_index=True)
    noisy = allu[allu.variant.isin(["S3a", "S3a-min", "S3b"])]
    util = summarize(noisy, ["dataset", "variant", "evaluation", "epsilon"], METRICS)
    util.to_csv(out / "utility_summary.csv", index=False)
    refs_rows = allu[allu.variant.isin(["S3-ctrl", "S3-ctrl-min", "S0-sub", "S2-sub"])][["dataset", "variant"] + METRICS]
    refs_rows.to_csv(out / "utility_controls_and_subsample_references.csv", index=False)
    refs = utility_references(root, args.f1_run)
    pd.DataFrame([{"dataset": d, "reference": r, **v} for d, rr in refs.items() for r, v in rr.items()]).to_csv(
        out / "utility_references_S0_S2.csv", index=False)

    # Prevalencia del test (azar de la PR-AUC)
    from src.data import load_raw, make_target
    from src.splits import load_split
    base = {}
    for d in DATASETS:
        y = make_target(load_raw(cfg, d), d, cfg).to_numpy()
        _, te = load_split(cfg, d)
        base[d] = float(y[te].mean())
    plot_utility(util, refs, base, out)

    # 3) ROC-AUC por semilla (ε ≤ 10), test limpio y test ruidoso
    low = noisy[(noisy.epsilon <= 10) & noisy.variant.isin(["S3a", "S3a-min"])]
    low.pivot_table(index=["dataset", "variant", "evaluation", "epsilon"], columns="seed", values="roc_auc").round(4) \
        .to_csv(out / "roc_auc_by_seed_eps_le_10.csv")

    # 4) Ataques
    att = pd.concat([pd.read_csv(s3 / f"{d}_attack_runs.csv") for d in DATASETS], ignore_index=True)
    am = ["accuracy", "macro_f1", "balanced_accuracy", "advantage", "majority_baseline"] + (["roc_auc"] if "roc_auc" in att else [])
    asum = summarize(att, ["dataset", "objective", "variant", "epsilon"], [m for m in am if m in att])
    # Referencias sin ruido
    ref_rows = []
    for d in DATASETS:
        f2 = pd.read_csv(root / args.f2_run / "F2_attribute_inference" / f"{d}_attribute_inference_runs.csv")
        for (obj, sc, atk), g in f2.groupby(["objective", "scenario", "attack"]):
            if sc not in ("S0", "S2"):
                continue
            name = {("S0", "exposed_directly"): "S0 publicado (exposición directa)",
                    ("S0", "inference"): "S0 sin la columna (F2)" if obj.startswith("A1") else "S0 (F2)",
                    ("S2", "inference"): "S2 sin ruido (F2)"}[(sc, atk)]
            ref_rows.append({"dataset": d, "objective": obj, "reference": name,
                             "macro_f1_mean": g.macro_f1.mean(), "advantage_mean": g.advantage.mean(),
                             "balanced_accuracy_mean": g.balanced_accuracy.mean()})
    a3 = pd.read_csv(root / args.a3_run / "A3_same_role_host" / "a3_runs.csv")
    a3 = a3[(a3.dataset == "HIKARI2021_v2_main") & a3.scenario.isin(["S0", "S2"])]
    for (obj, sc), g in a3.groupby(["objective", "scenario"]):
        ref_rows.append({"dataset": "HIKARI2021_v2_main", "objective": obj, "reference": f"{sc} sin ruido (A3)",
                         "macro_f1_mean": g.macro_f1.mean(), "advantage_mean": g.advantage.mean(),
                         "balanced_accuracy_mean": g.balanced_accuracy.mean(), "roc_auc_mean": g.roc_auc.mean()})
    refs_att = pd.DataFrame(ref_rows)
    asum.to_csv(out / "attack_summary_by_epsilon.csv", index=False)
    refs_att.to_csv(out / "attack_references_without_noise.csv", index=False)

    pd.set_option("display.width", 250)
    print(sig_w.round(4).to_string(index=False))
    cols = ["dataset", "variant", "evaluation", "epsilon", "n_seeds", "f1_mean", "f1_ci95_low", "f1_ci95_high",
            "pr_auc_mean", "pr_auc_ci95_low", "pr_auc_ci95_high", "roc_auc_mean"]
    print(util[cols].round(4).to_string(index=False))
    print(refs_rows.round(4).to_string(index=False))
    print(asum[["dataset", "objective", "variant", "epsilon", "n_seeds", "macro_f1_mean", "advantage_mean",
                "balanced_accuracy_mean"] + (["roc_auc_mean"] if "roc_auc_mean" in asum else [])].round(4).to_string(index=False))
    print(refs_att.round(4).to_string(index=False))
    print("prevalencias:", base)


if __name__ == "__main__":
    main()
