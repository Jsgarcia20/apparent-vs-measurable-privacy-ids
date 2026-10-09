"""Figuras finales para publicación (PNG 300 dpi + SVG), sin títulos internos y legibles en blanco y negro.

A) PR-AUC frente a ε (escala log), por dataset: S3a, S3a-min, S3b, S3c, S3c-min con IC 95 % entre semillas;
   líneas horizontales de S0, S2 y LR-ref (convergida, variables de S0). En CICIDS, S0 y S2 son las
   referencias entrenadas en la misma submuestra de 500.000 filas (S0-sub, S2-sub).
B) Plano privacidad–utilidad por dataset: X = ventaja del atacante (A2 en HIKARI, A1 en CICIDS),
   Y = PR-AUC. Puntos S0, S1a, S1b, S2 y S3a/S3a-min por ε (10, 100, 500, 1000); frente de Pareto
   (menor ventaja, mayor PR-AUC) resaltado y ε anotados.
C) Sensibilidad al conocimiento auxiliar: balanced accuracy frente a % del train (A2, A3-DNS, A3-443; S0 y S2).
Colores: paleta categórica validada (validate_palette.js); cada serie lleva además marcador y estilo
de línea propios para que se distinga sin color.
"""
import numpy as np
import pandas as pd

from _common import path_of, setup

DS = ["CICIDS2017", "HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]
DS_LABEL = {"CICIDS2017": "CICIDS2017 (submuestra 500 k)", "HIKARI2021_v2_main": "HIKARI v2_main",
            "HIKARI2021_v2_pairs": "HIKARI v2_pairs"}
STYLE = {  # color, marcador, estilo de línea
    "S3a": ("#2a78d6", "o", "-"), "S3a-min": ("#eb6834", "s", "--"), "S3b": ("#1baf7a", "^", ":"),
    "S3c": ("#eda100", "D", "-."), "S3c-min": ("#e87ba4", "v", (0, (5, 1, 1, 1, 1, 1))),
}
INK, MUTED, GRID = "#2b2b29", "#5f5e58", "#e2e1db"


def extra(parser):
    parser.add_argument("--f1-run", default="2026-10-06_F1")
    parser.add_argument("--f2-run", default="2026-10-06_F2")
    parser.add_argument("--a3-run", default="2026-10-07_A3")
    parser.add_argument("--s3b-run", default="2026-10-07_S3B")
    parser.add_argument("--batch2-run", default="2026-10-07_batch2")


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def panel_label(ax, text):
    """Etiqueta del panel fuera del área de datos (encima del eje, alineada a la izquierda)."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK)


def save(fig, out, name):
    for ext, kw in [("png", {"dpi": 300}), ("svg", {})]:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", **kw)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "font.family": "DejaVu Sans"})
    args, cfg, out = setup("Figuras finales", extra)
    root = path_of(cfg, "results_dir")
    out = out / "figures"
    out.mkdir(exist_ok=True)
    dp = pd.read_csv(out.parent / "phase10" / "phase10_dp_vs_reference.csv")
    f1run = pd.read_csv(root / args.f1_run / "utility_report" / "utility_full_test.csv")
    lrref = pd.read_csv(root / args.batch2_run / "S3c" / "lr_ref_converged.csv")
    lrref = lrref[lrref.converged_without_warning]

    def refs(d):
        if d == "CICIDS2017":
            u = pd.read_csv(root / args.s3b_run / "S3B" / f"{d}_utility_runs.csv")
            s0, s2 = u[u.variant == "S0-sub"].pr_auc.iloc[0], u[u.variant == "S2-sub"].pr_auc.iloc[0]
        else:
            f = f1run[f1run.dataset == d].set_index("scenario")
            s0, s2 = f.loc["S0", "pr_auc"], f.loc["S2", "pr_auc"]
        lr = lrref[(lrref.dataset == d) & (lrref.variant == "LR-ref (S3c)")].pr_auc.iloc[-1]
        return {"S0": s0, "S2": s2, "LR-ref": lr}

    # ---------- Figura A ----------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, d in zip(axes, DS):
        g = dp[dp.dataset == d]
        for v, (col, mk, ls) in STYLE.items():
            x = g[g.variant == v].sort_values("epsilon")
            ax.fill_between(x.epsilon, x.pr_auc_ci95_low.clip(0, 1), x.pr_auc_ci95_high.clip(0, 1), color=col,
                            alpha=0.15, lw=0)
            ax.plot(x.epsilon, x.pr_auc_mean, color=col, marker=mk, ls=ls, lw=1.6, ms=4.5, mec="white", mew=0.6, label=v)
        for name, lls in [("S0", "-"), ("S2", "--"), ("LR-ref", ":")]:
            ax.axhline(refs(d)[name], color=MUTED, lw=1.0, ls=lls,
                       label={"S0": "S0 sin ruido", "S2": "S2 sin ruido", "LR-ref": "LR-ref (sin DP)"}[name])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("ε (escala logarítmica)")
        style_axes(ax)
        panel_label(ax, DS_LABEL[d])
    axes[0].set_ylabel("PR-AUC (test limpio)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, out, "figA_utility_vs_epsilon")
    plt.close(fig)

    # ---------- Figura B ----------
    f2 = {d: pd.read_csv(root / args.f2_run / "F2_attribute_inference" / f"{d}_attribute_inference_runs.csv") for d in DS}
    att = pd.concat([pd.read_csv(root / args.s3b_run / "S3B" / f"{d}_attack_runs.csv") for d in DS])
    s3u = pd.concat([pd.read_csv(root / args.s3b_run / "S3B" / f"{d}_utility_runs.csv") for d in DS])
    s3u = s3u[s3u.evaluation == "clean_test"]
    points_all = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    struct_style = {"S0": ("#2b2b29", "*", 10), "S1a": ("#2b2b29", "P", 7), "S1b": ("#2b2b29", "X", 7),
                    "S2": ("#2b2b29", "d", 7)}
    for ax, d in zip(axes, DS):
        obj = "A1_destination_port" if d == "CICIDS2017" else "A2_source_host"
        f = f2[d][f2[d].objective == obj]
        pts = []
        for sc, internal in [("S0", "S0"), ("S1a", "S1"), ("S1b", "S1b"), ("S2", "S2")]:
            ff = f[f.scenario == sc]
            adv = 1.0 if (obj.startswith("A1") and sc == "S0") else ff[ff.attack == "inference"].advantage.mean()
            pr = f1run[(f1run.dataset == d) & (f1run.scenario == internal)].pr_auc.iloc[0]
            pts.append({"label": sc, "family": sc, "eps": np.nan, "x": adv, "y": pr})
        for v in ["S3a", "S3a-min"]:
            a = att[(att.dataset == d) & (att.objective == obj) & (att.variant == v)].groupby("epsilon").advantage.mean()
            u = s3u[(s3u.dataset == d) & (s3u.variant == v)].groupby("epsilon").pr_auc.mean()
            for eps in a.index:
                pts.append({"label": v, "family": v, "eps": eps, "x": a[eps], "y": u[eps]})
        P = pd.DataFrame(pts).assign(dataset=d, attack_objective=obj)
        # Frente de Pareto: no dominados (x menor es mejor, y mayor es mejor)
        dom = [((P.x <= r.x) & (P.y >= r.y) & ((P.x < r.x) | (P.y > r.y))).any() for r in P.itertuples()]
        P["pareto"] = ~np.array(dom)
        points_all.append(P)
        fr = P[P.pareto].sort_values("x")
        ax.plot(fr.x, fr.y, color=MUTED, lw=1.2, ls="-", zorder=1, label="Frente de Pareto")
        for r in P.itertuples():
            if r.family in struct_style:
                col, mk, ms = struct_style[r.family]
            else:
                col, mk, _ = STYLE[r.family]
                ms = 6
            ax.plot(r.x, r.y, marker=mk, color=col, ms=ms, ls="none", mec=INK if r.pareto else "white",
                    mew=1.2 if r.pareto else 0.5, zorder=3)
            if not np.isnan(r.eps):   # S3a: arriba a la izquierda; S3a-min: abajo a la derecha
                above = r.family == "S3a"
                ax.annotate(f"ε={r.eps:g}", (r.x, r.y), textcoords="offset points",
                            xytext=(-5, 5) if above else (5, -10), ha="right" if above else "left",
                            fontsize=6.5, color=INK)
        ax.set_xlabel("Ventaja del atacante" + (" A1 (puerto)" if obj.startswith("A1") else " A2 (host)"))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        style_axes(ax)
        panel_label(ax, DS_LABEL[d].replace(" (submuestra 500 k)", ""))
    axes[0].set_ylabel("PR-AUC")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=s if k == "S0" else 7, label=k) for k, (c, m, s) in struct_style.items()]
    handles += [Line2D([], [], color=STYLE[v][0], marker=STYLE[v][1], ls="none", ms=6, label=v) for v in ["S3a", "S3a-min"]]
    handles += [Line2D([], [], color=MUTED, lw=1.2, label="Frente de Pareto"),
                Line2D([], [], color="white", marker="o", mec=INK, mew=1.2, ls="none", label="Punto en el frente")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, out, "figB_privacy_utility_pareto")
    plt.close(fig)
    pd.concat(points_all).to_csv(out / "figB_points.csv", index=False)

    # ---------- Figura C ----------
    aux = pd.read_csv(root / args.batch2_run / "aux_knowledge" / "aux_knowledge_runs.csv")
    a3 = pd.read_csv(root / args.a3_run / "A3_same_role_host" / "a3_runs.csv")
    full = []
    for d in ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]:
        f = f2[d][(f2[d].objective == "A2_source_host") & (f2[d].attack == "inference")]
        for sc, g in f.groupby("scenario"):
            full.append({"dataset": d, "objective": "A2_source_host", "scenario": sc, "fraction": 1.0,
                         "balanced_accuracy": g.balanced_accuracy.mean()})
    for (d, o, sc), g in a3.groupby(["dataset", "objective", "scenario"]):
        full.append({"dataset": d, "objective": o, "scenario": sc, "fraction": 1.0, "balanced_accuracy": g.balanced_accuracy.mean()})
    agg = aux.groupby(["dataset", "objective", "scenario", "fraction"]).balanced_accuracy.agg(["mean", "std"]).reset_index()
    agg = pd.concat([agg.rename(columns={"mean": "balanced_accuracy"}), pd.DataFrame(full)])
    agg.to_csv(out / "figC_points.csv", index=False)
    objs = [("A2_source_host", "A2: host de origen (top-10 + other)"), ("A3_DNS", "A3-DNS: .23 frente a .150"),
            ("A3_443", "A3-443: .23 frente a .150")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True)
    for i, d in enumerate(["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]):
        for j, (o, olab) in enumerate(objs):
            ax = axes[i, j]
            for sc, col, mk, ls in [("S0", "#2a78d6", "o", "-"), ("S2", "#eb6834", "s", "--")]:
                g = agg[(agg.dataset == d) & (agg.objective == o) & (agg.scenario == sc)].sort_values("fraction")
                ax.plot(g.fraction * 100, g.balanced_accuracy, color=col, marker=mk, ls=ls, lw=1.6, ms=5,
                        mec="white", mew=0.6, label=f"{sc}" + (" (todas las variables)" if sc == "S0" else " (solo LOW)"))
            ax.axhline(0.5, color=MUTED, lw=1, ls=":", label="Azar (0,5)")
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 20, 100])
            ax.set_xticklabels(["1 %", "5 %", "20 %", "100 %"])
            ax.set_ylim(0.45, 1.0)
            style_axes(ax)
            panel_label(ax, f"{DS_LABEL[d]} · {olab}")
            if i == 1:
                ax.set_xlabel("Conocimiento auxiliar (% del train)")
            if j == 0:
                ax.set_ylabel("Balanced accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, out, "figC_aux_knowledge")
    plt.close(fig)
    print("Figuras guardadas en", out)


if __name__ == "__main__":
    main()
