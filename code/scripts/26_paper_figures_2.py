"""Figuras adicionales del artículo (en inglés), con src/plot_style.py. Solo redibuja resultados guardados.

  fig_2_apparent_privacy              CICIDS2017 y HIKARI-2021 (main): (a, c) ΔF1 de S1a, S1b y S2 frente a S0
                                      con el IC 95 % del bootstrap pareado (fase 10) y el margen de equivalencia;
                                      (b, d) éxito del atacante (A1, A2, A3).
  fig_S4_apparent_privacy_hikari_pairs  la misma figura para HIKARI-2021 (pairs) (suplementaria).
  fig_3_feature_families              caída de macro-F1 del atacante al permutar cada familia (S2).
  fig_S2_hikari_audit                 (a) flujos por categoría en original, pairs y main; (b) PR-AUC de S0.
  fig_S3_roc_auc_by_seed              ROC-AUC por semilla con ε ≤ 10 (S3a, S3a-min; test limpio).
"""
import numpy as np
import pandas as pd

from _common import REPO_ROOT, path_of, setup
from src import plot_style as ps

OUT = REPO_ROOT.parent / "results" / "paper_figures"
SCEN = ["S0", "S1a", "S1b", "S2"]
INTERNAL = {"S0": "S0", "S1a": "S1", "S1b": "S1b", "S2": "S2"}
# Series de ataque: tonos neutros (los colores quedan reservados a los escenarios) + marcador propio
ATTACK_STYLE = {
    "A1": ("A1 (destination port), advantage", "#000000", "o"),
    "A2": ("A2 (source host), advantage", "#555555", "s"),
    "A3_DNS": ("A3 (same-role hosts), port 53, balanced accuracy", "#8a8a8a", "^"),
    "A3_443": ("A3 (same-role hosts), port 443, balanced accuracy", "#b5b5b5", "D"),
}
EQ_MARGIN = 0.005                 # margen de equivalencia (TOST)
DELTA_YLIM = (-0.025, 0.010)      # eje fijo, igual en la Fig. 2 y en la S4
FAMILY_LABEL = {"payload_packet_sizes": "Payload / packet sizes", "timing": "Timing", "header_sizes": "Headers",
                "tcp_window": "TCP windows", "flags": "Flags"}


def extra(parser):
    parser.add_argument("--f1-run", default="2026-10-06_F1")
    parser.add_argument("--f2-run", default="2026-10-06_F2")
    parser.add_argument("--a3-run", default="2026-10-07_A3")
    parser.add_argument("--s3b-run", default="2026-10-07_S3B")
    parser.add_argument("--batch2-run", default="2026-10-07_batch2")
    parser.add_argument("--partb-run", default="2026-10-06_partB")
    parser.add_argument("--d2-run", default="2026-10-06_D2")
    parser.add_argument("--final-run", default="2026-10-07_final")


def main():
    args, cfg, _ = setup("Figuras del artículo (2)", extra)
    root = path_of(cfg, "results_dir")
    plt = ps.apply()
    from matplotlib.lines import Line2D

    # ---------- datos comunes ----------
    f1run = pd.read_csv(root / args.f1_run / "utility_report" / "utility_full_test.csv")
    a3 = pd.read_csv(root / args.a3_run / "A3_same_role_host" / "a3_runs.csv")
    phase10 = pd.read_csv(root / args.final_run / "phase10" / "phase10_structural.csv")

    def attack_rows(d):
        f2 = pd.read_csv(root / args.f2_run / "F2_attribute_inference" / f"{d}_attribute_inference_runs.csv")
        rows = []
        for sc in SCEN:
            g1 = f2[(f2.objective == "A1_destination_port") & (f2.scenario == sc)]
            if sc == "S0":
                rows.append({"attack": "A1", "scenario": sc, "mean": 1.0, "sd": 0.0, "exposed": True})
            else:
                inf = g1[g1.attack == "inference"].advantage
                rows.append({"attack": "A1", "scenario": sc, "mean": inf.mean(), "sd": inf.std(ddof=1), "exposed": False})
            if d.startswith("HIKARI"):
                g2 = f2[(f2.objective == "A2_source_host") & (f2.scenario == sc) & (f2.attack == "inference")].advantage
                rows.append({"attack": "A2", "scenario": sc, "mean": g2.mean(), "sd": g2.std(ddof=1), "exposed": False})
                for obj in ["A3_DNS", "A3_443"]:
                    g3 = a3[(a3.dataset == d) & (a3.objective == obj) & (a3.scenario == sc)].balanced_accuracy
                    rows.append({"attack": obj, "scenario": sc, "mean": g3.mean(), "sd": g3.std(ddof=1), "exposed": False})
        return pd.DataFrame(rows)

    def apparent_privacy(datasets, name):
        fig, axes = ps.figure(180, 62 * len(datasets) + 20, nrows=len(datasets), ncols=2,
                              gridspec_kw={"width_ratios": [1, 1.35]})
        axes = np.atleast_2d(axes)
        data_rows = []
        for i, d in enumerate(datasets):
            # ΔF1 frente a S0 (semilla 42) con el IC 95 % del bootstrap pareado de la fase 10
            ax = axes[i, 0]
            ax.axhspan(-EQ_MARGIN, EQ_MARGIN, color="#e6e6e6", lw=0, zorder=0)
            ax.text(len(SCEN) - 0.55, EQ_MARGIN, "Equivalence margin (±0.005)", ha="right", va="bottom",
                    fontsize=7, color=ps.INK)
            ax.axhline(0, color=ps.INK, lw=0.6, zorder=1)
            st10 = phase10[phase10.dataset == d].set_index("comparison")
            for k, sc in enumerate(SCEN):
                _, col, mk, _ = ps.SCENARIOS[sc]
                if sc == "S0":
                    m, lo, hi = 0.0, 0.0, 0.0
                else:
                    r = st10.loc[f"{sc} vs S0"]
                    m, lo, hi = r.delta_f1, r.delta_f1_ci_low, r.delta_f1_ci_high
                ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, mec=ps.INK, mew=0.4,
                            ms=8 if sc == "S0" else 6, capsize=3, elinewidth=1.0, zorder=3)
                data_rows.append({"dataset": d, "panel": "utility", "scenario": sc,
                                  "series": "Delta F1 vs S0 (seed 42; paired bootstrap 95% CI)",
                                  "mean": m, "ci95_low": lo, "ci95_high": hi})
            ax.set_xticks(range(len(SCEN)))
            ax.set_xticklabels(SCEN)
            ax.set_xlim(-0.5, len(SCEN) - 0.5)
            ax.set_ylim(*DELTA_YLIM)
            ax.set_ylabel("ΔF1 vs. S0")
            ps.style_axes(ax)
            ps.panel(ax, 2 * i, f"{ps.DATASETS[d]}: detection utility")

            ax = axes[i, 1]
            A = attack_rows(d)
            attacks = [a for a in ATTACK_STYLE if a in set(A.attack)]
            width = 0.16
            offsets = {a: (j - (len(attacks) - 1) / 2) * width for j, a in enumerate(attacks)}
            for a in attacks:
                lab, col, mk = ATTACK_STYLE[a]
                g = A[A.attack == a].set_index("scenario").loc[SCEN]
                for k, sc in enumerate(SCEN):
                    r = g.loc[sc]
                    x = k + offsets[a]
                    if r.exposed:
                        ax.plot(x, r["mean"], marker=mk, ls="none", ms=6, mfc="white", mec=col, mew=1.3)
                        ax.annotate("directly\nexposed", (x, r["mean"]), textcoords="offset points", xytext=(0, -24),
                                    ha="center", fontsize=6.5, color=ps.INK)
                    else:
                        ax.errorbar(x, r["mean"], yerr=r.sd, fmt=mk, color=col, mec=ps.INK, mew=0.4, ms=5.5,
                                    capsize=2, elinewidth=0.9)
                    data_rows.append({"dataset": d, "panel": "attacker", "scenario": sc, "series": lab,
                                      "mean": r["mean"], "sd": r.sd, "directly_exposed": bool(r.exposed)})
            ax.axhline(0.0, color=ps.CHANCE[1], lw=1.0, ls=ps.CHANCE[2])
            if any(a.startswith("A3") for a in attacks):
                ax.axhline(0.5, color=ps.CHANCE[1], lw=1.0, ls="-.")
            ax.set_xticks(range(len(SCEN)))
            ax.set_xticklabels(SCEN)
            ax.set_xlim(-0.5, len(SCEN) - 0.5)
            ax.set_ylim(-0.08, 1.08)
            ax.set_ylabel("Attacker success")
            ps.style_axes(ax)
            ps.panel(ax, 2 * i + 1, f"{ps.DATASETS[d]}: attacker success")
        handles = [Line2D([], [], ls="none", marker=mk, color=col, mec=ps.INK, mew=0.4, ms=5.5, label=lab)
                   for a, (lab, col, mk) in ATTACK_STYLE.items()]
        handles += [Line2D([], [], ls="none", marker="o", mfc="white", mec="#000000", mew=1.3, ms=6,
                           label="A1 at S0: port published unchanged (advantage = 1)"),
                    Line2D([], [], color=ps.CHANCE[1], ls=ps.CHANCE[2], label="Chance level (advantage = 0)"),
                    Line2D([], [], color=ps.CHANCE[1], ls="-.", label="Chance level (balanced accuracy = 0.5)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02 - 0.06 / len(datasets)))
        fig.tight_layout(rect=(0, 0.13 if len(datasets) > 1 else 0.24, 1, 1))
        ps.save(fig, OUT, name)
        plt.close(fig)
        pd.DataFrame(data_rows).to_csv(OUT / f"{name}_data.csv", index=False)

    apparent_privacy(["CICIDS2017", "HIKARI2021_v2_main"], "fig_2_apparent_privacy")
    apparent_privacy(["HIKARI2021_v2_pairs"], "fig_S4_apparent_privacy_hikari_pairs")

    # ---------- Figura 3: familias ----------
    cols = [("CICIDS2017", "A1_destination_port", "A1, CICIDS2017"),
            ("HIKARI2021_v2_main", "A1_destination_port", "A1, HIKARI-2021 (main)"),
            ("HIKARI2021_v2_main", "A2_source_host", "A2, HIKARI-2021 (main)")]
    fams = list(FAMILY_LABEL)
    M = np.zeros((len(fams), len(cols)))
    for j, (d, obj, _) in enumerate(cols):
        p = pd.read_csv(root / args.f2_run / "F2_attribute_inference" / f"{d}_S2_family_permutation.csv")
        p = p[p.objective == obj].set_index("family")
        M[:, j] = [p.loc[f, "macro_f1_drop_mean"] for f in fams]
    pd.DataFrame(M, index=[FAMILY_LABEL[f] for f in fams], columns=[c[2] for c in cols]).to_csv(
        OUT / "fig_3_feature_families_data.csv")
    fig, ax = ps.figure(ps.SINGLE_COLUMN_MM, 78)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.8, float(M.max())), aspect="auto")
    ax.xaxis.tick_top()   # etiquetas de columna arriba; la barra de color va debajo, en horizontal
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.45 else ps.INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["A1\nCICIDS2017", "A1\nHIKARI-2021\n(main)", "A2\nHIKARI-2021\n(main)"], fontsize=7)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in fams])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.06, aspect=30)
    cb.set_label("Drop in attacker macro-F1", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    ps.save(fig, OUT, "fig_3_feature_families")
    plt.close(fig)

    # ---------- Figura S2: auditoría de HIKARI ----------
    d2 = root / args.d2_run / "D2_hikari_clean_v2"
    tabs = {v: pd.read_csv(d2 / f"HIKARI2021_v2_{v}_before_after_by_category.csv") for v in ["pairs", "main"]}
    cats = ["Benign", "Background", "Probing", "Bruteforce", "Bruteforce-XML", "XMRIGCC CryptoMiner"]
    counts = pd.DataFrame({
        "Original": tabs["pairs"].set_index("traffic_category").loc[cats, "before"],
        "Pairs": tabs["pairs"].set_index("traffic_category").loc[cats, "after"],
        "Main": tabs["main"].set_index("traffic_category").loc[cats, "after"]})
    counts.to_csv(OUT / "fig_S2_hikari_audit_counts.csv")
    disc = pd.read_csv(root / args.partb_run / "discrimination_results.csv")
    pr = {"Original": float(disc[(disc.dataset == "HIKARI2021") & (disc.scenario == "S0")].pr_auc.iloc[0]),
          "Pairs": float(f1run[(f1run.dataset == "HIKARI2021_v2_pairs") & (f1run.scenario == "S0")].pr_auc.iloc[0]),
          "Main": float(f1run[(f1run.dataset == "HIKARI2021_v2_main") & (f1run.scenario == "S0")].pr_auc.iloc[0])}
    pd.Series(pr, name="S0_pr_auc").to_csv(OUT / "fig_S2_hikari_audit_pr_auc.csv")
    version_style = {"Original": ("#d9d9d9", "///"), "Pairs": ("#969696", "..."), "Main": ("#252525", "")}
    fig, axes = ps.figure(180, 72, ncols=2, gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    w = 0.26
    for j, v in enumerate(counts.columns):
        col, hatch = version_style[v]
        ax.bar(np.arange(len(cats)) + (j - 1) * w, counts[v].replace(0, np.nan), width=w, color=col, edgecolor=ps.INK,
               linewidth=0.4, hatch=hatch, label={"Original": "Original", "Pairs": "HIKARI-2021 (pairs)",
                                                   "Main": "HIKARI-2021 (main)"}[v])
    ax.set_yscale("log")
    ax.set_ylim(1e3, 1e6)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["Benign", "Background", "Probing", "Bruteforce", "Bruteforce-\nXML", "XMRIGCC\nCryptoMiner"],
                       fontsize=7.5)
    ax.set_ylabel("Number of flows (log scale)")
    ax.text(5 + w, 1.1e3, "removed", rotation=90, ha="center", va="bottom", fontsize=6.5, color=ps.INK)
    ps.style_axes(ax)
    ps.panel(ax, 0, "Flows per traffic category")
    ax.legend(loc="upper right", fontsize=7.5)
    ax = axes[1]
    vs = list(pr)
    for j, v in enumerate(vs):
        col, hatch = version_style[v]
        ax.bar(j, pr[v], color=col, edgecolor=ps.INK, linewidth=0.4, hatch=hatch, width=0.6)
        ax.text(j, pr[v] + 0.02, f"{pr[v]:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(["Original", "Pairs", "Main"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("PR-AUC of S0")
    ps.style_axes(ax)
    ps.panel(ax, 1, "S0 detection")
    fig.tight_layout()
    ps.save(fig, OUT, "fig_S2_hikari_audit")
    plt.close(fig)

    # ---------- Figura S3: ROC-AUC por semilla ----------
    rs = pd.read_csv(root / args.s3b_run / "S3B_report" / "roc_auc_by_seed_eps_le_10.csv")
    rs = rs[rs.evaluation == "clean_test"].melt(id_vars=["dataset", "variant", "evaluation", "epsilon"], var_name="seed",
                                                  value_name="roc_auc").dropna()
    rs["seed"] = rs.seed.astype(int)
    rs["extra_seed"] = False
    diag = pd.read_csv(root / args.batch2_run / "S3_inversion_diagnostic" / "per_seed.csv")
    diag = diag[diag.seed > 46].assign(dataset="HIKARI2021_v2_main", variant="S3a-min", evaluation="clean_test",
                                       roc_auc=lambda x: x.roc_auc_clean, extra_seed=True)
    allr = pd.concat([rs, diag[rs.columns]], ignore_index=True)
    allr.to_csv(OUT / "fig_S3_roc_auc_by_seed_data.csv", index=False)
    ds = ["CICIDS2017", "HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]
    fig, axes = ps.figure(180, 150, nrows=3, ncols=2, sharex=True, sharey=True)
    k = 0
    rng = np.random.default_rng(0)  # solo para el desplazamiento horizontal de los puntos (visual)
    for i, d in enumerate(ds):
        for j, v in enumerate(["S3a", "S3a-min"]):
            ax = axes[i, j]
            g = allr[(allr.dataset == d) & (allr.variant == v)]
            _, col, mk, _ = ps.SCENARIOS[v]
            for extra_seed, gg in g.groupby("extra_seed"):
                jitter = np.exp(rng.uniform(-0.12, 0.12, len(gg)))
                ax.plot(gg.epsilon * jitter, gg.roc_auc, ls="none", marker=mk, ms=4.5,
                        mfc="white" if extra_seed else col, mec=col if extra_seed else ps.INK, mew=0.8 if extra_seed else 0.4)
            means = g.groupby("epsilon").roc_auc.mean()
            ax.plot(means.index, means.values, color=col, lw=1.0, ls=ps.SCENARIOS[v][3])
            ax.axhline(0.5, color=ps.CHANCE[1], lw=1.0, ls=ps.CHANCE[2])
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ps.style_axes(ax)
            ps.panel(ax, k, f"{ps.DATASETS[d]}: {ps.SCENARIOS[v][0]}")
            k += 1
            if i == len(ds) - 1:
                ax.set_xlabel(ps.AXES["eps"])
            if j == 0:
                ax.set_ylabel("ROC-AUC (clean test)")
    handles = [Line2D([], [], ls="none", marker="o", color=ps.SCENARIOS["S3a"][1], mec=ps.INK, mew=0.4, label="S3a, one seed"),
               Line2D([], [], ls="none", marker="s", color=ps.SCENARIOS["S3a-min"][1], mec=ps.INK, mew=0.4, label="S3a-min, one seed"),
               Line2D([], [], ls="none", marker="s", mfc="white", mec=ps.SCENARIOS["S3a-min"][1], mew=0.8,
                      label="S3a-min, additional seeds 47–56 (diagnostic)"),
               Line2D([], [], color="#555555", lw=1.0, label="Mean across seeds"),
               Line2D([], [], color=ps.CHANCE[1], ls=ps.CHANCE[2], label="Chance level (0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ps.save(fig, OUT, "fig_S3_roc_auc_by_seed")
    plt.close(fig)
    print("Figuras en", OUT)


if __name__ == "__main__":
    main()
