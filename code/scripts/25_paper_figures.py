"""Figuras oficiales del artículo (en inglés), con el estilo único de src/plot_style.py.

Solo redibuja a partir de resultados ya guardados; no calcula modelos ni ataques.
Salida: results/paper_figures/ (PNG 300 dpi, SVG y PDF).
  fig_4_utility_vs_epsilon         PR-AUC frente a ε (S3a, S3a-min, S3b, S3c, S3c-min) por dataset.
  fig_5_pareto_privacy_utility     ventaja del atacante frente a PR-AUC; frente de Pareto.
  fig_6_aux_knowledge_sensitivity  balanced accuracy frente a % del train (A2, A3; S0 y S2).
  fig_S1_s3_utility_pr_auc_f1      (suplementaria) PR-AUC y F1 frente a ε para S3a, S3a-min y S3b.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from _common import REPO_ROOT, path_of, setup
from src import plot_style as ps
from src.data import load_raw, make_target
from src.splits import load_split

DS = ["CICIDS2017", "HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]
OUT = REPO_ROOT.parent / "results" / "paper_figures"


def extra(parser):
    parser.add_argument("--final-run", default="2026-10-07_final")
    parser.add_argument("--f1-run", default="2026-10-06_F1")
    parser.add_argument("--f2-run", default="2026-10-06_F2")
    parser.add_argument("--a3-run", default="2026-10-07_A3")
    parser.add_argument("--s3b-run", default="2026-10-07_S3B")
    parser.add_argument("--batch2-run", default="2026-10-07_batch2")


def ci_t(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def label_eps_once(fig, axes, pts):
    """Fig. 5: rotula cada ε una sola vez, junto al punto medio entre S3a y S3a-min de ese ε (comparten los ε).

    Prueba posiciones candidatas en orden y elige la primera cuya caja de texto no toque marcadores, el
    frente de Pareto, otras etiquetas, el borde del eje ni la zona del grupo S0/S1a/S1b/S2. Si la
    etiqueta queda lejos del punto, se dibuja una línea guía fina.
    """
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72                                     # píxeles por punto tipográfico
    near = [(6, 6), (-6, 6), (6, -6), (-6, -6), (8, 0), (-8, 0), (0, 8), (0, -8)]
    far = [(dx * k, dy * k) for k in (2.2, 3.2, 4.4) for dx, dy in near]
    for ax, d in zip(axes, DS):
        P = pts[pts.dataset == d]
        to_px = ax.transData.transform
        blocked = []
        markers = {}
        for r in P.itertuples():                          # marcadores (radio + 1.5 pt de margen)
            cx, cy = to_px((r.x, r.y))
            h = ((8 if r.family == "S0" else 5.5) / 2 + 1.5) * px
            markers[r.Index] = Bbox([[cx - h, cy - h], [cx + h, cy + h]])
        blocked = list(markers.values())
        grp = np.array([to_px((r.x, r.y)) for r in P.itertuples() if r.family in ("S0", "S1a", "S1b", "S2")])
        g = 10 * px                                       # zona reservada alrededor del grupo S0–S2
        blocked.append(Bbox([grp.min(0) - g, grp.max(0) + g]))
        fr = P[P.pareto].sort_values("x")
        v = to_px(fr[["x", "y"]].to_numpy())             # frente muestreado densamente (polilínea abierta)
        tt = np.linspace(0, 1, 400)[:, None]
        front = np.vstack([a + tt * (b - a) for a, b in zip(v[:-1], v[1:])])
        inner = ax.get_window_extent(rend)
        for r in P[P.family == "S3a"].sort_values("eps").itertuples():
            m = P[(P.family == "S3a-min") & (P.eps == r.eps)].iloc[0]
            own_ids = [r.Index, m.name]
            mid = ((r.x + m.x) / 2, (r.y + m.y) / 2)
            for strict, (dx, dy) in [(st_, o) for st_ in (True, False) for o in near + far]:
                ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
                guide = abs(dx) + abs(dy) > 14
                t = ax.annotate(f"ε={r.eps:g}", mid, textcoords="offset points", xytext=(dx, dy),
                                ha=ha, va=va, fontsize=6.5, color=ps.INK, zorder=2,   # guía bajo los marcadores
                                arrowprops=dict(arrowstyle="-", lw=0.5, color="#7a7a7a", shrinkA=0,
                                                shrinkB=4) if guide else None)
                t.update_positions(rend)
                bb = Text.get_window_extent(t, rend)          # solo el texto, sin la línea guía
                bb = Bbox([[bb.x0 - 1.5 * px, bb.y0 - 1.5 * px], [bb.x1 + 1.5 * px, bb.y1 + 1.5 * px]])
                ok = (inner.x0 <= bb.x0 and bb.x1 <= inner.x1 and inner.y0 <= bb.y0 and bb.y1 <= inner.y1
                      and not any(bb.overlaps(b) for b in blocked) and not bb.count_contains(front))
                if ok and guide:                          # la línea guía tampoco toca otros puntos ni el frente
                    a = np.array(to_px(mid))
                    b = np.array([min(max(a[0], bb.x0), bb.x1), min(max(a[1], bb.y0), bb.y1)])
                    seg = a + np.linspace(0, 1, 200)[:, None] * (b - a)
                    # se ignora el tramo que queda bajo el círculo y el cuadrado propios (y a < 4 pt del punto medio,
                    # que puede caer sobre el frente cuando ambos puntos son de Pareto)
                    inside = np.zeros(len(seg), bool)
                    for k in own_ids if not strict else []:   # pasada estricta: tampoco bajo los propios
                        o = markers[k]
                        inside |= (seg[:, 0] >= o.x0) & (seg[:, 0] <= o.x1) & (seg[:, 1] >= o.y0) & (seg[:, 1] <= o.y1)
                    inside |= np.hypot(*(seg - a).T) < 4 * px
                    seg = seg[~inside]
                    ok = len(seg) > 0 and (
                        not any(q.count_contains(seg) for k, q in markers.items() if not (k in own_ids and
                                                                                         not strict))
                        and not any(q.count_contains(seg) for q in blocked[len(markers):])
                        and np.hypot(*(front[:, None, :] - seg[None, :, :]).transpose(2, 0, 1)).min() > 1.5 * px)
                if ok:
                    blocked.append(bb)
                    break
                t.remove()
            else:
                raise RuntimeError(f"Sin hueco para la etiqueta ε={r.eps:g} en {d}")


def main():
    args, cfg, _ = setup("Figuras del artículo", extra)
    root = path_of(cfg, "results_dir")
    OUT.mkdir(exist_ok=True)
    plt = ps.apply()
    from matplotlib.lines import Line2D

    f1run = pd.read_csv(root / args.f1_run / "utility_report" / "utility_full_test.csv")
    s3dir = root / args.s3b_run / "S3B"
    lrref = pd.read_csv(root / args.batch2_run / "S3c" / "lr_ref_converged.csv")
    lrref = lrref[lrref.converged_without_warning]
    prevalence = {}
    for d in DS:
        y = make_target(load_raw(cfg, d), d, cfg).to_numpy()
        _, te = load_split(cfg, d)
        prevalence[d] = float(y[te].mean())

    def refs(d, metric="pr_auc"):
        if d == "CICIDS2017":
            u = pd.read_csv(s3dir / f"{d}_utility_runs.csv")
            s0, s2 = u[u.variant == "S0-sub"][metric].iloc[0], u[u.variant == "S2-sub"][metric].iloc[0]
        else:
            f = f1run[f1run.dataset == d].set_index("scenario")
            s0, s2 = f.loc["S0", metric], f.loc["S2", metric]
        lr = lrref[(lrref.dataset == d) & (lrref.variant == "LR-ref (S3c)")][metric].iloc[-1]
        return s0, s2, lr

    def ref_lines(ax, d, metric, with_lr=True, with_chance=True):
        s0, s2, lr = refs(d, metric)
        ax.axhline(s0, color=ps.SCENARIOS["S0"][1], lw=1.0, ls="-", label=ps.REFERENCE_S0_LABEL)
        ax.axhline(s2, color=ps.SCENARIOS["S2"][1], lw=1.0, ls="--", label=ps.SCENARIOS["S2"][0])
        if with_lr:
            _, lr_col, _, lr_ls = ps.SCENARIOS["LR-ref"]
            ax.axhline(lr, color=lr_col, lw=1.0, ls=lr_ls, label=ps.SCENARIOS["LR-ref"][0])
            # etiqueta directa al final de la línea (extremo derecho del eje)
            ax.annotate("LR-ref", (1.0, lr), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(2, 0), ha="left", va="center", fontsize=6.5, color=lr_col, annotation_clip=False)
        if with_chance and metric == "pr_auc":
            ax.axhline(prevalence[d], color=ps.CHANCE[1], lw=1.0, ls=ps.CHANCE[2], label=ps.CHANCE[0])

    # ---------------- Figure A ----------------
    dp = pd.read_csv(root / args.final_run / "phase10" / "phase10_dp_vs_reference.csv")
    fig, axes = ps.figure(180, 72, ncols=3, sharey=True)
    for i, (ax, d) in enumerate(zip(axes, DS)):
        g = dp[dp.dataset == d]
        for v in ["S3a", "S3a-min", "S3b", "S3c", "S3c-min"]:
            x = g[g.variant == v].sort_values("epsilon")
            ax.fill_between(x.epsilon, x.pr_auc_ci95_low.clip(0, 1), x.pr_auc_ci95_high.clip(0, 1),
                            color=ps.SCENARIOS[v][1], alpha=0.08 if v == "S3b" else 0.12, lw=0)
            ax.plot(x.epsilon, x.pr_auc_mean, label=ps.SCENARIOS[v][0], **ps.line_kw(v))
        ref_lines(ax, d, "pr_auc")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(ps.AXES["eps"])
        ps.style_axes(ax)
        ps.panel(ax, i, ps.DATASETS[d])
    axes[0].set_ylabel(ps.AXES["pr_auc"])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2, handlelength=2.6)
    fig.tight_layout()
    ps.save(fig, OUT, "fig_4_utility_vs_epsilon")
    plt.close(fig)

    # ---------------- Figure B ----------------
    pts = pd.read_csv(root / args.final_run / "figures" / "figB_points.csv")
    fig, axes = ps.figure(180, 78, ncols=3, sharey=True)
    for i, (ax, d) in enumerate(zip(axes, DS)):
        P = pts[pts.dataset == d]
        fr = P[P.pareto].sort_values("x")
        ax.plot(fr.x, fr.y, color="#7a7a7a", lw=1.0, ls="-", zorder=1)
        for r in P.itertuples():
            _, col, mk, _ = ps.SCENARIOS[r.family]
            ms = 8 if r.family == "S0" else 5.5
            ax.plot(r.x, r.y, ls="none", marker=mk, color=col, ms=ms, mec=ps.INK,
                    mew=1.1 if r.pareto else 0.4, zorder=3)
        attack = "A1" if d == "CICIDS2017" else "A2"
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(ps.AXES["advantage"])
        ps.style_axes(ax)
        ps.panel(ax, i, f"{ps.DATASETS[d]} · {attack} ({'port' if attack == 'A1' else 'host'})")
    axes[0].set_ylabel(ps.AXES["pr_auc"])
    handles = [Line2D([], [], ls="none", marker=ps.SCENARIOS[k][2], color=ps.SCENARIOS[k][1], mec=ps.INK, mew=0.4,
                      ms=7 if k == "S0" else 5.5, label=ps.SCENARIOS[k][0])
               for k in ["S0", "S1a", "S1b", "S2", "S3a", "S3a-min"]]
    handles += [Line2D([], [], color="#7a7a7a", lw=1.0, label="Pareto front"),
                Line2D([], [], ls="none", marker="o", color="white", mec=ps.INK, mew=1.1, ms=5.5,
                       label="Point on the Pareto front")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2)
    fig.tight_layout()
    label_eps_once(fig, axes, pts)
    ps.save(fig, OUT, "fig_5_pareto_privacy_utility")
    plt.close(fig)

    # ---------------- Figure C ----------------
    aux = pd.read_csv(root / args.batch2_run / "aux_knowledge" / "aux_knowledge_runs.csv")
    a3 = pd.read_csv(root / args.a3_run / "A3_same_role_host" / "a3_runs.csv")
    full_rows = []
    for d in ["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]:
        f2 = pd.read_csv(root / args.f2_run / "F2_attribute_inference" / f"{d}_attribute_inference_runs.csv")
        f2 = f2[(f2.objective == "A2_source_host") & (f2.attack == "inference")]
        full_rows += [{"dataset": d, "objective": "A2_source_host", "scenario": r.scenario, "fraction": 1.0,
                       "balanced_accuracy": r.balanced_accuracy} for r in f2.itertuples()]
    full_rows += [{"dataset": r.dataset, "objective": r.objective, "scenario": r.scenario, "fraction": 1.0,
                   "balanced_accuracy": r.balanced_accuracy} for r in a3.itertuples()]
    allruns = pd.concat([aux[["dataset", "objective", "scenario", "fraction", "balanced_accuracy"]], pd.DataFrame(full_rows)])
    objs = [("A2_source_host", ps.ATTACKS["A2"]), ("A3_DNS", "A3 (same-role hosts), port 53"),
            ("A3_443", "A3 (same-role hosts), port 443")]   # dataset = etiqueta de fila en el eje Y
    fig, axes = ps.figure(180, 112, nrows=2, ncols=3, sharex=True, sharey=True)
    summary = []
    k = 0
    for i, d in enumerate(["HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]):
        for j, (o, olab) in enumerate(objs):
            ax = axes[i, j]
            for sc in ["S0", "S2"]:
                g = allruns[(allruns.dataset == d) & (allruns.objective == o) & (allruns.scenario == sc)]
                rows = []
                for frac, gg in g.groupby("fraction"):
                    m, lo, hi = ci_t(gg.balanced_accuracy)
                    rows.append((frac * 100, m, lo, hi, len(gg)))
                    summary.append({"dataset": d, "objective": o, "scenario": sc, "fraction_pct": frac * 100,
                                    "n_runs": len(gg), "mean": m, "ci95_low": lo, "ci95_high": hi})
                R = np.array(rows)
                ax.errorbar(R[:, 0], R[:, 1], yerr=[R[:, 1] - R[:, 2], R[:, 3] - R[:, 1]], capsize=2, elinewidth=0.8,
                            label=ps.SCENARIOS[sc][0], **ps.line_kw(sc, ms=4.5 if sc == "S2" else 6))
            ax.axhline(0.5, color=ps.CHANCE[1], lw=1.0, ls=ps.CHANCE[2], label=ps.CHANCE[0])
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 20, 100])
            ax.set_xticklabels(["1", "5", "20", "100"])
            ax.set_ylim(0.45, 1.0)
            ps.style_axes(ax)
            ps.panel(ax, k, olab)
            k += 1
            if j == 0:
                ax.set_ylabel(ps.DATASETS[d] + "\n" + ps.AXES["bacc"])
    fig.supxlabel(ps.AXES["aux"], fontsize=9, y=0.02)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ps.save(fig, OUT, "fig_6_aux_knowledge_sensitivity")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(OUT / "fig_6_aux_knowledge_sensitivity_data.csv", index=False)

    # ---------------- Figure S1 (supplementary) ----------------
    util = pd.read_csv(root / args.s3b_run / "S3B_report" / "utility_summary.csv")
    util = util[util.evaluation == "clean_test"]
    fig, axes = ps.figure(180, 190, nrows=3, ncols=2, sharex=True)
    k = 0
    for i, d in enumerate(DS):
        for j, (mtr, ylab) in enumerate([("pr_auc", ps.AXES["pr_auc"]), ("f1", ps.AXES["f1"])]):
            ax = axes[i, j]
            g = util[util.dataset == d]
            for v in ["S3a", "S3a-min", "S3b"]:
                x = g[g.variant == v].sort_values("epsilon")
                ax.fill_between(x.epsilon, x[f"{mtr}_ci95_low"].clip(0, 1), x[f"{mtr}_ci95_high"].clip(0, 1),
                                color=ps.SCENARIOS[v][1], alpha=0.15, lw=0)
                ax.plot(x.epsilon, x[f"{mtr}_mean"], label=ps.SCENARIOS[v][0], **ps.line_kw(v))
            ref_lines(ax, d, mtr, with_lr=False, with_chance=True)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.03)
            ax.set_ylabel(ylab)
            ps.style_axes(ax)
            ps.panel(ax, k, ps.DATASETS[d])
            k += 1
            if i == len(DS) - 1:
                ax.set_xlabel(ps.AXES["eps"])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    ps.save(fig, OUT, "fig_S1_s3_utility_pr_auc_f1")
    plt.close(fig)
    print("Figuras en", OUT)


if __name__ == "__main__":
    main()
