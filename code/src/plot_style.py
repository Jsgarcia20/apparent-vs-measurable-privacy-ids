"""Estilo único de las figuras del artículo (texto en inglés).

- Fuente Arial (alternativas: Helvetica, DejaVu Sans), 8–9 pt a tamaño final; punto decimal.
- Anchos: una columna 85 mm, doble columna 180 mm. Salida PNG 300 dpi + SVG + PDF (fuentes embebidas).
- Paleta: Okabe-Ito sin el amarillo (no pasa la banda de luminosidad sobre fondo blanco) más el vino de
  la paleta "muted" de Paul Tol para S3c-min. Validada con validate_palette.js (--pairs all): pasa la
  banda de luminosidad, el croma y el umbral de visión normal. Los avisos (CVD 7.6 en el par S2/S3c y el
  contraste < 3:1 de tres tonos) se compensan con un marcador y un estilo de línea propios por escenario
  y con bordes oscuros en los marcadores. S0, S1b y LR-ref usan tonos neutros (negro y grises oscuros).
- Un color, marcador y estilo de línea FIJOS por escenario en todas las figuras.
- Sin títulos; paneles etiquetados (a), (b), (c)…
"""
import string

MM = 1 / 25.4
SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 180

SCENARIOS = {  # clave: (etiqueta en inglés, color, marcador, estilo de línea)
    "S0": ("S0 (baseline)", "#000000", "*", "-"),
    "S1a": ("S1a (port suppression)", "#E69F00", "P", "-"),
    "S1b": ("S1b (port generalization)", "#5A5A5A", "X", "-"),
    "S2": ("S2 (minimization)", "#009E73", "d", "--"),
    "S3a": ("S3a (local DP)", "#0072B2", "o", "-"),
    "S3a-min": ("S3a-min (local DP + minimization)", "#D55E00", "s", "--"),
    "S3b": ("S3b (central DP, random forest)", "#56B4E9", "^", ":"),
    "S3c": ("S3c (central DP, logistic regression)", "#CC79A7", "D", "-."),
    "S3c-min": ("S3c-min", "#882255", "v", (0, (5, 1, 1, 1, 1, 1))),
    "LR-ref": ("LR-ref", "#4D4D4D", None, (0, (7, 3))),   # trazo largo gris oscuro (≠ punteado del azar)
}
REFERENCE_S0_LABEL = "S0 (no protection)"
CHANCE = ("Chance level", "#B0B0B0", (0, (1, 2)))
DATASETS = {"CICIDS2017": "CICIDS2017", "HIKARI2021_v2_main": "HIKARI-2021 (main)",
            "HIKARI2021_v2_pairs": "HIKARI-2021 (pairs)"}
ATTACKS = {"A1": "A1 (destination port)", "A2": "A2 (source host)", "A3": "A3 (same-role hosts)"}
AXES = {"eps": "Privacy budget ε (log scale)", "pr_auc": "PR-AUC", "f1": "F1 score",
        "advantage": "Attacker advantage", "bacc": "Balanced accuracy",
        "aux": "Auxiliary knowledge (% of training set)"}
INK, GRID = "#1a1a1a", "#e3e3e3"


def apply():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    family = next(f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in available)
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.titlesize": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.formatter.use_locale": False, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "legend.frameon": False,
    })
    return plt


def figure(width_mm=DOUBLE_COLUMN_MM, height_mm=70, **kw):
    plt = apply()
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), **kw)


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def panel(ax, index, text=""):
    """Etiqueta de panel '(a) texto' encima del eje, alineada a la izquierda (no es un título)."""
    label = f"({string.ascii_lowercase[index]})" + (f" {text}" if text else "")
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK)


def line_kw(key, **override):
    _, color, marker, ls = SCENARIOS[key]
    kw = dict(color=color, ls=ls, lw=1.4, marker=marker, ms=4, mec=INK, mew=0.4)
    if marker is None:
        kw.pop("marker"), kw.pop("ms"), kw.pop("mec"), kw.pop("mew")
    kw.update(override)
    return kw


def save(fig, out_dir, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
