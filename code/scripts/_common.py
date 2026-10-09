"""Utilidades compartidas por los scripts (rutas del repo y carpeta de la corrida)."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, path_of  # noqa: E402


def parse_args(description, extra=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", required=True, help="Subcarpeta dentro de results/, p. ej. 2026-10-05_partB")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    if extra:
        extra(parser)
    return parser.parse_args()


DEFAULT_DATASETS = ["CICIDS2017", "HIKARI2021"]


def add_datasets_arg(parser, default=DEFAULT_DATASETS):
    parser.add_argument("--datasets", nargs="+", default=list(default))


def prefix(cfg, dataset):
    return cfg["datasets"][dataset]["prefix"]


def run_dir(cfg, run_id):
    d = path_of(cfg, "results_dir") / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup(description, extra=None):
    args = parse_args(description, extra)
    cfg = load_config(args.config)
    out = run_dir(cfg, args.run_id)
    # Copia de la configuración usada en la corrida
    (out / "config_used.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    return args, cfg, out
