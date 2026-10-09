"""Carga de config.yaml."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg, key):
    """Las rutas relativas de config.yaml se resuelven desde la carpeta code/."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
