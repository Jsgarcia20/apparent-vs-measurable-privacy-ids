"""Registra versiones de Python y librerías en results/environment.txt (y en la carpeta de la corrida)."""
import importlib
import os
import platform
import subprocess
import sys

from _common import REPO_ROOT, path_of, setup

PACKAGES = ["numpy", "pandas", "sklearn", "scipy", "pyarrow", "diffprivlib", "yaml"]


def main():
    args, cfg, out = setup("Registro del entorno")
    lines = [
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"platform: {platform.platform()}",
        f"cpu_count: {os.cpu_count()}",
    ]
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            label = "scikit-learn" if name == "sklearn" else ("pyyaml" if name == "yaml" else name)
            lines.append(f"{label}: {mod.__version__}")
        except ImportError:
            lines.append(f"{name}: NO INSTALADO")
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        commit = proc.stdout.strip() if proc.returncode == 0 else ""
    except OSError:
        commit = ""
    lines.append(f"git_commit: {commit or 'sin commit'}")
    text = "\n".join(lines) + "\n"
    (path_of(cfg, "results_dir") / "environment.txt").write_text(text, encoding="utf-8")
    (out / "environment.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
