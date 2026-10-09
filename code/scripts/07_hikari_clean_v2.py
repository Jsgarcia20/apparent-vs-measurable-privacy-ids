"""D2 — Limpieza v2 de HIKARI (dos versiones) y splits nuevos SOLO para v2.

No modifica HIKARI2021_CLEAN.parquet ni los splits originales.

Reglas:
  1. Duplicados y conflictos sobre los 81 predictores de S0 (+ Label); nunca con uid.
  2. Patrón conflictivo verificable (todas sus filas comparten originh, responh, originp, responp):
     eliminar las copias Label 0 y conservar las de ataque.
  3. Patrón conflictivo no verificable: eliminar todas sus filas.
  3b. (solo v2_main) Eliminar las filas de las categorías excluidas (XMRIGCC CryptoMiner).
  3c. (solo v2_main) Eliminar las filas Label 0 no emparejadas (fuera de patrones conflictivos)
      con originh == atacante de c, responh en víctimas de c y responp en puertos de c,
      para c en {Probing, Bruteforce, Bruteforce-XML}. IPs y puertos de D1b; sin criterios de forma.
  4. Eliminar duplicados exactos (81 predictores + Label) dejando la primera aparición.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _common import path_of, setup
from src.data import load_raw, make_predictors, make_target
from src.integrity import conflict_mask, row_hash, test_has_train_copy
from src.splits import split_paths, verify_split

SOURCE = "HIKARI2021"
VERSIONS = {"HIKARI2021_v2_pairs": False, "HIKARI2021_v2_main": True}   # True = aplica 3b y 3c
D1B_JSON = "2026-10-06_D1/D1_hikari_pairs/D1b_typical_ports.json"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_reasons(df, X, y, cfg, main_version):
    clean_cfg = cfg["hikari_v2_cleaning"]
    k_feat = row_hash(X)
    conf = conflict_mask(k_feat, y)
    identity = clean_cfg["pair_identity_columns"]
    verifiable_by_key = (df.loc[conf, identity].assign(k=k_feat[conf]).groupby("k")[identity].nunique() == 1).all(axis=1)
    verifiable = np.zeros(len(df), bool)
    verifiable[conf] = verifiable_by_key.reindex(k_feat[conf]).to_numpy()

    reason = np.full(len(df), "", dtype=object)
    reason[conf & verifiable & (y == 0)] = "rule2_benign_copy_of_verifiable_conflict"
    reason[conf & ~verifiable] = "rule3_non_verifiable_conflict"

    if main_version:
        excluded = df["traffic_category"].isin(clean_cfg["excluded_categories_main"]).to_numpy()
        reason[(reason == "") & excluded] = "rule3b_excluded_category"
        residual_cats = np.full(len(df), "", dtype=object)
        for cat, rule in clean_cfg["residual_rules_main"].items():
            m = ((y == 0) & ~conf & (df["originh"] == rule["attacker"]).to_numpy()
                 & df["responh"].isin(rule["victims"]).to_numpy() & df["responp"].isin(rule["ports"]).to_numpy())
            residual_cats[m] = np.where(residual_cats[m] == "", cat, residual_cats[m] + "+" + cat)
        m3c = (reason == "") & (residual_cats != "")
        reason[m3c] = "rule3c_residual_" + residual_cats[m3c]

    k_full = row_hash(pd.concat([X, pd.Series(y, name="Label", index=X.index)], axis=1))
    alive = np.flatnonzero(reason == "")
    dup = pd.Series(k_full[alive]).duplicated(keep="first").to_numpy()
    reason[alive[dup]] = "rule4_exact_duplicate"
    return reason, conf, verifiable, k_feat


def category_table(df, y, reason):
    keep = reason == ""
    base = df.assign(y=y, keep=keep, reason=reason)
    t = base.groupby(["traffic_category", "Label"]).size().rename("before").to_frame()
    for r in sorted(set(reason) - {""}):
        t[r] = base[base.reason == r].groupby(["traffic_category", "Label"]).size()
    t["after"] = base[base.keep].groupby(["traffic_category", "Label"]).size()
    t = t.fillna(0).astype(int).reset_index()
    t["removed_pct"] = (t["before"] - t["after"]) / t["before"] * 100
    total = t.drop(columns=["traffic_category", "Label", "removed_pct"]).sum()
    total_row = {"traffic_category": "TOTAL", "Label": -1, **total.to_dict()}
    total_row["removed_pct"] = (total["before"] - total["after"]) / total["before"] * 100
    t = pd.concat([t, pd.DataFrame([total_row])], ignore_index=True)
    return t


def main():
    args, cfg, out = setup("D2: limpieza v2 de HIKARI")
    out = out / "D2_hikari_clean_v2"
    out.mkdir(exist_ok=True)
    df = load_raw(cfg, SOURCE)
    X = make_predictors(df, cfg, SOURCE)
    y = make_target(df, SOURCE, cfg).to_numpy()
    original_path = path_of(cfg, cfg["datasets"][SOURCE]["parquet_key"])
    d1b = json.loads((path_of(cfg, "results_dir") / D1B_JSON).read_text(encoding="utf-8"))
    summary = {"source": str(original_path), "source_md5": md5(original_path), "source_rows": int(len(df)),
               "source_attack_rate": float(y.mean())}

    for version, main_version in VERSIONS.items():
        reason, conf, verifiable, _ = assign_reasons(df, X, y, cfg, main_version)
        counts = pd.Series(reason[reason != ""]).value_counts().to_dict()
        n_patterns_rule2 = int(pd.Series(row_hash(X)[conf & verifiable]).nunique())
        n_patterns_rule3 = int(pd.Series(row_hash(X)[conf & ~verifiable]).nunique())
        assert n_patterns_rule2 == 34245 and n_patterns_rule3 == 1, (n_patterns_rule2, n_patterns_rule3)
        if main_version:
            # 3c debe coincidir exactamente con lo cuantificado en D1b (antes de 3b y 4)
            expected_3c = d1b["Probing"]["unpaired_label0_attacker_to_victim_typical_port"] + \
                d1b["Bruteforce"]["unpaired_label0_attacker_to_victim_typical_port"]
            got_3c = int(sum(v for r, v in counts.items() if r.startswith("rule3c")))
            assert got_3c == expected_3c, f"3c: {got_3c} filas, D1b esperaba {expected_3c}"

        keep = reason == ""
        clean = df.loc[keep].reset_index(drop=True)
        dest = path_of(cfg, cfg["datasets"][version]["parquet_key"])
        assert dest.resolve() != original_path.resolve(), "Nunca sobrescribir el parquet original"
        clean.to_parquet(dest, index=False)

        removed = df.loc[~keep, ["uid", "originh", "originp", "responh", "responp", "traffic_category", "Label"]].copy()
        removed.insert(0, "original_row", np.flatnonzero(~keep))
        removed["reason"] = reason[~keep]
        removed_path = dest.with_name(dest.stem + "_removed_rows.csv")
        removed.to_csv(removed_path, index=False)
        removed.to_csv(out / f"{version}_removed_rows.csv", index=False)
        table = category_table(df, y, reason)
        table.to_csv(out / f"{version}_before_after_by_category.csv", index=False)

        # Splits nuevos (excepción autorizada a la regla 3; solo para v2)
        Xc = make_predictors(clean, cfg, version)
        yc = make_target(clean, version, cfg).to_numpy()
        tr, te = train_test_split(np.arange(len(yc)), test_size=cfg["split"]["test_size"],
                                  random_state=cfg["seeds"]["split"], stratify=yc)
        train_path, test_path = split_paths(cfg, version)
        if train_path.exists() or test_path.exists():
            same = np.array_equal(np.load(train_path), tr) and np.array_equal(np.load(test_path), te)
            assert same, f"Ya existen splits distintos para {version}: no se sobrescriben"
        else:
            np.save(train_path, tr)
            np.save(test_path, te)

        # Verificaciones
        kc = row_hash(Xc)
        k_full_c = row_hash(pd.concat([Xc, pd.Series(yc, name="Label", index=Xc.index)], axis=1))
        split_rep = verify_split(tr, te, yc, len(tr), len(te))
        checks = {
            "rows": int(len(clean)), "attacks": int(yc.sum()), "attack_rate": float(yc.mean()),
            "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "remaining_exact_duplicates_predictors_plus_label": int(pd.Series(k_full_c).duplicated().sum()),
            "remaining_duplicate_predictor_vectors": int(pd.Series(kc).duplicated().sum()),
            "remaining_conflicting_rows": int(conflict_mask(kc, yc).sum()),
            "s0_test_rows_with_exact_copy_in_train": int(test_has_train_copy(kc, tr, te).sum()),
            "split_overlap": split_rep["overlap"], "split_stratification_ok": split_rep["stratification_ok"],
            "train_attack_rate": split_rep["train_attack_rate"], "test_attack_rate": split_rep["test_attack_rate"],
        }
        assert checks["s0_test_rows_with_exact_copy_in_train"] == 0
        assert checks["remaining_conflicting_rows"] == 0 and checks["remaining_exact_duplicates_predictors_plus_label"] == 0
        summary[version] = {"removed_by_reason": counts, "patterns_rule2": n_patterns_rule2,
                            "patterns_rule3": n_patterns_rule3, **checks,
                            "parquet": str(dest), "parquet_md5": md5(dest), "removed_rows_csv": str(removed_path),
                            "train_split": str(train_path), "test_split": str(test_path)}
        print(f"== {version} ==\n{json.dumps(summary[version], indent=2)}")
        print(table.to_string(index=False))

    (out / "D2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
