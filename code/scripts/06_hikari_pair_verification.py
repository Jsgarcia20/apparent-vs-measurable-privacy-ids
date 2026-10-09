"""D1 — Verificaciones de los pares conflictivos de HIKARI (SOLO LECTURA; no reetiqueta ni limpia).

a) Columnas temporales: busca columnas de tiempo en el parquet y en el CSV crudo. Compara entre las
   dos copias de cada par las columnas de índice de exportación del CSV crudo ("" y "Unnamed: 0"),
   que la limpieza eliminó.
b) Ruido residual: filas Label 0 NO emparejadas (fuera de patrones conflictivos) que involucran IPs
   de filas de ataque, por categoría de ataque, rol de la IP y coincidencia de puerto de destino típico.
c) Consultas de la Parte C convertidas en script: columnas no predictoras que difieren dentro de cada
   par conflictivo y simulación (en memoria) de la regla alternativa "quitar solo la copia benigna".

Definiciones:
- Par / patrón conflictivo: mismo valor en los 81 predictores de S0 y Label distinto (sin uid).
- Puertos típicos de una categoría de ataque: valores de responp presentes en al menos
  TYPICAL_PORT_MIN_SHARE de las filas de ataque de esa categoría (criterio basado en los datos).
"""
import json
import re

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from _common import path_of, setup
from src.data import load_raw, make_predictors, make_target

TYPICAL_PORT_MIN_SHARE = 0.01
TIME_PATTERN = re.compile(r"(^ts$|time|timestamp|date|start|end|epoch)", re.IGNORECASE)
RAW_CSV_NAME = "ALLFLOWMETER_HIKARI2021.csv"


def conflict_info(cfg):
    df = load_raw(cfg, "HIKARI2021")
    X = make_predictors(df, cfg, "HIKARI2021")
    y = make_target(df, "HIKARI2021", cfg).to_numpy()
    k = hash_pandas_object(X, index=False).to_numpy()
    conf = pd.Series(y).groupby(k).transform("nunique").to_numpy() > 1
    return df, X, y, k, conf


def part_a(cfg, df, k, conf, out):
    res = {"time_like_columns_parquet": [c for c in df.columns if TIME_PATTERN.search(c)]}
    raw_path = path_of(cfg, "data_root") / RAW_CSV_NAME
    header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    res["time_like_columns_raw_csv"] = [c for c in header if TIME_PATTERN.search(str(c))]
    res["raw_csv_first_columns"] = header[:3]
    raw_idx = pd.read_csv(raw_path, usecols=[header[0], header[1], "uid"])[[header[0], header[1], "uid"]]
    raw_idx.columns = ["export_index_0", "export_index_1", "uid"]
    res["raw_row_order_matches_parquet"] = bool((raw_idx["uid"].to_numpy() == df["uid"].to_numpy()).all())
    c = raw_idx.loc[conf].assign(k=k[conf], cat=df.loc[conf, "traffic_category"].to_numpy(),
                                 y=df.loc[conf, "Label"].to_numpy())
    res["conflicting_patterns"] = int(c["k"].nunique())
    for col in ["export_index_0", "export_index_1"]:
        res[f"patterns_where_{col}_equal"] = int((c.groupby("k")[col].nunique() == 1).sum())
    # Diferencia del índice de exportación entre la copia de ataque y la benigna, por categoría de ataque
    att = c[c.y == 1].groupby("k").agg(cat=("cat", "first"), idx1_attack=("export_index_1", "min"),
                                         idx0_attack=("export_index_0", "min"))
    ben = c[c.y == 0].groupby("k").agg(idx1_benign=("export_index_1", "min"),
                                         idx0_benign=("export_index_0", "min"))
    pairs = att.join(ben)
    pairs["idx1_equal"] = pairs.idx1_attack == pairs.idx1_benign
    summary = pairs.groupby("cat").agg(patterns=("idx1_equal", "size"),
                                       export_index_1_equal=("idx1_equal", "sum")).reset_index()
    res["export_index_1_equal_by_attack_category"] = summary.to_dict("records")
    (out / "D1a_time_columns.json").write_text(json.dumps(res, indent=2, default=int), encoding="utf-8")
    print("== D1a ==\n" + json.dumps(res, indent=2, default=int))


def part_b(df, y, conf, out):
    attacks = df[y == 1]
    categories = sorted(attacks["traffic_category"].unique())
    unpaired_benign = df[(y == 0) & ~conf]
    rows, ports_used = [], {}
    for cat in categories:
        a = attacks[attacks.traffic_category == cat]
        attackers, victims = set(a.originh), set(a.responh)
        share = a["responp"].value_counts(normalize=True)
        typical = set(share[share >= TYPICAL_PORT_MIN_SHARE].index.astype(int))
        b = unpaired_benign
        same_flow_shape = b[b.originh.isin(attackers) & b.responh.isin(victims) & b.responp.isin(typical)]
        ports_used[cat] = {"typical_ports": sorted(int(p) for p in typical),
                           "share_of_attack_rows_covered": float(share[share >= TYPICAL_PORT_MIN_SHARE].sum()),
                           "attacker_ips": sorted(attackers), "victim_ips": sorted(victims),
                           "label0_rows_from_attacker_ips_total": int(df[(y == 0) & df.originh.isin(attackers)].shape[0]),
                           "unpaired_label0_attacker_to_victim_typical_port": int(len(same_flow_shape)),
                           "medians_attack_rows": a[["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot",
                                                     "flow_pkts_payload.tot"]].median().to_dict(),
                           "medians_unpaired_label0_attacker_to_victim_typical_port":
                               same_flow_shape[["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot",
                                                "flow_pkts_payload.tot"]].median().to_dict()}
        o_att = b.originh.isin(attackers)
        r_vic = b.responh.isin(victims)
        role = np.select([o_att & r_vic, o_att & ~r_vic, ~o_att & r_vic, b.responh.isin(attackers)],
                         ["attacker_to_victim", "attacker_to_other", "other_to_victim", "other_to_attacker"],
                         default="none")
        port_match = b.responp.isin(typical).to_numpy()
        t = pd.DataFrame({"benign_category": b.traffic_category.to_numpy(), "role": role, "port_typical": port_match})
        t = t[t.role != "none"]
        g = t.groupby(["benign_category", "role", "port_typical"]).size().rename("rows").reset_index()
        g.insert(0, "attack_category", cat)
        rows.append(g)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "D1b_residual_benign_rows_with_attack_ips.csv", index=False)
    (out / "D1b_typical_ports.json").write_text(json.dumps(ports_used, indent=2), encoding="utf-8")
    print("== D1b ==\n" + json.dumps(ports_used, indent=2))
    print(res.to_string(index=False))
    print(f"Filas Label 0 no emparejadas: {len(unpaired_benign):,}")


def part_c(df, X, y, k, conf, out):
    c = df.loc[conf].assign(k=k[conf])
    non_predictors = [col for col in df.columns if col not in X.columns]
    diff = {col: int((c.groupby("k")[col].nunique(dropna=False) > 1).sum()) for col in non_predictors}
    sizes = c.groupby("k").size().value_counts().sort_index()
    verifiable = (c.groupby("k")[["originh", "responh", "originp", "responp"]].nunique() == 1).all(axis=1)
    res = {"conflicting_patterns": int(c["k"].nunique()),
           "rows_in_conflicting_patterns": int(conf.sum()),
           "patterns_where_non_predictor_column_differs": diff,
           "pattern_size_distribution": {int(s): int(n) for s, n in sizes.items()},
           "patterns_sharing_originh_responh_originp_responp": int(verifiable.sum()),
           "patterns_not_sharing_them": int((~verifiable).sum())}
    # Regla alternativa (en memoria): quitar solo las copias benignas de patrones conflictivos y deduplicar
    keep = ~(conf & (y == 0))
    k_full = hash_pandas_object(pd.concat([X, pd.Series(y, name="Label", index=X.index)], axis=1), index=False).to_numpy()
    keep_idx = np.flatnonzero(keep)
    dup = pd.Series(k_full[keep_idx]).duplicated(keep="first").to_numpy()
    final = np.zeros(len(y), bool)
    final[keep_idx[~dup]] = True
    res["alternative_rule"] = {"rows": int(final.sum()), "attacks": int(y[final].sum()),
                               "attack_rate": float(y[final].mean()),
                               "by_category": df.loc[final].groupby("traffic_category").size().to_dict()}
    (out / "D1c_pair_differences_and_alternative_rule.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("== D1c ==\n" + json.dumps(res, indent=2))
    c.loc[c["k"].isin(verifiable[~verifiable].index),
          ["uid", "originh", "originp", "responh", "responp", "traffic_category", "Label"]].to_csv(
        out / "D1c_non_verifiable_conflict_rows.csv", index=False)


def main():
    args, cfg, out = setup("D1: verificación de pares conflictivos de HIKARI (solo lectura)")
    out = out / "D1_hikari_pairs"
    out.mkdir(exist_ok=True)
    df, X, y, k, conf = conflict_info(cfg)
    part_a(cfg, df, k, conf, out)
    part_b(df, y, conf, out)
    part_c(df, X, y, k, conf, out)


if __name__ == "__main__":
    main()
