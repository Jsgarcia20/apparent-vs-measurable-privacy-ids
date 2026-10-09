"""S3-A — Límites por dominio para la normalización de S3 (propuesta; no entrena modelos).

Los límites [lo, hi] de cada predictor salen SOLO de documentación del protocolo y de los
extractores; no se usan los datos para fijarlos. Transformación propuesta:
    x' = log1p(clip(x - lo, 0, hi - lo)) / log1p(hi - lo)   ∈ [0, 1]

Constantes documentadas (ver BOUNDS_README de la salida):
- CICIDS2017 (CICFlowMeter): timeout de flujo T = 120 s (Engelen et al., WTMC 2021, §III-A.1).
  Duración, IAT y active/idle en microsegundos (código de CICFlowMeter, BasicFlow.java).
- HIKARI-2021 (Zeek + FlowMeter): flow_duration en segundos; IAT y active/idle en microsegundos
  (README de zeek-flowmeter). Sin timeout máximo de flujo; se acota por la duración de las
  sesiones de captura, de 3 a 5 h (Ferriyan et al., Appl. Sci. 2021, §4.5) → T = 5 h = 18.000 s.
- Contadores sin límite natural: tope genérico = capacidad de un enlace de 10 Gb/s durante T.
  Paquetes: 14.880.952 paquetes/s (tramas mínimas de 84 B en el cable, IEEE 802.3) × T.
  Bytes: 1,25e9 B/s × T. Tasas: 14.880.952 paquetes/s y 1,25e9 B/s.
- Tamaño de paquete IP ≤ 65.535 B (campo Total Length de IPv4). Cabecera por paquete ≤ 120 B
  (cabecera IPv4 ≤ 60 B + cabecera TCP ≤ 60 B). Ventana TCP 0–65.535 (campo de 16 bits; zeek-flowmeter
  ignora el window scale). Puertos 0–65.535.
- Dispersión: ambos extractores usan el estimador MUESTRAL (n-1). Para valores en [0, M], la
  desviación es ≤ M/√2 y la varianza ≤ M²/2 (máximo con n = 2). Corrección del 2026-10-07; antes M/2 y (M/2)².

Diagnóstico (no modifica los límites): % de valores del train por debajo de lo y por encima de hi.
"""
import numpy as np
import pandas as pd

from _common import add_datasets_arg, setup
from src.data import load_raw, make_predictors
from src.families import family_of
from src.splits import load_split

PPS_10G = 14_880_952          # paquetes/s a 10 Gb/s con tramas mínimas (84 B en el cable)
BPS_10G = 1.25e9              # bytes/s a 10 Gb/s
IP_MAX = 65535
HDR_MAX = 120                 # CICFlowMeter: IPv4 ≤ 60 + TCP ≤ 60
HDR_MAX_ZEEK = 60             # zeek-flowmeter: solo cabecera de transporte (TCP ≤ 60; UDP/ICMP = 8)
WIN_MAX = 65535
PORT_MAX = 65535

SRC_CIC_T = "Timeout 120 s en CICIDS2017 (Engelen et al., WTMC 2021, §III-A.1)"
SRC_CIC_US = "CICFlowMeter BasicFlow.java: marcas de tiempo en microsegundos"
SRC_ZEEK = "README de zeek-flowmeter (unidades por variable)"
SRC_HIK_T = "Sesiones de captura de 3–5 h (Ferriyan et al., Appl. Sci. 2021, §4.5)"
SRC_LINK = "Tope genérico: enlace de 10 Gb/s (14,88 Mpps con tramas mínimas; 1,25e9 B/s) durante T"
SRC_IP = "Campo Total Length de IPv4 (16 bits): 65.535 B"
SRC_HDR = "Cabecera por paquete ≤ 120 B (IPv4 ≤ 60 + TCP ≤ 60)"
SRC_HDR_ZEEK = ("zeek-flowmeter: header_size = p$tcp$hl (TCP, campo Data Offset de 4 bits → ≤ 60 B) "
                "o 8 (UDP/ICMP); corrección por definición de formato del 2026-10-07 (antes 120)")
SRC_WIN = "Campo Window de TCP (16 bits)"
SRC_PORT = "Campo de puerto de TCP/UDP (16 bits)"
SRC_STD = ("Estimador muestral (n-1): desviación de valores en [0, M] ≤ M/√2 y varianza ≤ M²/2 (máximo con n = 2). "
           "CICFlowMeter: Apache Commons Math SummaryStatistics (varianza con n-1); zeek-flowmeter: std/(len-1)")


def cicids_bound(f):
    T = 120e6                      # µs
    N = PPS_10G * 120
    B = BPS_10G * 120
    if f == "Destination Port":
        return 0, PORT_MAX, "puerto", SRC_PORT
    if f.startswith("Init_Win_bytes"):
        return -1, WIN_MAX, "bytes (ventana TCP)", SRC_WIN + ". lo = -1: el valor -1 de los CSV de CICIDS2017 se trata como centinela 'sin ventana observada' y se mapea a x'=0 (ver BOUNDS_README)"
    if f == "min_seg_size_forward":
        return 0, HDR_MAX, "bytes de cabecera", SRC_HDR + ". CORRECCIÓN DE UNIDAD: el código de CICFlowMeter (BasicFlow.java) calcula min(packet.getHeaderBytes()), no un tamaño de segmento"
    if "Header Length" in f:
        return 0, B, "bytes de cabecera (total del flujo)", SRC_LINK + "; los valores negativos (desbordamiento) se recortan a lo"
    if f in ("Flow Duration",) or "IAT" in f or f.startswith(("Active", "Idle")):
        hi = T / np.sqrt(2) if f.endswith("Std") else T
        return 0, hi, "microsegundos", f"{SRC_CIC_T}; {SRC_CIC_US}" + (f"; {SRC_STD}" if f.endswith("Std") else "")
    if f in ("Flow Bytes/s",) or f.endswith("Bulk Rate"):
        return 0, BPS_10G, "bytes/s", SRC_LINK
    if f.endswith("Packets/s"):
        return 0, PPS_10G, "paquetes/s", SRC_LINK
    if f == "Packet Length Variance":
        return 0, IP_MAX ** 2 / 2, "bytes²", f"{SRC_IP}; {SRC_STD}"
    if f.endswith("Std") and ("Packet Length" in f):
        return 0, IP_MAX / np.sqrt(2), "bytes", f"{SRC_IP}; {SRC_STD}"
    if ("Packet Length" in f) or f in ("Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size"):
        return 0, IP_MAX, "bytes (por paquete)", SRC_IP
    if f.startswith("Total Length of") or "Bytes" in f:      # totales de bytes, Subflow Bytes, Avg Bytes/Bulk
        return 0, B, "bytes", SRC_LINK
    if f in ("Total Fwd Packets", "Total Backward Packets", "act_data_pkt_fwd", "Down/Up Ratio") \
            or "Packets" in f or "Flag" in f:
        return 0, N, "paquetes (conteo o razón)", SRC_LINK
    raise ValueError(f"Sin regla de límite: {f}")


def hikari_bound(f):
    T_s = 5 * 3600
    T = T_s * 1e6                  # µs
    N = PPS_10G * T_s
    B = BPS_10G * T_s
    if f in ("originp", "responp"):
        return 0, PORT_MAX, "puerto", SRC_PORT
    if "window_size" in f:
        return 0, WIN_MAX, "bytes (ventana TCP)", SRC_WIN + "; zeek-flowmeter ignora el window scale"
    if f.startswith(("fwd_header_size", "bwd_header_size")):
        if f.endswith("_tot"):
            return 0, B, "bytes de cabecera (total del flujo)", SRC_LINK
        return 0, HDR_MAX_ZEEK, "bytes de cabecera de transporte (por paquete)", SRC_HDR_ZEEK
    if f == "flow_duration":
        return 0, T_s, "segundos", f"{SRC_HIK_T}; {SRC_ZEEK}"
    if f.startswith(("fwd_iat.", "bwd_iat.", "flow_iat.", "active.", "idle.")):
        hi = T / np.sqrt(2) if f.endswith(".std") else T
        return 0, hi, "microsegundos", f"{SRC_HIK_T}; {SRC_ZEEK}" + (f"; {SRC_STD}" if f.endswith(".std") else "")
    if "pkts_payload." in f:
        if f.endswith(".tot"):
            return 0, B, "bytes de payload (total)", SRC_LINK
        if f.endswith(".std"):
            return 0, IP_MAX / np.sqrt(2), "bytes", f"{SRC_IP}; {SRC_STD}"
        return 0, IP_MAX, "bytes de payload (por paquete)", SRC_IP
    if f in ("payload_bytes_per_second",) or f.endswith("bulk_rate"):
        return 0, BPS_10G, "bytes/s", SRC_LINK
    if f.endswith("pkts_per_sec"):
        return 0, PPS_10G, "paquetes/s", SRC_LINK
    if f.endswith(("subflow_bytes", "bulk_bytes")):
        return 0, B, "bytes", SRC_LINK
    if f.endswith(("pkts_tot", "subflow_pkts", "bulk_packets", "flag_count")) or f == "down_up_ratio":
        return 0, N, "paquetes (conteo o razón)", SRC_LINK
    raise ValueError(f"Sin regla de límite: {f}")


def transform(x, lo, hi):
    return np.log1p(np.clip(x - lo, 0, hi - lo)) / np.log1p(hi - lo)


def main():
    args, cfg, out = setup("S3-A: límites por dominio", lambda p: add_datasets_arg(
        p, ["CICIDS2017", "HIKARI2021_v2_main", "HIKARI2021_v2_pairs"]))
    out = out / "S3A_domain_bounds"
    out.mkdir(exist_ok=True)
    for dataset in args.datasets:
        df = load_raw(cfg, dataset)
        X = make_predictors(df, cfg, dataset)
        del df
        train_idx, _ = load_split(cfg, dataset)
        Xtr = X.iloc[train_idx]
        rule = cicids_bound if dataset.startswith("CICIDS") else hikari_bound
        rows = []
        for f in X.columns:
            lo, hi, unit, src = rule(f)
            v = Xtr[f].to_numpy(dtype=np.float64)
            t = transform(v, lo, hi)
            assert np.isfinite(t).all() and t.min() >= 0 and t.max() <= 1
            rows.append({"feature": f, "family": family_of(f, dataset), "lo": lo, "hi": hi, "unit": unit,
                         "source_or_justification": src,
                         "train_pct_below_lo": float(np.mean(v < lo) * 100),
                         "train_pct_above_hi": float(np.mean(v > hi) * 100),
                         "train_pct_clipped": float(np.mean((v < lo) | (v > hi)) * 100),
                         "train_median_transformed": float(np.median(t))})
        res = pd.DataFrame(rows)
        res.to_csv(out / f"{dataset}_domain_bounds.csv", index=False)
        pd.set_option("display.width", 250)
        print(f"== {dataset}: {len(res)} variables; con recorte > 0: {(res.train_pct_clipped > 0).sum()}")
        print(res[res.train_pct_clipped > 0][["feature", "lo", "hi", "train_pct_below_lo", "train_pct_above_hi"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
