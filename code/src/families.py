"""Familias de variables por dataset (para importancia por permutación y ablación).

La primera regla que coincide asigna la familia. Las columnas generalizadas de S1b (sufijo _iana)
pertenecen a la familia de puertos.
"""
import re

HIKARI_RULES = [
    ("ports", r"^(originp|responp)(_iana)?$"),
    ("tcp_window", r"window_size"),
    ("header_sizes", r"header_size"),
    ("flags", r"flag_count"),
    ("payload_packet_sizes", r"pkts_payload\."),
    ("timing", r"^(flow_duration|fwd_iat\.|bwd_iat\.|flow_iat\.|active\.|idle\.)"),
    ("volume_rates_bulk_subflow", r"(pkts_tot|_per_sec|per_second|down_up_ratio|subflow|bulk)"),
]

CICIDS_RULES = [
    ("ports", r"^Destination Port(_iana)?$"),
    ("tcp_window", r"^Init_Win_bytes"),
    ("header_sizes", r"(Header Length|min_seg_size_forward)"),
    ("flags", r"(Flag Count|PSH Flags|URG Flags)"),
    ("payload_packet_sizes", r"(Packet Length|Segment Size|Average Packet Size|Total Length of)"),
    ("timing", r"(Duration|IAT|^Active|^Idle)"),
    ("volume_rates_bulk_subflow", r"(Total Fwd Packets|Total Backward Packets|/s$|Bulk|Subflow|Down/Up|act_data_pkt_fwd)"),
]


def rules_for(dataset):
    return CICIDS_RULES if dataset.startswith("CICIDS") else HIKARI_RULES


def family_of(col, dataset="HIKARI2021"):
    for fam, pattern in rules_for(dataset):
        if re.search(pattern, col):
            return fam
    raise ValueError(f"Variable sin familia: {col}")


def families(columns, dataset):
    """Diccionario familia -> lista de columnas (en el orden de `columns`)."""
    out = {}
    for c in columns:
        out.setdefault(family_of(c, dataset), []).append(c)
    return out
