"""Reconstruye datasets/CICIDS2017_CLEAN.parquet a partir de sus 4 partes (GitHub limita los archivos a 100 MB).

Uso (desde code/):  py -3.10 scripts/prepare_cicids_parquet.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DS = Path(__file__).resolve().parents[2] / "datasets"
EXPECTED_ROWS = 2519297


def main():
    parts = sorted(DS.glob("CICIDS2017_CLEAN_part*of4.parquet"))
    assert len(parts) == 4, f"Se esperaban 4 partes y hay {len(parts)}"
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    assert table.num_rows == EXPECTED_ROWS, table.num_rows
    pq.write_table(table, DS / "CICIDS2017_CLEAN.parquet")
    print(f"CICIDS2017_CLEAN.parquet: {table.num_rows} filas x {table.num_columns} columnas")


if __name__ == "__main__":
    main()
