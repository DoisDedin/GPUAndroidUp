#!/usr/bin/env python3
"""
Une múltiplos benchmark_results.csv em um arquivo único com a coluna `device_model`.

Exemplo:

```bash
python3 merge_benchmarks.py \
  --device "Galaxy S21 (09-12)=app/src/BANCHMARK/s21-09-12/benchmark_results.csv" \
  --device "Moto G04s (09-12)=app/src/BANCHMARK/motog04-09-12/benchmark_results.csv" \
  --device "Moto G84 (30-11)=app/src/BANCHMARK/motog-84-30-11/benchmark_results.csv" \
  --output app/src/BANCHMARK/comparativo-09-12/benchmark_results.csv
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import csv
import pandas as pd

EXTRA_TEMPERATURE_COLS = [
    "battery_temp_start_c",
    "battery_temp_end_c",
    "cpu_temp_start_c",
    "cpu_temp_end_c",
    "gpu_temp_start_c",
    "gpu_temp_end_c",
]


def parse_device_args(entries: List[str]) -> List[Tuple[Optional[str], Path]]:
    parsed: List[Tuple[Optional[str], Path]] = []
    for entry in entries:
        if "=" in entry:
            label, path = entry.split("=", 1)
            parsed.append((label.strip() or None, Path(path).expanduser()))
        else:
            parsed.append((None, Path(entry).expanduser()))
    return parsed


def fallback_device_label(df: pd.DataFrame) -> pd.Series:
    for column in ["device_model", "deviceModel", "deviceInfo", "model"]:
        if column in df.columns:
            series = df[column].fillna("").astype(str).str.strip()
            if series.ne("").any():
                return series.where(series != "", "Dispositivo")
    return pd.Series(["Dispositivo"] * len(df))


def load_with_label(csv_path: Path, label: Optional[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        df = _load_with_padding(csv_path)
    if label:
        df["device_model"] = label
    else:
        df["device_model"] = fallback_device_label(df)
    df["source_csv"] = str(csv_path)
    return df


def _load_with_padding(csv_path: Path) -> pd.DataFrame:
    with csv_path.open(newline="") as fp:
        reader = list(csv.reader(fp))
    if not reader:
        return pd.DataFrame()
    header = reader[0]
    max_len = max(len(row) for row in reader)
    if max_len > len(header):
        missing = max_len - len(header)
        if missing == len(EXTRA_TEMPERATURE_COLS):
            header = header + EXTRA_TEMPERATURE_COLS
        else:
            header = header + [f"extra_col_{i+1}" for i in range(missing)]
    padded_rows = [
        row + [""] * (len(header) - len(row)) for row in reader[1:]
    ]
    return pd.DataFrame(padded_rows, columns=header)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mescla múltiplos benchmark_results.csv etiquetando cada dispositivo.")
    parser.add_argument(
        "--device",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Par label=arquivo.csv (label opcional). Pode ser passado várias vezes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Caminho do CSV consolidado que será criado.",
    )
    args = parser.parse_args()

    entries = parse_device_args(args.device)
    frames = []
    for label, path in entries:
        frames.append(load_with_label(path, label))
        print(f"Adicionando {path} como '{label or 'auto'}' ({len(frames[-1])} linhas)")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"CSV consolidado salvo em {output_path} com {len(combined)} linhas e {len(combined.columns)} colunas.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
