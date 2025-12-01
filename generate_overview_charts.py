#!/usr/bin/env python3
"""
Gera gráficos multi-dispositivo (MAD/FFT × delegates) a partir de benchmark_results.csv.

Produz duas famílias de gráficos por algoritmo/delegate:
- single: batch_size == 1
- batch: batch_size > 1 (ex.: 12 pacotes nas medições de 30-11)

Cada gráfico usa o tamanho efetivo do vetor (4096, 8192, 16384) no eixo X e
plota uma linha por dispositivo (Galaxy S21, Moto G04s, Moto G84).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib

BASE_DIR = Path(__file__).parent
MPL_CACHE = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DELEGATES = ["CPU Kotlin", "TFLite CPU", "TFLite GPU", "TFLite NNAPI"]
ALGORITHMS = ["MAD", "FFT"]
DEVICE_COLORS = {
    "Galaxy S21 (30-11)": "#1f77b4",
    "Moto G04s (30-11)": "#ff7f0e",
    "Moto G84 (30-11)": "#2ca02c",
}


def detect_algorithm(name: str) -> str:
    upper = str(name).upper()
    if "MAD" in upper:
        return "MAD"
    if "FFT" in upper:
        return "FFT"
    return "Outro"


def derive_vector_length(row: pd.Series) -> int:
    batch = int(row.get("batch_size", 1) or 1)
    input_size = int(row.get("input_size", 0) or 0)
    if row["algorithm"] == "FFT":
        sensors = 10
        if input_size == 0:
            return 0
        return int(input_size / (batch * sensors))
    # MAD
    return int(input_size / batch) if batch else int(input_size)


def prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["device_model"] = df["device_model"].astype(str).str.strip()
    df["delegate"] = df["delegate"].astype(str).str.strip()
    df["algorithm"] = df["test_name"].apply(detect_algorithm)
    df["is_batch"] = df["test_name"].str.contains("x10", case=False, na=False)
    df["vector_length"] = df.apply(derive_vector_length, axis=1)
    df = df[df["algorithm"].isin(ALGORITHMS)]
    df = df[df["delegate"].isin(DELEGATES)]
    df = df[df["vector_length"] > 0]
    return df


def ensure_output_dirs(base: Path, algorithms: Iterable[str], delegates: Iterable[str]) -> None:
    for algo in algorithms:
        for delegate in delegates:
            (base / algo / delegate.replace(" ", "_")).mkdir(parents=True, exist_ok=True)


def plot_lines(
    data: pd.DataFrame,
    algorithm: str,
    delegate: str,
    *,
    batch_mode: bool,
    output_dir: Path,
) -> None:
    title_mode = "Batch (xN pacotes)" if batch_mode else "Single (1 pacote)"
    filename_mode = "batch" if batch_mode else "single"
    subset = data[
        (data["algorithm"] == algorithm)
        & (data["delegate"] == delegate)
        & (data["is_batch"] == batch_mode)
    ].copy()
    if subset.empty:
        return
    subset = subset.sort_values("vector_length")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 4.5))
    ax = sns.lineplot(
        data=subset,
        x="vector_length",
        y="duration_ms",
        hue="device_model",
        marker="o",
        palette=DEVICE_COLORS,
    )
    ax.set_xlabel("Tamanho efetivo do vetor (amostras)")
    ax.set_ylabel("Tempo total médio (ms)")
    ax.set_title(f"{algorithm} — {delegate} — {title_mode}")
    ax.set_xticks(sorted(subset["vector_length"].unique()))
    ax.legend(title="Dispositivo")
    plt.tight_layout()
    filename = f"{algorithm.lower()}_{delegate.replace(' ', '_').lower()}_{filename_mode}.png"
    plt.savefig(output_dir / algorithm / delegate.replace(" ", "_") / filename, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos multi-dispositivo (MAD/FFT × delegates).")
    parser.add_argument(
        "--csv",
        required=True,
        help="Caminho para benchmark_results.csv consolidado.",
    )
    parser.add_argument(
        "--output",
        default="docs/charts/comparativo-30-11/overview",
        help="Diretório onde os PNGs serão salvos.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(csv_path)
    ensure_output_dirs(output_dir, ALGORITHMS, DELEGATES)

    for algo in ALGORITHMS:
        for delegate in DELEGATES:
            plot_lines(data, algo, delegate, batch_mode=False, output_dir=output_dir)
            plot_lines(data, algo, delegate, batch_mode=True, output_dir=output_dir)

    print(f"Gráficos multi-dispositivo salvos em {output_dir}")


if __name__ == "__main__":
    main()
