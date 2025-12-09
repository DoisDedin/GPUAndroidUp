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
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List

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
KNOWN_DEVICE_COLORS = {
    "Galaxy S21 (30-11)": "#1f77b4",
    "Galaxy S21 (09-12)": "#1f77b4",
    "Moto G04s (30-11)": "#ff7f0e",
    "Moto G04s (09-12)": "#ff7f0e",
    "Moto G84 (30-11)": "#2ca02c",
    "Moto G84 (09-12)": "#2ca02c",
}
DELEGATE_COLORS = {
    "CPU Kotlin": "#7f7f7f",
    "TFLite CPU": "#1f77b4",
    "TFLite GPU": "#2ca02c",
    "TFLite NNAPI": "#ff7f0e",
}
FALLBACK_COLORS = [
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#17becf",
    "#bcbd22",
    "#d62728",
]
TITLE_FONTSIZE = 12
TITLE_PAD = 14


def format_vector_label(length: int) -> str:
    if length >= 1024:
        if length % 1024 == 0:
            return f"{int(length // 1024)}k"
        return f"{length / 1024:.1f}k"
    return str(int(length))


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


def _fallback_device_series(df: pd.DataFrame) -> pd.Series:
    if "deviceModel" in df.columns:
        return df["deviceModel"]
    if "deviceInfo" in df.columns:
        return df["deviceInfo"]
    if "model" in df.columns:
        return df["model"]
    return pd.Series(["Dispositivo"] * len(df))


def ensure_device_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "device_model" in df.columns:
        df["device_model"] = df["device_model"].astype(str).str.strip()
        empties = df["device_model"] == ""
        if empties.any():
            fallback = _fallback_device_series(df)
            df.loc[empties, "device_model"] = (
                fallback.fillna("Dispositivo").astype(str).str.strip().where(lambda s: s != "", "Dispositivo")
            )
        return df

    fallback = _fallback_device_series(df)
    df["device_model"] = fallback.fillna("Dispositivo").astype(str).str.strip()
    df.loc[df["device_model"] == "", "device_model"] = "Dispositivo"
    return df


def build_device_palette(devices: List[str]) -> Dict[str, str]:
    palette: Dict[str, str] = {}
    extra_colors = cycle(FALLBACK_COLORS)
    for device in devices:
        if device in KNOWN_DEVICE_COLORS:
            palette[device] = KNOWN_DEVICE_COLORS[device]
        else:
            color = next(extra_colors)
            while color in palette.values():
                color = next(extra_colors)
            palette[device] = color
    return palette


def prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_device_labels(df)
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
    (base / "combined").mkdir(parents=True, exist_ok=True)


def apply_vector_ticks(ax, values: Iterable[int]) -> None:
    ordered = sorted(values)
    ax.set_xticks(ordered)
    ax.set_xticklabels([format_vector_label(v) for v in ordered])


def apply_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)


def finalize_plot() -> None:
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])


def plot_lines(
    data: pd.DataFrame,
    algorithm: str,
    delegate: str,
    *,
    batch_mode: bool,
    output_dir: Path,
    palette: Dict[str, str],
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
        palette=palette,
    )
    ax.set_xlabel("Tamanho efetivo do vetor (amostras)")
    ax.set_ylabel("Tempo total médio (ms)")
    apply_title(ax, f"{algorithm} — {delegate} — {title_mode}")
    apply_vector_ticks(ax, subset["vector_length"].unique())
    ax.legend(title="Dispositivo", fontsize=9, title_fontsize=10)
    finalize_plot()
    filename = f"{algorithm.lower()}_{delegate.replace(' ', '_').lower()}_{filename_mode}.png"
    plt.savefig(output_dir / algorithm / delegate.replace(" ", "_") / filename, dpi=300)
    plt.close()


def plot_combined_delegate_chart(
    data: pd.DataFrame,
    algorithm: str,
    *,
    batch_mode: bool,
    output_dir: Path,
) -> None:
    title_mode = "Batch (xN pacotes)" if batch_mode else "Single (1 pacote)"
    filename_mode = "batch" if batch_mode else "single"
    subset = data[
        (data["algorithm"] == algorithm)
        & (data["is_batch"] == batch_mode)
        & (data["delegate"].isin(DELEGATES))
    ].copy()
    if subset.empty:
        return
    subset = subset.sort_values("vector_length")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=subset,
        x="vector_length",
        y="duration_ms",
        hue="delegate",
        style="device_model",
        markers=True,
        palette=DELEGATE_COLORS,
    )
    ax.set_xlabel("Tamanho efetivo do vetor (amostras)")
    ax.set_ylabel("Tempo total médio (ms)")
    apply_title(ax, f"{algorithm} — Delegates completos — {title_mode}")
    apply_vector_ticks(ax, subset["vector_length"].unique())
    ax.legend(title="Delegate / Dispositivo", loc="best", fontsize=9, title_fontsize=10)
    finalize_plot()
    target_dir = output_dir / "combined" / algorithm
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{algorithm.lower()}_{filename_mode}_delegates.png"
    plt.savefig(target_dir / filename, dpi=300)
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
    devices = sorted(data["device_model"].unique())
    palette = build_device_palette(devices)
    ensure_output_dirs(output_dir, ALGORITHMS, DELEGATES)

    for algo in ALGORITHMS:
        for delegate in DELEGATES:
            plot_lines(data, algo, delegate, batch_mode=False, output_dir=output_dir, palette=palette)
            plot_lines(data, algo, delegate, batch_mode=True, output_dir=output_dir, palette=palette)
        plot_combined_delegate_chart(data, algo, batch_mode=False, output_dir=output_dir)
        plot_combined_delegate_chart(data, algo, batch_mode=True, output_dir=output_dir)

    print(f"Gráficos multi-dispositivo salvos em {output_dir}")


if __name__ == "__main__":
    main()
