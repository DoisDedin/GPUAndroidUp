#!/usr/bin/env python3
"""
Gera graficos a partir do benchmark_results.csv do S21 (27-11).

Saidas:
- app/src/BANCHMARK/meu-s21-27-11/plots/mad_tempo.png
- app/src/BANCHMARK/meu-s21-27-11/plots/mad_tempo_x10.png
- app/src/BANCHMARK/meu-s21-27-11/plots/fft_tempo.png
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "benchmark_results.csv"
PLOTS_DIR = BASE_DIR / "plots"
MPL_CACHE = PLOTS_DIR / ".matplotlib"
DEVICE_LABEL = "Galaxy S21 (27-11)"

# Ajuste antecipado para evitar problemas de cache do matplotlib/fontconfig.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dirs() -> None:
    """Cria pastas de saida e cache do matplotlib."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def clear_old_plots() -> None:
    """Remove PNGs antigos para evitar confusao."""
    for png in PLOTS_DIR.glob("*.png"):
        png.unlink(missing_ok=True)


def filter_and_aggregate(df: pd.DataFrame, prefix: str, include_x10: bool) -> pd.DataFrame:
    """Filtra por operacao (prefixo) e agrega por delegate/input_size/batch_size."""
    filtered = df[df["test_name"].str.startswith(prefix)].copy()
    mask_x10 = filtered["test_name"].str.contains("x10")
    filtered = filtered[mask_x10] if include_x10 else filtered[~mask_x10]
    grouped = (
        filtered.groupby(["delegate", "input_size", "batch_size"])[
            ["duration_ms", "compute_ms", "throughput_ops_per_sec"]
        ]
        .mean()
        .reset_index()
        .sort_values("input_size")
    )
    return grouped


def plot_duration(data: pd.DataFrame, title: str, filename: str, logy: bool = False) -> None:
    """Gera linha de tempo medio vs tamanho de batch."""
    plt.figure(figsize=(11, 5))
    ax = sns.lineplot(
        data=data,
        x="input_size",
        y="duration_ms",
        hue="delegate",
        style="batch_size",
        markers=True,
        dashes=False,
        linewidth=2.0,
        markersize=7,
    )
    ax.set_title(f"{title} — {DEVICE_LABEL}", fontsize=13, weight="bold")
    ax.set_xlabel("Tamanho do batch de memoria")
    ax.set_ylabel("Tempo medio por execucao (ms)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)
    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("Tempo medio por execucao (ms, escala log)")
    # Legenda fora para nao cobrir o grafico
    ax.legend(
        title="Tecnologia / batch_size",
        frameon=True,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_throughput(data: pd.DataFrame, title: str, filename: str) -> None:
    """Barra de throughput, facet por batch_size."""
    g = sns.catplot(
        data=data,
        x="input_size",
        y="throughput_ops_per_sec",
        hue="delegate",
        col="batch_size",
        kind="bar",
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Tamanho do batch de memoria", "Throughput (operacoes por segundo)")
    g.fig.suptitle(f"{title} — {DEVICE_LABEL}", y=1.05, fontsize=13, weight="bold")
    g.set_titles("batch = {col_name}")
    # Legenda fora
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_title("Tecnologia")
    plt.tight_layout()
    g.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close(g.fig)


def compute_speedup_vs_cpu(data: pd.DataFrame, baseline_delegate: str) -> pd.DataFrame:
    """Calcula speedup em relacao ao baseline (CPU Kotlin)."""
    baseline = (
        data[data["delegate"] == baseline_delegate][
            ["input_size", "batch_size", "duration_ms"]
        ]
        .rename(columns={"duration_ms": "baseline_ms"})
    )
    merged = data.merge(baseline, on=["input_size", "batch_size"], how="left")
    merged["speedup_vs_cpu"] = merged["baseline_ms"] / merged["duration_ms"]
    merged = merged[merged["delegate"] != baseline_delegate]
    return merged


def plot_speedup(data: pd.DataFrame, title: str, filename: str) -> None:
    """Barra de speedup vs CPU Kotlin."""
    g = sns.catplot(
        data=data,
        x="input_size",
        y="speedup_vs_cpu",
        hue="delegate",
        col="batch_size",
        kind="bar",
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Tamanho do batch de memoria", "Speedup vs CPU Kotlin (x)")
    g.fig.suptitle(f"{title} — {DEVICE_LABEL}", y=1.05, fontsize=13, weight="bold")
    g.set_titles("batch = {col_name}")
    for ax in g.axes.flatten():
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    # Legenda fora
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_title("Tecnologia")
    plt.tight_layout()
    g.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close(g.fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(context="talk", style="whitegrid", palette="tab10")
    clear_old_plots()
    df = load_data()

    mad = filter_and_aggregate(df, prefix="MAD ", include_x10=False)
    mad_small = mad[mad["input_size"].isin([4096, 8192, 16384])]
    plot_duration(mad_small, "MAD - tempo vs tamanho do batch (single)", "mad_tempo.png")
    plot_throughput(mad_small, "MAD - throughput por batch", "mad_throughput.png")
    mad_speed = compute_speedup_vs_cpu(mad_small, baseline_delegate="CPU Kotlin")
    plot_speedup(mad_speed, "MAD - speedup vs CPU Kotlin", "mad_speedup.png")

    mad_x10 = filter_and_aggregate(df, prefix="MAD ", include_x10=True)
    plot_duration(mad_x10, "MAD - tempo vs tamanho do batch (x10)", "mad_tempo_x10.png")
    plot_throughput(mad_x10, "MAD - throughput por batch (x10)", "mad_throughput_x10.png")
    mad_speed_x10 = compute_speedup_vs_cpu(mad_x10, baseline_delegate="CPU Kotlin")
    plot_speedup(mad_speed_x10, "MAD - speedup vs CPU Kotlin (x10)", "mad_speedup_x10.png")

    fft = filter_and_aggregate(df, prefix="FFT ", include_x10=False)
    plot_duration(
        fft, "FFT - tempo vs tamanho do batch", "fft_tempo.png", logy=True
    )
    plot_throughput(fft, "FFT - throughput por batch", "fft_throughput.png")
    fft_speed = compute_speedup_vs_cpu(fft, baseline_delegate="CPU Kotlin")
    plot_speedup(fft_speed, "FFT - speedup vs CPU Kotlin", "fft_speedup.png")

    fft_x10 = filter_and_aggregate(df, prefix="FFT ", include_x10=True)
    plot_duration(
        fft_x10, "FFT - tempo vs tamanho do batch (x10)", "fft_tempo_x10.png", logy=True
    )
    plot_throughput(fft_x10, "FFT - throughput por batch (x10)", "fft_throughput_x10.png")
    fft_speed_x10 = compute_speedup_vs_cpu(fft_x10, baseline_delegate="CPU Kotlin")
    plot_speedup(fft_speed_x10, "FFT - speedup vs CPU Kotlin (x10)", "fft_speedup_x10.png")

    print("Graficos gerados em:", PLOTS_DIR)


if __name__ == "__main__":
    main()
