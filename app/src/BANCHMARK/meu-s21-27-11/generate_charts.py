#!/usr/bin/env python3
"""
Gera graficos de desempenho (MAD/FFT) para o S21.

- Conjunto A (charts/all): todos os delegates, eixo Y log na vista geral para caber CPU.
- Conjunto B (charts/tflite-only): apenas TFLite CPU/GPU/NNAPI, foco em aceleracao real.

Notas importantes:
- batch=4: "Média de quatro execuções sequenciais, não paralelas".
- Separamos conjuntos porque o baseline CPU é ordens de grandeza mais lento e distorce a escala,
  impedindo comparação clara entre aceleradores.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

# Diretórios e cache
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "benchmark_results.csv"
CHARTS_ALL = BASE_DIR / "charts" / "all"
CHARTS_TFLITE = BASE_DIR / "charts" / "tflite-only"
MPL_CACHE = BASE_DIR / "charts" / ".matplotlib"
for d in [CHARTS_ALL, CHARTS_TFLITE, MPL_CACHE]:
    d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paleta consistente
PALETTE = {
    "CPU Kotlin": "#6B7280",   # cinza
    "TFLite CPU": "#1D4ED8",   # azul
    "TFLite GPU": "#10B981",   # verde
    "TFLite NNAPI": "#F59E0B", # laranja
}

BATCH_LABELS = {
    1: "1 execução",
    4: "Média de 4 execuções sequenciais (batch ≠ paralelismo)",
}

DEVICE_LABEL = "Galaxy S21 (27-11)"


def load_data(path: Path) -> pd.DataFrame:
    required = {
        "test_name",
        "delegate",
        "input_size",
        "duration_ms",
        "duration_std_ms",
        "transfer_ms",
        "compute_ms",
        "batch_size",
        "iterations",
    }
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV faltando colunas: {missing}")

    df = df[df["batch_size"].isin([1, 4])].copy()
    df["algorithm"] = df["test_name"].str.split().str[0]
    df["batch_label"] = df["batch_size"].map(BATCH_LABELS)

    # CPU e TFLite CPU sem custo de transferência
    df.loc[df["delegate"].isin(["CPU Kotlin", "TFLite CPU"]), "transfer_ms"] = 0.0

    # Agrega médias/desvios
    agg = (
        df.groupby(
            ["algorithm", "delegate", "input_size", "batch_size", "batch_label"],
            as_index=False,
        )
        .agg(
            duration_mean=("duration_ms", "mean"),
            duration_std=("duration_ms", "std"),
            transfer_mean=("transfer_ms", "mean"),
            transfer_std=("transfer_ms", "std"),
            compute_mean=("compute_ms", "mean"),
        )
    )
    # speedup precisa do baseline CPU
    cpu = agg[agg["delegate"] == "CPU Kotlin"][
        ["algorithm", "input_size", "batch_size", "batch_label", "duration_mean"]
    ].rename(columns={"duration_mean": "cpu_duration"})
    agg = agg.merge(cpu, on=["algorithm", "input_size", "batch_size", "batch_label"], how="left")
    agg["speedup_vs_cpu"] = agg["cpu_duration"] / agg["duration_mean"]
    return agg


def ensure_colors(delegates: Iterable[str]) -> dict:
    return {d: PALETTE.get(d, "#374151") for d in delegates}


def savepath(subdir: Path, name: str) -> Path:
    return subdir / f"{name}.png"


def plot_time_all(df: pd.DataFrame, algorithm: str) -> None:
    data = df[df["algorithm"] == algorithm]
    colors = ensure_colors(data["delegate"].unique())
    plt.figure(figsize=(11, 5))
    ax = sns.lineplot(
        data=data,
        x="input_size",
        y="duration_mean",
        hue="delegate",
        style="batch_label",
        markers=True,
        dashes=False,
        palette=colors,
        linewidth=2,
        markersize=8,
    )
    # barras de erro por serie
    for (_, _), sub in data.groupby(["delegate", "batch_label"]):
        ax.errorbar(
            sub["input_size"],
            sub["duration_mean"],
            yerr=sub["duration_std"],
            fmt="none",
            ecolor="black",
            alpha=0.2,
            capsize=4,
        )
    ax.set_title(
        f"{algorithm} — Tempo total vs tamanho (ms) — Escala logarítmica para visualização global — {DEVICE_LABEL}",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Pontos")
    ax.set_ylabel("Tempo (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(
        title="Delegate / Execuções",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(savepath(CHARTS_ALL, f"{algorithm.lower()}_tempo_log"), dpi=300)
    plt.close()


def plot_transfer_all(df: pd.DataFrame, algorithm: str) -> None:
    data = df[df["algorithm"] == algorithm].copy()
    data["transfer_ratio"] = np.where(
        data["duration_mean"] > 0,
        data["transfer_mean"] / data["duration_mean"],
        0.0,
    )
    colors = ensure_colors(data["delegate"].unique())
    g = sns.catplot(
        data=data,
        x="input_size",
        y="transfer_mean",
        hue="delegate",
        col="batch_label",
        kind="bar",
        palette=colors,
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Pontos", "Tempo de transferência (ms)")
    g.fig.suptitle(
        f"{algorithm} — Overhead de transferência — {DEVICE_LABEL}",
        y=1.05,
        fontsize=12,
        weight="bold",
    )
    g.set_titles("{col_name}")
    for ax, (_, sub) in zip(g.axes.flatten(), data.groupby("batch_label")):
        sub_sorted = sub.sort_values(["input_size", "delegate"])
        for patch, (_, row) in zip(ax.patches, sub_sorted.iterrows()):
            height = patch.get_height()
            if height == 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            ax.text(
                x,
                height + 0.05,
                f"{row['transfer_ratio']*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_title("Delegate")
    plt.tight_layout()
    g.savefig(savepath(CHARTS_ALL, f"{algorithm.lower()}_transfer"), dpi=300)
    plt.close(g.fig)


def plot_speedup_all(df: pd.DataFrame, algorithm: str) -> None:
    data = df[(df["algorithm"] == algorithm) & (df["delegate"] != "CPU Kotlin")]
    colors = ensure_colors(data["delegate"].unique())
    g = sns.catplot(
        data=data,
        x="input_size",
        y="speedup_vs_cpu",
        hue="delegate",
        col="batch_label",
        kind="bar",
        palette=colors,
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Pontos", "Speedup vs CPU Kotlin (×)")
    g.fig.suptitle(
        f"{algorithm} — Speedup vs CPU Kotlin — {DEVICE_LABEL}",
        y=1.05,
        fontsize=12,
        weight="bold",
    )
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_title("Delegate")
    plt.tight_layout()
    g.savefig(savepath(CHARTS_ALL, f"{algorithm.lower()}_speedup"), dpi=300)
    plt.close(g.fig)


def plot_stacked_fft_all(df: pd.DataFrame) -> None:
    data = df[df["algorithm"] == "FFT"].copy()
    colors = ensure_colors(data["delegate"].unique())
    delegates_order = ["CPU Kotlin", "TFLite CPU", "TFLite GPU", "TFLite NNAPI"]
    delegates_order = [d for d in delegates_order if d in data["delegate"].unique()]
    for batch_label, subset in data.groupby("batch_label"):
        input_sizes = sorted(subset["input_size"].unique())
        width = 0.18
        x = np.arange(len(input_sizes))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, delegate in enumerate(delegates_order):
            sub = subset[subset["delegate"] == delegate].set_index("input_size")
            transfer_vals = [sub.loc[size, "transfer_mean"] if size in sub.index else 0 for size in input_sizes]
            compute_vals = [sub.loc[size, "compute_mean"] if size in sub.index else 0 for size in input_sizes]
            ax.bar(
                x + i * width,
                transfer_vals,
                width=width,
                label=f"{delegate} - Transferência",
                color="#CBD5E0",
            )
            ax.bar(
                x + i * width,
                compute_vals,
                width=width,
                bottom=transfer_vals,
                label=f"{delegate} - Processamento",
                color=colors.get(delegate, "#374151"),
            )
        ax.set_xticks(x + width * (len(delegates_order) - 1) / 2)
        ax.set_xticklabels(input_sizes)
        ax.set_xlabel("Pontos")
        ax.set_ylabel("Tempo (ms)")
        ax.set_title(
            f"FFT — Tempo empilhado (transferência + processamento)\n{batch_label} — {DEVICE_LABEL}",
            fontsize=12,
            weight="bold",
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(savepath(CHARTS_ALL, f"fft_empilhado_{batch_label.replace(' ', '_')}"), dpi=300)
        plt.close(fig)


# ------------ Conjunto B: apenas TFLite ------------
def plot_time_tflite(df: pd.DataFrame, algorithm: str) -> None:
    data = df[(df["algorithm"] == algorithm) & (df["delegate"] != "CPU Kotlin")]
    colors = ensure_colors(data["delegate"].unique())
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(
        data=data,
        x="input_size",
        y="duration_mean",
        hue="delegate",
        style="batch_label",
        markers=True,
        dashes=False,
        palette=colors,
        linewidth=2,
        markersize=8,
    )
    for (_, _), sub in data.groupby(["delegate", "batch_label"]):
        ax.errorbar(
            sub["input_size"],
            sub["duration_mean"],
            yerr=sub["duration_std"],
            fmt="none",
            ecolor="black",
            alpha=0.2,
            capsize=4,
        )
    ax.set_title(
        f"{algorithm} — Comparação de delegates TFLite sem baseline CPU Kotlin",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Pontos")
    ax.set_ylabel("Tempo (ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        title="Delegate / Execuções",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(savepath(CHARTS_TFLITE, f"{algorithm.lower()}_tempo_tflite"), dpi=300)
    plt.close()


def plot_transfer_compute_tflite(df: pd.DataFrame, algorithm: str) -> None:
    data = df[(df["algorithm"] == algorithm) & (df["delegate"] != "CPU Kotlin")].copy()
    colors = ensure_colors(data["delegate"].unique())
    # anotacao de % overhead
    data["transfer_ratio"] = np.where(
        data["duration_mean"] > 0,
        data["transfer_mean"] / data["duration_mean"],
        0.0,
    )
    g = sns.catplot(
        data=data,
        x="input_size",
        y="transfer_mean",
        hue="delegate",
        col="batch_label",
        kind="bar",
        palette=colors,
        height=4,
        aspect=1.1,
    )
    for i, ax in enumerate(g.axes.flatten()):
        batch = data["batch_label"].unique()[i]
        sub = data[data["batch_label"] == batch].sort_values(["input_size", "delegate"])
        for patch, (_, row) in zip(ax.patches, sub.iterrows()):
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2
            ax.text(
                x,
                height + 0.05,
                f"{row['transfer_ratio']*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if g._legend:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_title("Delegate")
    g.set_axis_labels("Pontos", "Tempo de transferência (ms)")
    g.fig.suptitle(
        f"{algorithm} — Transferência (GPU/NNAPI) — Delegates TFLite",
        y=1.05,
        fontsize=12,
        weight="bold",
    )
    g.set_titles("{col_name}")
    plt.tight_layout()
    g.savefig(savepath(CHARTS_TFLITE, f"{algorithm.lower()}_transfer_tflite"), dpi=300)
    plt.close(g.fig)


def print_speedup_vs_tflite_cpu(df: pd.DataFrame) -> None:
    base = df[(df["delegate"] == "TFLite CPU")][
        ["algorithm", "input_size", "batch_size", "batch_label", "duration_mean"]
    ].rename(columns={"duration_mean": "tflite_cpu_duration"})
    merged = df.merge(
        base,
        on=["algorithm", "input_size", "batch_size", "batch_label"],
        how="left",
        suffixes=("", "_base"),
    )
    merged = merged[merged["delegate"].isin(["TFLite GPU", "TFLite NNAPI"])]
    merged["speedup_vs_tflite_cpu"] = merged["tflite_cpu_duration"] / merged["duration_mean"]
    cols = [
        "algorithm",
        "delegate",
        "input_size",
        "batch_label",
        "duration_mean",
        "tflite_cpu_duration",
        "speedup_vs_tflite_cpu",
    ]
    print("\nSpeedup vs TFLite CPU:")
    print(
        merged[cols]
        .sort_values(["algorithm", "input_size", "delegate", "batch_label"])
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )


def main() -> None:
    sns.set_theme(context="talk", style="whitegrid")
    df = load_data(CSV_PATH)

    # Conjunto A (todos)
    for algo in ["MAD", "FFT"]:
        plot_time_all(df, algo)
        plot_transfer_all(df, algo)
        plot_speedup_all(df, algo)
    plot_stacked_fft_all(df)

    # Conjunto B (apenas TFLite)
    for algo in ["MAD", "FFT"]:
        plot_time_tflite(df, algo)
        plot_transfer_compute_tflite(df, algo)

    print_speedup_vs_tflite_cpu(df)
    print(f"\nPNGs (conjunto A) em: {CHARTS_ALL}")
    print(f"PNGs (conjunto B) em: {CHARTS_TFLITE}")


if __name__ == "__main__":
    main()
