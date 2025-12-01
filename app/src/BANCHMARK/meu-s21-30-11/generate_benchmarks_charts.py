#!/usr/bin/env python3
"""
Gera graficos de desempenho (MAD/FFT) para o Galaxy S21 (30-11).

- Le o CSV benchmark_results.csv.
- Considera batch_size=1 (1 execucao) e batch_size=4 (media de 4 execucoes sequenciais, nao paralelas).
- Salva PNGs em ./charts (dpi=300).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

# Evita problemas de fontconfig/cache em ambientes restritos.
BASE_DIR = Path(__file__).parent
CHARTS_DIR = BASE_DIR / "charts"
MPL_CACHE = CHARTS_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEVICE_LABEL = "Galaxy S21 (30-11)"
CSV_PATH = BASE_DIR / "benchmark_results.csv"

# Paleta neutra
PALETTE = {
    "CPU Kotlin": "#4C6EF5",      # azul
    "TFLite CPU": "#38A169",      # verde
    "TFLite GPU": "#F6AD55",      # laranja
    "TFLite NNAPI": "#718096",    # cinza
}

BATCH_LABELS = {
    1: "1 execução",
    4: "Média de 4 execuções sequenciais (batch ≠ paralelismo)",
}


def load_data(path: Path) -> pd.DataFrame:
    required = {
        "test_name",
        "delegate",
        "input_size",
        "duration_ms",
        "duration_std_ms",
        "transfer_ms",
        "compute_ms",
        "throughput_ops_per_sec",
        "batch_size",
        "iterations",
    }
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV faltando colunas: {missing}")

    # Normaliza nomes
    df["delegate"] = df["delegate"].str.replace("TFLite ", "TFLite ", regex=False)
    df["algorithm"] = df["test_name"].str.split().str[0]  # MAD ou FFT
    df["is_x10"] = df["test_name"].str.contains("x10")
    # Mantemos batch=1 e 4
    df = df[df["batch_size"].isin([1, 4])]

    def batch_label(row) -> str:
        base = BATCH_LABELS.get(row["batch_size"], f"batch={row['batch_size']}")
        return f"x10 ({base})" if row["is_x10"] else base

    df["batch_label"] = df.apply(batch_label, axis=1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(
            ["algorithm", "delegate", "input_size", "batch_size", "batch_label", "is_x10"]
        )
        .agg(
            duration_mean=("duration_ms", "mean"),
            duration_std=("duration_ms", "std"),
            transfer_mean=("transfer_ms", "mean"),
            transfer_std=("transfer_ms", "std"),
            compute_mean=("compute_ms", "mean"),
        )
        .reset_index()
    )
    return agg


def ensure_colors(delegates: Iterable[str]) -> dict:
    colors = {d: PALETTE.get(d, "#333333") for d in delegates}
    return colors


def plot_time_lines(df: pd.DataFrame, algorithm: str, *, only_tflite: bool, only_x10: bool, filename_suffix: str, title_suffix: str) -> None:
    data = df[df["algorithm"] == algorithm]
    if only_tflite:
        data = data[data["delegate"].isin(["TFLite CPU", "TFLite GPU", "TFLite NNAPI"])]
    if only_x10:
        data = data[data["is_x10"]]
    else:
        data = data[~data["is_x10"]]
    if data.empty:
        return
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
        f"{algorithm} — Tempo total médio vs tamanho (ms) — {DEVICE_LABEL}{title_suffix}",
        fontsize=13,
        weight="bold",
    )
    ax.set_xlabel("Tamanho em pontos")
    ax.set_ylabel("Tempo total médio (ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        title="Delegate / Execuções",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{algorithm.lower()}_tempo{filename_suffix}.png", dpi=300)
    plt.close()


def plot_transfer(df: pd.DataFrame, algorithm: str) -> None:
    data = df[df["algorithm"] == algorithm].copy()
    # CPU e TFLite CPU sem custo de transferencia
    data.loc[data["delegate"].isin(["CPU Kotlin", "TFLite CPU"]), "transfer_mean"] = 0.0
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
    g.set_axis_labels("Tamanho em pontos", "Tempo de transferência (ms)")
    g.fig.suptitle(
        f"{algorithm} — Custo de transferência (GPU/NNAPI) — {DEVICE_LABEL}",
        y=1.05,
        fontsize=13,
        weight="bold",
    )
    g.set_titles("{col_name}")

    # Anota porcentagem usando as barras (evita distorcer limites do eixo)
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
    g.savefig(CHARTS_DIR / f"{algorithm.lower()}_transfer.png", dpi=300)
    plt.close(g.fig)


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    cpu = df[df["delegate"] == "CPU Kotlin"][
        ["algorithm", "input_size", "batch_size", "batch_label", "is_x10", "duration_mean"]
    ].rename(columns={"duration_mean": "cpu_duration"})
    merged = df.merge(cpu, on=["algorithm", "input_size", "batch_size", "batch_label", "is_x10"])
    merged["speedup"] = merged["cpu_duration"] / merged["duration_mean"]
    merged = merged[merged["delegate"] != "CPU Kotlin"]
    return merged


def plot_speedup(df: pd.DataFrame, algorithm: str) -> None:
    data = df[df["algorithm"] == algorithm]
    colors = ensure_colors(data["delegate"].unique())
    g = sns.catplot(
        data=data,
        x="input_size",
        y="speedup",
        hue="delegate",
        col="batch_label",
        kind="bar",
        palette=colors,
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("Tamanho em pontos", "Speedup vs CPU Kotlin (×)")
    g.fig.suptitle(
        f"{algorithm} — Speedup vs CPU Kotlin — {DEVICE_LABEL}",
        y=1.05,
        fontsize=13,
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
    g.savefig(CHARTS_DIR / f"{algorithm.lower()}_speedup.png", dpi=300)
    plt.close(g.fig)


def plot_stacked_fft(df: pd.DataFrame) -> None:
    data = df[df["algorithm"] == "FFT"].copy()
    colors = ensure_colors(data["delegate"].unique())
    for batch_label, subset in data.groupby("batch_label"):
        # Ordena por input_size e garante alinhamento por delegate.
        input_sizes = sorted(subset["input_size"].unique())
        delegates = ["CPU Kotlin", "TFLite CPU", "TFLite GPU", "TFLite NNAPI"]
        delegates = [d for d in delegates if d in subset["delegate"].unique()]
        width = 0.18
        x = np.arange(len(input_sizes))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, delegate in enumerate(delegates):
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
                color=colors.get(delegate, "#333333"),
            )
        ax.set_xticks(x + width * (len(delegates) - 1) / 2)
        ax.set_xticklabels(input_sizes)
        ax.set_xlabel("Tamanho em pontos")
        ax.set_ylabel("Tempo (ms)")
        ax.set_title(
            f"FFT — Tempo empilhado (transferência + processamento)\n{batch_label} — {DEVICE_LABEL}",
            fontsize=13,
            weight="bold",
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR
            / f"fft_empilhado_{batch_label.replace(' ', '_').replace('execuções', 'exec')}.png",
            dpi=300,
        )
        plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    cols = [
        "algorithm",
        "delegate",
        "input_size",
        "batch_label",
        "duration_mean",
        "duration_std",
        "transfer_mean",
        "compute_mean",
    ]
    summary = df[cols].sort_values(["algorithm", "input_size", "delegate", "batch_label"])
    print("\nResumo (ms):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


def main() -> None:
    sns.set_theme(context="talk", style="whitegrid")
    df_raw = load_data(CSV_PATH)
    df = summarize(df_raw)
    print_summary(df)

    for algo in ["MAD", "FFT"]:
        plot_time_lines(df, algo, only_tflite=False, only_x10=False, filename_suffix="", title_suffix=" (sem x10)")
        plot_time_lines(df, algo, only_tflite=False, only_x10=True, filename_suffix="_x10", title_suffix=" (x10)")
        plot_time_lines(df, algo, only_tflite=True, only_x10=False, filename_suffix="_tflite", title_suffix=" (TFLite, sem x10)")
        plot_time_lines(df, algo, only_tflite=True, only_x10=True, filename_suffix="_tflite_x10", title_suffix=" (TFLite, x10)")
        plot_transfer(df, algo)

    speed = compute_speedup(df)
    for algo in ["MAD", "FFT"]:
        plot_speedup(speed, algo)

    plot_stacked_fft(df)
    print(f"\nPNGs gerados em: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
