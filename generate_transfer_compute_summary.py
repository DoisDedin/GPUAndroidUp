#!/usr/bin/env python3
"""Gera um único gráfico combinando MAD e FFT, três dispositivos e delegates TFLite.

O gráfico possui uma matriz de subplots (algoritmo × modo single/batch). Cada
subplot mostra barras agrupadas por dispositivo e divididas em transferência e
processamento para cada delegate (TFLite CPU/GPU/NNAPI).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib

BASE_DIR = Path(__file__).parent
MPL_CACHE = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

DELEGATES = ["CPU Kotlin", "TFLite CPU", "TFLite GPU", "TFLite NNAPI"]
ALGORITHMS = ["MAD", "FFT"]
KNOWN_DEVICE_ORDER = [
    "Galaxy S21 (09-12)",
    "Moto G04s (09-12)",
    "Moto G84 (09-12)",
    "Galaxy S21 (30-11)",
    "Moto G04s (30-11)",
    "Moto G84 (30-11)",
]
BATCH_MODES = [True]  # apenas x10
DELEGATE_COLORS = {
    "CPU Kotlin": "#4a5568",
    "TFLite CPU": "#2b6cb0",
    "TFLite GPU": "#38a169",
    "TFLite NNAPI": "#d69e2e",
}
TRANSFER_COLOR = "#CBD5E0"


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


def determine_device_order(devices: List[str]) -> List[str]:
    order: List[str] = []
    for known in KNOWN_DEVICE_ORDER:
        if known in devices:
            order.append(known)
    for device in sorted(devices):
        if device not in order:
            order.append(device)
    return order


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
        return int(input_size / (batch * sensors)) if input_size else 0
    return int(input_size / batch) if batch else int(input_size)


def prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_device_labels(df)
    df["delegate"] = df["delegate"].astype(str).str.strip()
    df["algorithm"] = df["test_name"].apply(detect_algorithm)
    df["is_batch"] = df["batch_size"].fillna(1).astype(int) > 1
    df["vector_length"] = df.apply(derive_vector_length, axis=1)
    df = df[df["algorithm"].isin(ALGORITHMS)]
    df = df[df["delegate"].isin(DELEGATES)]
    df = df[df["vector_length"] > 0]
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["algorithm", "delegate", "device_model", "is_batch"])
        .agg(
            transfer_mean=("transfer_ms", "mean"),
            compute_mean=("compute_ms", "mean"),
        )
        .reset_index()
    )
    # garante que CPU tenha transferência zero e compute como tempo total
    cpu_mask = agg["delegate"] == "CPU Kotlin"
    agg.loc[cpu_mask, "transfer_mean"] = 0.0
    agg.loc[cpu_mask, "compute_mean"] = agg.loc[cpu_mask, "compute_mean"].fillna(0.0)
    return agg


def plot_summary(df: pd.DataFrame, algorithm: str, output_path: Path, device_order: List[str]) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(BATCH_MODES),
        figsize=(8, 5),
        sharey=True,
    )

    if len(BATCH_MODES) == 1:
        axes = np.array([axes])

    subset_algo = df[df["algorithm"] == algorithm]

    for col_idx, batch_mode in enumerate(BATCH_MODES):
        ax = axes[col_idx]
        subset = subset_algo[subset_algo["is_batch"] == batch_mode]
        if subset.empty:
            ax.set_axis_off()
            continue

        devices = device_order
        x = np.arange(len(devices))
        width = 0.22

        for i, delegate in enumerate(DELEGATES):
            partial = subset[subset["delegate"] == delegate].set_index("device_model")
            transfers = [
                partial.loc[device, "transfer_mean"] if device in partial.index else 0
                for device in devices
            ]
            computes = [
                partial.loc[device, "compute_mean"] if device in partial.index else 0
                for device in devices
            ]
            positions = x + (i - 1) * width
            ax.bar(
                positions,
                transfers,
                width=width,
                color=TRANSFER_COLOR,
            )
            ax.bar(
                positions,
                computes,
                width=width,
                bottom=transfers,
                color=DELEGATE_COLORS.get(delegate, "#4a5568"),
            )
            for pos, t, c in zip(positions, transfers, computes):
                total = t + c
                if total <= 0:
                    continue
                ax.text(
                    pos,
                    total + max(total * 0.05, 0.1),
                    f"{t:.1f}+{c:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(devices, rotation=10)
        mode_label = "Batch (10x)" if batch_mode else "Single"
        ax.set_title(f"{algorithm} — {mode_label}")
        ax.set_ylabel("Tempo (ms)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    legend_handles = [Patch(color=TRANSFER_COLOR, label="Transferência (GPU/NNAPI)")]
    legend_handles += [Patch(color=DELEGATE_COLORS[d], label=f"{d} - Processamento") for d in DELEGATES]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(f"Tempo de transferência vs processamento — {algorithm}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráfico único de transferência x processamento para MAD/FFT."
    )
    parser.add_argument("--csv", required=True, help="Caminho para benchmark_results.csv consolidado.")
    parser.add_argument(
        "--output",
        default="docs/charts/comparativo-30-11/transfer",
        help="Diretório de saída para os PNGs.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    data = prepare_data(csv_path)
    agg = aggregate_metrics(data)
    device_order = determine_device_order(list(data["device_model"].unique()))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for algo in ALGORITHMS:
        target = output_dir / f"transfer_compute_{algo.lower()}.png"
        plot_summary(agg, algo, target, device_order)
        print(f"Gráfico salvo em {target}")


if __name__ == "__main__":
    main()
