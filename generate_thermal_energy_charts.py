#!/usr/bin/env python3
"""
Gera gráficos de temperatura (bateria/CPU/GPU) e energia (labels) a partir de benchmark_results.csv consolidado.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

BASE_DIR = Path(__file__).parent
MPL_CACHE = BASE_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = {
    "CPU Kotlin": "#7f7f7f",
    "TFLite CPU": "#1f77b4",
    "TFLite GPU": "#2ca02c",
    "TFLite NNAPI": "#ff7f0e",
}
ENERGY_LABEL_ORDER = ["Baixa", "Baixa/Média", "Média", "Alta"]
ENERGY_TO_SCORE = {"Baixa": 1.0, "Baixa/Média": 1.5, "Média": 2.0, "Alta": 3.0}
SENSOR_PREFIXES = [
    ("battery", "Bateria"),
    ("cpu", "CPU"),
    ("gpu", "GPU"),
]


def ensure_device_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["device_model", "deviceModel", "deviceInfo", "model"]:
        if column in df.columns:
            labels = df[column].fillna("").astype(str).str.strip()
            if labels.ne("").any():
                df["device_model"] = labels.where(labels != "", "Dispositivo")
                break
    else:
        df["device_model"] = "Dispositivo"
    return df


def normalize_delegate(text: str) -> str:
    mapping = {
        "cpu kotlin": "CPU Kotlin",
        "kotlin": "CPU Kotlin",
        "cpu": "CPU Kotlin",
        "tflite cpu": "TFLite CPU",
        "tflite gpu": "TFLite GPU",
        "gpu": "TFLite GPU",
        "tflite nnapi": "TFLite NNAPI",
        "nnapi": "TFLite NNAPI",
    }
    key = str(text).strip().lower()
    return mapping.get(key, text.strip() or "CPU Kotlin")


def detect_algorithm(name: str) -> str:
    upper = str(name).upper()
    if "MAD" in upper:
        return "MAD"
    if "FFT" in upper:
        return "FFT"
    return "Outro"


def derive_vector_length(row: pd.Series) -> int:
    try:
        base = float(row.get("input_size", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0
    batch = int(row.get("batch_size", 1) or 1)
    if batch > 0:
        base /= batch
    if row.get("algorithm") == "FFT":
        sensors = 10
        base /= sensors
    return int(round(base)) if base > 0 else 0


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_energy_label(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    match = re.search(r"(Baixa/Média|Alta|Média|Baixa)", text, re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).capitalize()
    if label.lower() == "baixa/média":
        return "Baixa/Média"
    if label.lower() == "alta":
        return "Alta"
    if label.lower() == "média":
        return "Média"
    return "Baixa"


def energy_mode(series: pd.Series) -> Optional[str]:
    counts = series.dropna().value_counts()
    if counts.empty:
        return None
    max_count = counts.max()
    candidates = [label for label, value in counts.items() if value == max_count]
    candidates.sort(key=lambda label: ENERGY_LABEL_ORDER.index(label) if label in ENERGY_LABEL_ORDER else len(ENERGY_LABEL_ORDER))
    return candidates[0]


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_device_labels(df)
    df["delegate_norm"] = df["delegate"].apply(normalize_delegate)
    df["algorithm"] = df["test_name"].apply(detect_algorithm)
    df = df[df["algorithm"].isin(["MAD", "FFT"])]
    df["mode"] = df["batch_size"].fillna(1).astype(int).apply(lambda v: "Batch (x10)" if v > 1 else "Single")
    df["vector_length"] = df.apply(derive_vector_length, axis=1)
    df = df[df["vector_length"] > 0]
    df["energy_label"] = df["estimated_energy"].apply(parse_energy_label)
    df["energy_score"] = df["energy_label"].map(ENERGY_TO_SCORE)
    for prefix, _ in SENSOR_PREFIXES:
        for suffix in ("start", "end"):
            col = f"{prefix}_temp_{suffix}_c"
            if col not in df.columns:
                df[col] = pd.NA
    return df


def aggregate_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["device_model", "algorithm", "delegate_norm", "mode", "vector_length"]
    agg = (
        df.groupby(group_cols)
        .agg(
            battery_temp_start_c=("battery_temp_start_c", "mean"),
            battery_temp_end_c=("battery_temp_end_c", "mean"),
            cpu_temp_start_c=("cpu_temp_start_c", "mean"),
            cpu_temp_end_c=("cpu_temp_end_c", "mean"),
            gpu_temp_start_c=("gpu_temp_start_c", "mean"),
            gpu_temp_end_c=("gpu_temp_end_c", "mean"),
        )
        .reset_index()
    )
    for prefix, _ in SENSOR_PREFIXES:
        agg[f"{prefix}_temp_delta_c"] = agg[f"{prefix}_temp_end_c"] - agg[f"{prefix}_temp_start_c"]
    return agg


def plot_device_temperature_panels(
    agg: pd.DataFrame,
    device: str,
    prefix: str,
    label: str,
    output_dir: Path,
) -> None:
    subset = agg[agg["device_model"] == device].copy()
    start_col = f"{prefix}_temp_start_c"
    end_col = f"{prefix}_temp_end_c"
    if subset[[start_col, end_col]].dropna(how="all").empty:
        return
    melted = subset.melt(
        id_vars=["algorithm", "delegate_norm", "mode", "vector_length"],
        value_vars=[start_col, end_col],
        var_name="phase",
        value_name="temp_c",
    ).dropna(subset=["temp_c"])
    if melted.empty:
        return
    phase_map = {
        start_col: "Início",
        end_col: "Fim",
    }
    melted["phase_label"] = melted["phase"].map(phase_map)
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=melted,
        x="vector_length",
        y="temp_c",
        hue="delegate_norm",
        style="phase_label",
        col="mode",
        row="algorithm",
        kind="line",
        palette=PALETTE,
        marker="o",
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Tamanho do vetor (amostras)", "Temperatura (°C)")
    g.add_legend(title="Delegate / Medição")
    g.figure.suptitle(f"{device} — {label} (Início × Fim)", y=1.02, fontsize=13)
    filename = output_dir / f"{slugify(device)}_{prefix}_start_end.png"
    g.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_delta_summary(agg: pd.DataFrame, prefix: str, label: str, output_dir: Path) -> None:
    col = f"{prefix}_temp_delta_c"
    if col not in agg.columns:
        return
    summary = (
        agg.groupby(["device_model", "algorithm", "mode", "vector_length"])[col]
        .mean()
        .reset_index()
    )
    summary = summary.dropna(subset=[col])
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=summary,
        x="vector_length",
        y=col,
        hue="device_model",
        style="mode",
        col="algorithm",
        kind="line",
        marker="o",
    )
    for ax in g.axes.flatten():
        ax.axhline(0, color="#718096", linestyle="--", linewidth=0.8)
    g.set_axis_labels("Tamanho do vetor (amostras)", "Δ temperatura (°C)")
    g.figure.suptitle(f"Variação média de temperatura — {label}", y=1.02, fontsize=13)
    filename = output_dir / f"deltas_{prefix}.png"
    g.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_energy_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    energy_df = df.dropna(subset=["energy_label"]).copy()
    if energy_df.empty:
        return

    summary = (
        energy_df.groupby(["device_model", "algorithm", "delegate_norm"])
        .agg(energy_label=("energy_label", energy_mode))
        .reset_index()
    )
    summary["energy_score"] = summary["energy_label"].map(ENERGY_TO_SCORE)
    for algorithm in sorted(summary["algorithm"].unique()):
        subset = summary[summary["algorithm"] == algorithm]
        pivot_values = subset.pivot(
            index="device_model",
            columns="delegate_norm",
            values="energy_score",
        )
        if pivot_values.empty:
            continue
        labels = subset.pivot(
            index="device_model",
            columns="delegate_norm",
            values="energy_label",
        )
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6, 0.8 * len(pivot_values.index) + 2))
        ax = sns.heatmap(
            pivot_values,
            annot=labels,
            fmt="",
            cmap="YlGnBu",
            vmin=min(ENERGY_TO_SCORE.values()),
            vmax=max(ENERGY_TO_SCORE.values()),
            cbar_kws={"label": "Intensidade (1=Baixa, 3=Alta)"},
        )
        ax.set_xlabel("Delegate")
        ax.set_ylabel("Dispositivo")
        ax.set_title(f"Etiqueta energética predominante — {algorithm}")
        filename = output_dir / f"energy_{algorithm.lower()}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos térmicos e energéticos dos benchmarks.")
    parser.add_argument("--csv", required=True, help="Caminho para o CSV consolidado.")
    parser.add_argument(
        "--output",
        default="docs/charts/comparativo-09-12/thermal_energy",
        help="Diretório base para salvar os gráficos.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    thermal_dir = output_dir / "thermal"
    energy_dir = output_dir / "energy"
    thermal_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(csv_path)
    temp_agg = aggregate_temperatures(df)
    devices = sorted(temp_agg["device_model"].unique())
    for prefix, label in SENSOR_PREFIXES:
        for device in devices:
            plot_device_temperature_panels(temp_agg, device, prefix, label, thermal_dir)
        plot_delta_summary(temp_agg, prefix, label, thermal_dir)

    plot_energy_heatmaps(df, energy_dir)
    print(f"Gráficos térmicos salvos em {thermal_dir}")
    print(f"Gráficos energéticos salvos em {energy_dir}")


if __name__ == "__main__":
    main()
