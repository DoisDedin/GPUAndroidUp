#!/usr/bin/env python3
"""
Gera gráficos de benchmark a partir de benchmark_results.csv.

- batch=4 significa média de 4 execuções sequenciais para estabilidade estatística,
  não paralelismo ou aumento de carga.
- Há gráficos separados com CPU puro vs TFLite para evidenciar a diferença de ordem
  de grandeza, e gráficos apenas com delegates TFLite para comparação real entre
  aceleradores.
- Speedup (CPU Kotlin puro / delegate) é usado como métrica clara de ganho relativo.
"""

from __future__ import annotations

import argparse
import shutil
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).parent
MPL_CACHE = BASE_DIR / ".matplotlib"
# Mantém cache local do Matplotlib para evitar tentativas de escrita em ~/.matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "CPU Kotlin": "#7f7f7f",  # cinza
    "TFLite CPU": "#1f77b4",  # azul
    "TFLite GPU": "#2ca02c",  # verde
    "TFLite NNAPI": "#ff7f0e",  # laranja
}

TITLE_FONTSIZE = 12
TITLE_PAD = 14

# Texto fixo que aparece nos gráficos
SUBTITLE_BATCH_INFO = "Média das execuções sequenciais (batch ≠ paralelismo)"


def format_vector_label(length: int) -> str:
    if length >= 1024:
        if length % 1024 == 0:
            return f"{int(length // 1024)}k"
        return f"{length / 1024:.1f}k"
    return str(int(length))


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado em {csv_path}")
    df = pd.read_csv(csv_path)
    # padroniza nomes de colunas para evitar problemas com espaços
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_device_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "device_model" in df.columns:
        df["device_model"] = df["device_model"].astype(str).str.strip()
        return df

    if "deviceInfo" in df.columns:
        base_model = df["deviceInfo"].fillna("Dispositivo").astype(str).str.strip()
    elif "model" in df.columns:
        base_model = df["model"].fillna("Dispositivo").astype(str).str.strip()
    else:
        base_model = pd.Series(["Dispositivo"] * len(df))

    df["device_model"] = base_model.where(base_model != "", "Dispositivo")
    return df


def maybe_add_fake_devices(df: pd.DataFrame) -> pd.DataFrame:
    """Se houver só um dispositivo, duplica linhas com labels fictícios para testar gráficos multi-device."""
    unique_devices = df["device_model"].nunique()
    if unique_devices > 1:
        return df

    clones: List[pd.DataFrame] = []
    for fake in ["S21_fake", "MotoG04s_fake", "MotoG84_fake"]:
        clone = df.copy()
        clone["device_model"] = fake
        clones.append(clone)
    combined = pd.concat([df] + clones, ignore_index=True)
    print(
        "Apenas um dispositivo detectado; criando cópias fictícias "
        "S21_fake, MotoG04s_fake, MotoG84_fake para testes de gráficos multi-dispositivo."
    )
    return combined


def normalize_delegate(delegate: str) -> str:
    if not isinstance(delegate, str):
        return "CPU Kotlin"
    name = delegate.strip().lower()
    mapping = {
        "cpu kotlin": "CPU Kotlin",
        "cpu": "CPU Kotlin",
        "kotlin cpu": "CPU Kotlin",
        "tflite cpu": "TFLite CPU",
        "tflite gpu": "TFLite GPU",
        "gpu": "TFLite GPU",
        "tflite nnapi": "TFLite NNAPI",
        "nnapi": "TFLite NNAPI",
    }
    return mapping.get(name, delegate.strip() or "CPU Kotlin")


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delegate_norm"] = df["delegate"].apply(normalize_delegate)

    def detect_algo(text: str) -> str:
        t = str(text).upper()
        if "MAD" in t:
            return "MAD"
        if "FFT" in t:
            return "FFT"
        return "Outro"

    df["algoritmo"] = df["test_name"].apply(detect_algo)
    return df


def derive_vector_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def normalize_row(row: pd.Series) -> int:
        try:
            base = float(row.get("input_size", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

        batch = int(row.get("batch_size", 1) or 1)
        if batch > 0:
            base /= batch

        algoritmo = row.get("algoritmo")
        if isinstance(algoritmo, str) and algoritmo.upper() == "FFT":
            sensors = 10
            if sensors > 0:
                base /= sensors

        if base <= 0:
            return 0
        return int(round(base))

    df["tamanho_do_vetor"] = df.apply(normalize_row, axis=1)
    return df


def ensure_timing_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["duration_ms", "transfer_ms", "compute_ms"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["device_model", "algoritmo", "delegate_norm", "tamanho_do_vetor"])
        .agg(
            duration_mean=("duration_ms", "mean"),
            duration_std=("duration_ms", "std"),
            transfer_mean=("transfer_ms", "mean"),
            compute_mean=("compute_ms", "mean"),
            n=("duration_ms", "count"),
        )
        .reset_index()
    )

    # se não houver compute_ms, usamos duration - transfer como aproximação
    grouped["compute_mean"] = grouped.apply(
        lambda row: row.compute_mean
        if row.compute_mean != 0
        else max(row.duration_mean - row.transfer_mean, 0),
        axis=1,
    )
    return grouped


def compute_speedup(agg: pd.DataFrame) -> pd.DataFrame:
    cpu = agg[agg["delegate_norm"] == "CPU Kotlin"][
        ["device_model", "algoritmo", "tamanho_do_vetor", "duration_mean"]
    ].rename(columns={"duration_mean": "cpu_duration"})

    merged = agg.merge(
        cpu,
        on=["device_model", "algoritmo", "tamanho_do_vetor"],
        how="left",
    )
    merged["speedup"] = merged["cpu_duration"] / merged["duration_mean"]
    return merged


def prepare_output_dirs(base_dir: Path, devices: Iterable[str]) -> None:
    if base_dir.exists():
        shutil.rmtree(base_dir)
    for device in devices:
        for sub in ["global", "tflite", "speedup"]:
            (base_dir / device / sub).mkdir(parents=True, exist_ok=True)
    (base_dir / "summary").mkdir(parents=True, exist_ok=True)


def add_common_formatting(ax, ylabel: str, xlabel: str = "Tamanho do vetor (amostras)") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def apply_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)


def finalize_figure(fig: plt.Figure) -> None:
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])


def apply_vector_ticks(ax, values: Iterable[int]) -> None:
    ordered = sorted(values)
    ax.set_xticks(ordered)
    ax.set_xticklabels([format_vector_label(v) for v in ordered])


def plot_global_comparison(
    data: pd.DataFrame, device: str, algoritmo: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    subset = data[
        (data["device_model"] == device)
        & (data["algoritmo"] == algoritmo)
        & (data["delegate_norm"].isin(PALETTE.keys()))
    ].sort_values("tamanho_do_vetor")

    sns.lineplot(
        data=subset,
        x="tamanho_do_vetor",
        y="duration_mean",
        hue="delegate_norm",
        palette=PALETTE,
        marker="o",
        ax=ax,
    )
    for _, row in subset.iterrows():
        if pd.notna(row["duration_std"]):
            ax.errorbar(
                row["tamanho_do_vetor"],
                row["duration_mean"],
                yerr=row["duration_std"],
                fmt="none",
                ecolor=PALETTE.get(row["delegate_norm"], "black"),
                alpha=0.4,
            )

    title = f"{device} · {algoritmo} – CPU vs Delegates"
    apply_title(ax, title)
    add_common_formatting(ax, "Tempo (ms)")
    apply_vector_ticks(ax, subset["tamanho_do_vetor"].unique())
    fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
    finalize_figure(fig)
    fig.savefig(out_dir / f"global_{algoritmo}.png", dpi=300)
    plt.close(fig)


def plot_tflite_comparison(
    data: pd.DataFrame, device: str, algoritmo: str, out_dir: Path
) -> None:
    tflite_delegates = ["TFLite CPU", "TFLite GPU", "TFLite NNAPI"]
    subset = data[
        (data["device_model"] == device)
        & (data["algoritmo"] == algoritmo)
        & (data["delegate_norm"].isin(tflite_delegates))
    ].sort_values("tamanho_do_vetor")

    # Tempo total
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=subset,
        x="tamanho_do_vetor",
        y="duration_mean",
        hue="delegate_norm",
        palette=PALETTE,
        marker="o",
        ax=ax,
    )
    apply_title(ax, f"{device} · {algoritmo} – Delegates TFLite (tempo total)")
    add_common_formatting(ax, "Tempo (ms)")
    apply_vector_ticks(ax, subset["tamanho_do_vetor"].unique())
    fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
    finalize_figure(fig)
    fig.savefig(out_dir / f"tflite_total_{algoritmo}.png", dpi=300)
    plt.close(fig)

    # Transferência vs processamento (barras empilhadas)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    x_positions = range(len(subset["tamanho_do_vetor"].unique()))
    size_order = sorted(subset["tamanho_do_vetor"].unique())

    for i, delegate in enumerate(tflite_delegates):
        partial = subset[subset["delegate_norm"] == delegate]
        partial = partial.set_index("tamanho_do_vetor").reindex(size_order)
        transfer = partial["transfer_mean"].fillna(0.0)
        compute = partial["compute_mean"].fillna(0.0)
        if delegate == "TFLite CPU":
            transfer = transfer * 0.0  # CPU sem custo de transferência

        positions = [x + i * width for x in x_positions]
        bars1 = ax.bar(
            positions,
            transfer,
            width=width,
            color=PALETTE[delegate],
            alpha=0.55,
            label=f"{delegate} - Transferência" if i == 0 else None,
        )
        bars2 = ax.bar(
            positions,
            compute,
            bottom=transfer,
            width=width,
            color=PALETTE[delegate],
            label=f"{delegate} - Processamento",
        )

        for pos, t, c in zip(positions, transfer, compute):
            total = t + c
            if total > 0:
                overhead = (t / total) * 100
                ax.text(
                    pos,
                    total + max(total * 0.02, 0.5),
                    f"{overhead:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    xtick_positions = [x + width for x in x_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([format_vector_label(s) for s in size_order])
    apply_title(
        ax, f"{device} · {algoritmo} – Transferência vs Processamento"
    )
    add_common_formatting(ax, "Tempo (ms)")
    fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
    ax.legend()
    finalize_figure(fig)
    fig.savefig(out_dir / f"tflite_transfer_{algoritmo}.png", dpi=300)
    plt.close(fig)


def plot_speedup(
    speedup_df: pd.DataFrame, device: str, algoritmo: str, out_dir: Path
) -> None:
    subset = speedup_df[
        (speedup_df["device_model"] == device)
        & (speedup_df["algoritmo"] == algoritmo)
        & (speedup_df["delegate_norm"].isin(["TFLite CPU", "TFLite GPU", "TFLite NNAPI"]))
    ].sort_values("tamanho_do_vetor")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=subset,
        x="tamanho_do_vetor",
        y="speedup",
        hue="delegate_norm",
        palette=PALETTE,
        marker="o",
        ax=ax,
    )
    apply_title(ax, f"{device} · {algoritmo} – Speedup vs CPU Kotlin")
    add_common_formatting(ax, "Speedup (×)")
    apply_vector_ticks(ax, subset["tamanho_do_vetor"].unique())
    fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
    finalize_figure(fig)
    fig.savefig(out_dir / f"speedup_{algoritmo}.png", dpi=300)
    plt.close(fig)


def print_device_summary(agg: pd.DataFrame, speedup_df: pd.DataFrame, device: str) -> None:
    durations = (
        agg[agg["device_model"] == device]
        .pivot_table(
            index="tamanho_do_vetor",
            columns="delegate_norm",
            values="duration_mean",
        )
        .sort_index()
    )
    speedups = (
        speedup_df[
            (speedup_df["device_model"] == device)
            & (speedup_df["delegate_norm"] != "CPU Kotlin")
        ]
        .pivot_table(
            index="tamanho_do_vetor",
            columns="delegate_norm",
            values="speedup",
        )
        .sort_index()
    )

    combined = pd.DataFrame(index=durations.index)
    for delegate in durations.columns:
        combined[f"{delegate} (ms)"] = durations[delegate]
        if delegate != "CPU Kotlin" and delegate in speedups.columns:
            combined[f"{delegate} speedup"] = speedups[delegate]

    print(f"\nResumo para {device}")
    print(combined.round(3).fillna("-"))


def generate_device_plots(
    agg: pd.DataFrame, speedup_df: pd.DataFrame, base_dir: Path, device: str
) -> None:
    for algoritmo in ["MAD", "FFT"]:
        plot_global_comparison(agg, device, algoritmo, base_dir / device / "global")
        plot_tflite_comparison(agg, device, algoritmo, base_dir / device / "tflite")
        plot_speedup(speedup_df, device, algoritmo, base_dir / device / "speedup")


def generate_summary_plots(
    agg: pd.DataFrame, speedup_df: pd.DataFrame, base_dir: Path
) -> None:
    summary_dir = base_dir / "summary"
    delegates = ["TFLite CPU", "TFLite GPU", "TFLite NNAPI"]

    for algoritmo in ["MAD", "FFT"]:
        for delegate in delegates:
            for size in sorted(agg["tamanho_do_vetor"].unique()):
                subset = agg[
                    (agg["algoritmo"] == algoritmo)
                    & (agg["delegate_norm"] == delegate)
                    & (agg["tamanho_do_vetor"] == size)
                ]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.barplot(
                    data=subset,
                    x="device_model",
                    y="duration_mean",
                    color=PALETTE[delegate],
                    ax=ax,
                )
                apply_title(
                    ax,
                    f"{algoritmo} – {delegate} – vetor {format_vector_label(size)}",
                )
                add_common_formatting(ax, "Tempo (ms)", "Dispositivo")
                fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
                finalize_figure(fig)
                fig.savefig(
                    summary_dir
                    / f"tempo_{algoritmo}_{delegate}_{size}.png",
                    dpi=300,
                )
                plt.close(fig)

            subset_speed = speedup_df[
                (speedup_df["algoritmo"] == algoritmo)
                & (speedup_df["delegate_norm"] == delegate)
                & (speedup_df["delegate_norm"] != "CPU Kotlin")
            ]
            if subset_speed.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(
                data=subset_speed,
                x="device_model",
                y="speedup",
                hue="tamanho_do_vetor",
                ax=ax,
            )
            apply_title(
                ax, f"{algoritmo} – {delegate} – Speedup vs CPU Kotlin"
            )
            add_common_formatting(ax, "Speedup (×)", "Dispositivo")
            fig.text(0.5, 0.005, SUBTITLE_BATCH_INFO, ha="center", fontsize=9)
            finalize_figure(fig)
            fig.savefig(
                summary_dir / f"speedup_{algoritmo}_{delegate}.png",
                dpi=300,
            )
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de benchmarks por dispositivo e comparação entre dispositivos."
    )
    parser.add_argument(
        "--csv",
        default="benchmark_results.csv",
        help="Caminho para benchmark_results.csv",
    )
    parser.add_argument(
        "--output",
        default="charts",
        help="Diretório raiz para salvar gráficos.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)

    sns.set_theme(style="whitegrid")

    raw = load_data(csv_path)
    raw = extract_device_model(raw)
    raw = maybe_add_fake_devices(raw)
    raw = add_normalized_columns(raw)
    raw = derive_vector_size(raw)
    raw = ensure_timing_columns(raw)

    agg = aggregate_metrics(raw)
    speedup_df = compute_speedup(agg)

    prepare_output_dirs(output_dir, agg["device_model"].unique())

    for device in agg["device_model"].unique():
        generate_device_plots(agg, speedup_df, output_dir, device)
        print_device_summary(agg, speedup_df, device)
        print(f"Gráficos gerados para dispositivo {device} em {output_dir}/{device}/...")

    generate_summary_plots(agg, speedup_df, output_dir)
    print(f"Gráficos de comparação entre dispositivos gerados em {output_dir}/summary/...")


if __name__ == "__main__":
    main()
