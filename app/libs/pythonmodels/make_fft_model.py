"""
Gera modelos TFLite que executam FFT real + aplicação de pesos dinâmicos por sensor.

Uso:
    python3 make_fft_model.py --lengths 4096 8192 16384

Cada modelo será salvo em `vulkanfft/src/main/assets/fft_model_<len>.tflite`. Para 4096
mantemos também o nome histórico `fft_model.tflite`.
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf

NUM_SENSORS = 10

# Evita warnings da Matplotlib/fontconfig em ambientes restritos
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)


def build_model(signal_length: int) -> bytes:
    freq_bins = signal_length // 2 + 1

    class WeightedFFT(tf.Module):
        def __init__(self):
            super().__init__()

        @tf.function(
            input_signature=(
                tf.TensorSpec(
                    shape=[NUM_SENSORS, signal_length],
                    dtype=tf.float32,
                    name="samples",
                ),
                tf.TensorSpec(
                    shape=[NUM_SENSORS, freq_bins],
                    dtype=tf.float32,
                    name="weights",
                ),
            )
        )
        def __call__(self, samples, weights):
            fft = tf.signal.rfft(samples)
            magnitude = tf.abs(fft)
            weighted = magnitude * weights
            real = tf.math.real(fft)
            imag = tf.math.imag(fft)
            stacked = tf.stack([real, imag, magnitude, weighted], axis=-1)
            return stacked

    model = WeightedFFT()
    concrete = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    return converter.convert()


def save_model(buffer: bytes, assets_dir: Path, signal_length: int):
    assets_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"fft_model_{signal_length}.tflite"
    (assets_dir / file_name).write_bytes(buffer)
    print(f"Modelo salvo em: {assets_dir / file_name}")
    if signal_length == 4096:
        (assets_dir / "fft_model.tflite").write_bytes(buffer)
        print(f"Modelo base atualizado em: {assets_dir / 'fft_model.tflite'}")


def main():
    parser = argparse.ArgumentParser(description="Gera modelos FFT TFLite para múltiplos comprimentos.")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Tamanhos de janela a exportar.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    assets_dir = project_root / "vulkanfft" / "src" / "main" / "assets"

    for length in args.lengths:
        buffer = build_model(length)
        save_model(buffer, assets_dir, length)


if __name__ == "__main__":
    main()
