"""
Gera um modelo TFLite simples que executa FFT real + aplicação de pesos dinâmicos
por sensor. O objetivo é reproduzir no TensorFlow Lite a mesma lógica do
FftCpuProcessor para permitir comparativos diretos.

Uso:
    python3 make_fft_model.py

O arquivo `fft_model.tflite` será salvo em `vulkanfft/src/main/assets`.
"""

import os
from pathlib import Path
import tensorflow as tf

NUM_SENSORS = 10
SIGNAL_LENGTH = 4096  # janela grande baseada nos vetores do MAD
FREQ_BINS = SIGNAL_LENGTH // 2 + 1

# Evita warnings da Matplotlib/fontconfig em ambientes restringidos
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)


class WeightedFFT(tf.Module):
    def __init__(self):
        super().__init__()
        self.signal_length = SIGNAL_LENGTH
        self.freq_bins = FREQ_BINS

    @tf.function(
        input_signature=(
            tf.TensorSpec(
                shape=[NUM_SENSORS, SIGNAL_LENGTH],
                dtype=tf.float32,
                name="samples",
            ),
            tf.TensorSpec(
                shape=[NUM_SENSORS, FREQ_BINS],
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


def build_tflite(output_path: Path):
    model = WeightedFFT()
    concrete = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete],
        model,
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    tflite_buffer = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_buffer)
    print(f"Modelo salvo em: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    assets_dir = project_root / "vulkanfft" / "src" / "main" / "assets"
    build_tflite(assets_dir / "fft_model.tflite")
