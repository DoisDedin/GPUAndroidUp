"""
Gera versões float32 do `mad_model.tflite` compatíveis com delegates GPU/NNAPI.
O modelo replica o cálculo usado no pipeline Kotlin e pode ser emitido para
múltiplos comprimentos de vetor (4k, 8k, 16k, etc.).
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)


def build_model(sample_length: int) -> bytes:
    class MadFloatModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[sample_length, 3],
                    dtype=tf.float32,
                    name="axes",
                ),
            ]
        )
        def __call__(self, axes):
            magnitudes = tf.sqrt(tf.reduce_sum(tf.square(axes), axis=1))
            mean = tf.reduce_mean(magnitudes)
            diff = magnitudes - mean
            variance = tf.reduce_mean(tf.square(diff))
            std = tf.sqrt(tf.maximum(variance, 1e-12))
            magnitudes_2d = tf.reshape(magnitudes, [1, sample_length])
            min_val = tf.reduce_min(magnitudes_2d, axis=1)[0]
            max_val = tf.reduce_max(magnitudes_2d, axis=1)[0]
            return tf.stack([mean, std, min_val, max_val], axis=0)

    model = MadFloatModel()
    concrete = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def save_model(buffer: bytes, assets_dir: Path, sample_length: int):
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"mad_model_{sample_length}.tflite"
    (assets_dir / target_name).write_bytes(buffer)
    print(f"Modelo salvo em: {assets_dir / target_name}")
    if sample_length == 4096:
        # Mantém compatibilidade com o nome legacy.
        (assets_dir / "mad_model.tflite").write_bytes(buffer)
        print(f"Modelo base atualizado em: {assets_dir / 'mad_model.tflite'}")


def main():
    parser = argparse.ArgumentParser(description="Gera modelos MAD em float32.")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Comprimentos de vetor a serem exportados.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    assets_dir = project_root / "vulkanfft" / "src" / "main" / "assets"

    for length in args.lengths:
        buffer = build_model(length)
        save_model(buffer, assets_dir, length)


if __name__ == "__main__":
    main()
