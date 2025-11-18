"""
Gera uma nova versão float32 do mad_model.tflite compatível com delegates GPU/NNAPI.
O modelo replica o cálculo usado no pipeline Kotlin:
- entrada [4, 4096] (timestamps + X/Y/Z) em float32
- converte cada amostra em magnitude
- retorna [mean, std, min, max]
"""

import os
from pathlib import Path
import tensorflow as tf

SAMPLE_LENGTH = 4096

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)


class MadFloatModel(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[SAMPLE_LENGTH, 3], dtype=tf.float32, name="axes"),
        ]
    )
    def __call__(self, axes):
        magnitudes = tf.sqrt(tf.reduce_sum(tf.square(axes), axis=1))
        mean = tf.reduce_mean(magnitudes)
        diff = magnitudes - mean
        variance = tf.reduce_mean(tf.square(diff))
        std = tf.sqrt(tf.maximum(variance, 1e-12))
        magnitudes_2d = tf.reshape(magnitudes, [1, SAMPLE_LENGTH])
        min_val = tf.reduce_min(magnitudes_2d, axis=1)[0]
        max_val = tf.reduce_max(magnitudes_2d, axis=1)[0]
        return tf.stack([mean, std, min_val, max_val], axis=0)


def build_tflite(output_path: Path):
    model = MadFloatModel()
    concrete = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_buffer = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_buffer)
    print(f"Modelo salvo em: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    assets_dir = project_root / "vulkanfft" / "src" / "main" / "assets"
    build_tflite(assets_dir / "mad_model.tflite")
