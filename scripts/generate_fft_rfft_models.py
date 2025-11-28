"""
Gera modelos TensorFlow com RFFT (magnitudes) para vetores de 4096, 8192 e 16384,
converte para TFLite com Select TF Ops, valida frente ao NumPy e mede latência.

Uso:
    python3 scripts/generate_fft_rfft_models.py

Saída:
    - Modelos salvos em ./fft_models/fft_model_<N>.tflite
    - Logs de validação com erro máximo absoluto e tempo médio (TFLite CPU)
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

# Tamanhos oficiais usados no pipeline real
VECTOR_SIZES: Tuple[int, ...] = (4096, 8192, 16384)
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "fft_models")


def create_fft_model(input_length: int) -> tf.keras.Model:
    """Cria modelo Keras que calcula |RFFT| para um vetor 1-D."""
    inputs = tf.keras.Input(
        shape=(input_length,), dtype=tf.float32, name=f"input_{input_length}"
    )
    # Usamos Lambda para manter compatibilidade com Keras Functional.
    fft = tf.keras.layers.Lambda(lambda x: tf.signal.rfft(x))(inputs)
    magnitude = tf.keras.layers.Lambda(lambda x: tf.math.abs(x))(fft)
    return tf.keras.Model(inputs, magnitude, name=f"FFT_Model_{input_length}")


def convert_to_tflite(model: tf.keras.Model, input_length: int) -> bytes:
    """Converte o modelo para TFLite permitindo Select TF Ops."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT_DIR, f"fft_model_{input_length}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Modelo salvo: {path}  | tamanho: {len(tflite_model)/1024:.1f} KB")
    return tflite_model


def validate_model(tflite_model: bytes, input_length: int, runs: int = 5) -> None:
    """Valida saída e latência média do modelo TFLite."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Semente fixa para reprodutibilidade
    rng = np.random.default_rng(1234)
    signal = rng.random(input_length, dtype=np.float32)

    # TFLite inferências repetidas para estimar latência
    times_ms: list[float] = []
    for _ in range(runs):
        start = time.time()
        interpreter.set_tensor(input_details["index"], signal.reshape(1, -1))
        interpreter.invoke()
        times_ms.append((time.time() - start) * 1000.0)

    output_tflite = interpreter.get_tensor(output_details["index"])[0]
    numpy_ref = np.abs(np.fft.rfft(signal))
    max_error = float(np.max(np.abs(output_tflite - numpy_ref)))
    mean_error = float(np.mean(np.abs(output_tflite - numpy_ref)))

    print(f"\n===== VALIDAÇÃO PARA {input_length} PONTOS =====")
    print(f"Erro máximo absoluto: {max_error:.6e}")
    print(f"Erro médio absoluto: {mean_error:.6e}")
    print(f"Latência média TFLite (CPU): {np.mean(times_ms):.3f} ms  |  runs={runs}")
    print("=================================================")


def generate_all(sizes: Iterable[int]) -> None:
    for size in sizes:
        print(f"\n*** GERANDO MODELO PARA {size} PONTOS ***")
        model = create_fft_model(size)
        tflite_data = convert_to_tflite(model, size)
        validate_model(tflite_data, size)
    print("\n🏁 Processamento finalizado com sucesso.")


if __name__ == "__main__":
    generate_all(VECTOR_SIZES)
