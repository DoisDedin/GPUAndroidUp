package com.example.vulkanfft.external

import android.content.Context


object StatMadRunner {

    /**
     * Executa o modelo TFLite `mad_model.tflite` e retorna os valores estatísticos:
     * média, desvio padrão, mínimo e máximo da magnitude do vetor de aceleração,
     * calculados em blocos de 5 segundos.
     *
     * ## Requisitos da entrada:
     * - O input deve ser um `Array<IntArray>` com **shape [4][N]**, onde:
     *   - `input[0]`: timestamps em milissegundos (por exemplo, 0, 20, 40, ...)
     *   - `input[1]`: eixo X do acelerômetro (int)
     *   - `input[2]`: eixo Y do acelerômetro (int)
     *   - `input[3]`: eixo Z do acelerômetro (int)
     *
     * O modelo aceita apenas esse formato (int32) como entrada.
     *
     * @param context Contexto Android para carregar o modelo do assets.
     * @param input Vetor bidimensional com os dados de aceleração e timestamp.
     * @return FloatArray com 4 posições: [mean, stdDev, min, max]
     */
    suspend fun runMad(context: Context, input: Array<IntArray>): FloatArray {
        val sizeVector = input.getOrNull(0)?.size
            ?: throw IllegalArgumentException("Input inválido: input[0] (timestamps) está vazio ou ausente.")
        if (input.size != 4 || input.any { it.size != sizeVector }) {
            throw IllegalArgumentException("Input inválido: esperado shape [4][N], mas recebido [${input.size}][?]")
        }

        val processor = StatModelReduxProcessor(
            context = context,
            sizeVector = sizeVector
        )

        try {
            return processor.process(input)
        } finally {
            processor.close()
        }
    }
}