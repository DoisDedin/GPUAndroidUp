//
// Created by Joao Vitor Cotta on 14/05/25.
//
#include <jni.h>
#include <vector>
#include <android/log.h>

#define LOG_TAG "VkFFTBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_com_seuprojeto_vkfftlib_VulkanBridge_runVkFFT(JNIEnv *env, jobject, jdoubleArray inputArray) {
    jsize len = env->GetArrayLength(inputArray);
    std::vector<double> input(len);
    env->GetDoubleArrayRegion(inputArray, 0, len, input.data());

    for (int i = 0; i < len; ++i) {
        input[i] *= 2.0; // simula processamento
    }

    jdoubleArray result = env->NewDoubleArray(len);
    env->SetDoubleArrayRegion(result, 0, len, input.data());
    return result;
}