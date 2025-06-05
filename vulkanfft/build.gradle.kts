import java.text.SimpleDateFormat
import java.util.Date

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.seuprojeto.vkfftlib" // ajuste conforme seu package
    compileSdk = 35

    defaultConfig {
        minSdk = 31
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Aqui definimos os parâmetros do NDK
        externalNativeBuild {
            cmake {
                cppFlags += "" // pode adicionar flags tipo -std=c++11 se precisar
            }
        }

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")

        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    dependencies {
        // Testes unitários (JUnit 4)
        testImplementation(libs.junit)

        // Testes instrumentados (AndroidJUnitRunner + JUnit 4 + assertEquals, etc)
        androidTestImplementation(libs.androidx.junit.v115)
        androidTestImplementation(libs.androidx.espresso.core.v351)

        implementation("com.google.ai.edge.litert:litert:1.3.0")
        implementation("io.github.google-ai-edge:litert-gpu-delegate-plugin:0.1.0")
    }
}
dependencies {
    implementation(libs.litert.gpu)
}


val libVersion = "1.1.1"

tasks.register("buildAarWithVersion") {
    group = "build"
    description = "Gera o .aar com nome contendo a versão."

    dependsOn("assembleRelease") // ✅ Depende da task padrão de build do módulo

    doLast {
        val buildType = "release"
        val moduleName = "vulkanfft"
        val originalAar = file("$buildDir/outputs/aar/$moduleName-$buildType.aar")
        val renamedAar = file("$buildDir/outputs/aar/$moduleName-v$libVersion.aar")

        if (originalAar.exists()) {
            originalAar.copyTo(renamedAar, overwrite = true)
            println("✅ Arquivo .aar gerado com sucesso:")
            println("📍 Caminho: ${renamedAar.absolutePath}")
            println("🔗 Link: file://${renamedAar.absolutePath}")
        } else {
            println("❌ Arquivo .aar não encontrado. Certifique-se de que a compilação foi realizada com sucesso.")
        }
    }
}