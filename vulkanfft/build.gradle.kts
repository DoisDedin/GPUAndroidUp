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


        // TensorFlow Lite Core (obrigatório)
        implementation(libs.tensorflow.lite)
        // Delegate GPU (opcional, recomendado se for usar aceleração na GPU)
        implementation(libs.tensorflow.lite.gpu)
        // Delegate NNAPI (opcional, usa aceleradores nativos)
        implementation(libs.tensorflow.lite.select.tf.ops)
        // Suporte (facilita manipulação de tensores, pré e pós-processamento)
        implementation(libs.tensorflow.lite.support)
        // Task Library (opcional, não essencial no seu caso, mas mantido)
        implementation(libs.tensorflow.lite.task.core)
    }
}
dependencies {
    implementation(libs.material)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
}
