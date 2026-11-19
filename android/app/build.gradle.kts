plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.sentinel.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sentinel.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

// The parity tests read the exported models straight off disk with File(), so
// Gradle cannot see them as inputs and calls the task up to date after
// `python export_android.py` has changed them — a green suite that never ran.
// Declaring the directory is what makes a re-export re-run the comparison.
tasks.withType<Test>().configureEach {
    inputs.dir("src/main/assets")
        .withPropertyName("modelAssets")
        .withPathSensitivity(PathSensitivity.RELATIVE)
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    // Pull-to-refresh on the message list. The platform widget rather than
    // touch handling of our own: it is the gesture users already know, and
    // it is one dependency against thirty lines that get the fling wrong.
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")

    // Room keeps the scanned messages and their verdicts across restarts.
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")

    testImplementation("junit:junit:4.13.2")
    // A real JSONObject on the unit-test classpath; android.jar's is a stub that
    // throws. Must come before the mockable android.jar, which it does.
    testImplementation("org.json:json:20240303")
}
