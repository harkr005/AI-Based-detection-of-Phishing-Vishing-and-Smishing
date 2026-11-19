plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    // KSP's version is pinned to Kotlin's: the suffix moves, the prefix must
    // match org.jetbrains.kotlin.android above or the build refuses to start.
    id("com.google.devtools.ksp") version "1.9.24-1.0.20" apply false
}
