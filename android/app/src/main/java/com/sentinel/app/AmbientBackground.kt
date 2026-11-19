package com.sentinel.app

import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RadialGradient
import android.graphics.Rect
import android.graphics.Shader
import android.graphics.drawable.Drawable

/**
 * The red-on-black wash behind the app — the website's own, ported stop for
 * stop from `.k-ambient` in `web/app/globals.css` so the two cannot drift.
 * It replaced a graph-paper grid on 2026-09-23.
 *
 * ponytail: drawn, not an asset. The web's reference builds this out of three
 * blurred PNGs and our CSS already replaced them with gradients; a bitmap here
 * would be worse still, since it would ship at one density and band on every
 * other. Four shaders, no files, sharp on any screen.
 */
class AmbientBackground : Drawable() {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    // Rebuilt on resize rather than per frame: the shaders are bound to pixel
    // coordinates, so they are only wrong when the bounds change.
    private var layers: List<Shader> = emptyList()

    override fun onBoundsChange(bounds: Rect) {
        if (bounds.isEmpty) return
        layers = listOf(
            // linear-gradient(180deg, #1a0806 0%, #0d0403 42%, #060202 100%)
            LinearGradient(
                0f, bounds.top.toFloat(), 0f, bounds.bottom.toFloat(),
                intArrayOf(0xff1a0806.toInt(), 0xff0d0403.toInt(), 0xff060202.toInt()),
                floatArrayOf(0f, 0.42f, 1f),
                Shader.TileMode.CLAMP,
            ),
            // The three glows, painted bottom-up — CSS lists background layers
            // topmost first, and a canvas draws in the opposite order.
            glow(bounds, x = 0.50f, y = 1.06f, rx = 0.94f, ry = 0.65f, color = 0x24ff3c1c, fade = 0.62f),
            glow(bounds, x = 0.96f, y = 0.02f, rx = 0.85f, ry = 0.65f, color = 0x33ff1e00, fade = 0.60f),
            glow(bounds, x = 0.08f, y = -0.12f, rx = 1.15f, ry = 0.81f, color = 0x57ff3c1c, fade = 0.62f),
        )
    }

    /**
     * One `radial-gradient(RX RY at X% Y%, color, transparent FADE%)`.
     *
     * Both radii are fractions of the WIDTH, never one per axis: the site sizes
     * them in absolute pixels, so scaling each to its own axis would stretch
     * the glows to the phone's aspect ratio and lose the shape entirely. These
     * are the site's pixel sizes over the ~960px they were drawn against.
     */
    private fun glow(
        bounds: Rect,
        x: Float,
        y: Float,
        rx: Float,
        ry: Float,
        color: Int,
        fade: Float,
    ): Shader {
        val width = bounds.width().toFloat()
        val cx = bounds.left + x * width
        val cy = bounds.top + y * bounds.height()
        val shader = RadialGradient(
            cx, cy, rx * width,
            // `transparent` in CSS is the same hue at zero alpha. Naming it as
            // 0x00000000 instead would fade every glow through grey.
            intArrayOf(color, color and 0x00ffffff),
            floatArrayOf(0f, fade),
            Shader.TileMode.CLAMP,
        )
        // RadialGradient is circular; the matrix is what makes it the ellipse.
        shader.setLocalMatrix(Matrix().apply { setScale(1f, ry / rx, cx, cy) })
        return shader
    }

    override fun draw(canvas: Canvas) {
        // Before the first layout there is nothing to interpolate between, and
        // the deepest stop is the app's own background colour.
        if (layers.isEmpty()) return canvas.drawColor(0xff060202.toInt())
        for (shader in layers) {
            paint.shader = shader
            canvas.drawRect(bounds, paint)
        }
    }

    override fun setAlpha(alpha: Int) { paint.alpha = alpha }
    override fun setColorFilter(filter: ColorFilter?) { paint.colorFilter = filter }

    // The first layer covers the bounds, so nothing behind this can show.
    override fun getOpacity() = PixelFormat.OPAQUE
}
