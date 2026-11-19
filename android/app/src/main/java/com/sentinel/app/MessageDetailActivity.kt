package com.sentinel.app

import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.ColorUtils
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

const val EXTRA_MESSAGE_ID = "message_id"

class MessageDetailActivity : AppCompatActivity() {

    private lateinit var inflater: LayoutInflater

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_message_detail)
        inflater = LayoutInflater.from(this)

        findViewById<MaterialToolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }

        val bandView = findViewById<TextView>(R.id.detailBand)
        val message = MessageStore.get(intent.getLongExtra(EXTRA_MESSAGE_ID, -1L))
        if (message == null) {
            bandView.text = "Message not found"
            return
        }

        val sentAt = SimpleDateFormat("d MMM yyyy, h:mm a", Locale.US).format(Date(message.sentAt))
        findViewById<TextView>(R.id.detailSender).text = "From ${message.sender} · $sentAt"
        findViewById<TextView>(R.id.detailBody).text = message.body

        val result = message.result
        if (result == null) {
            bandView.text = if (message.failed) {
                "This message could not be checked — the model server did not " +
                    "answer. Treat it with the same caution you would any " +
                    "unexpected message."
            } else {
                "Still checking this message…"
            }
            bandView.setTextColor(color(R.color.text_secondary))
            findViewById<TextView>(R.id.pointsCaption).visibility = View.GONE
            findViewById<View>(R.id.technicalToggle).visibility = View.GONE
            return
        }

        val verdict = result.optString("verdict")
        // The class label is not the band: an unsolicited promo is spam AND
        // risk 0.009. rowBadge is where the two are allowed to meet, so that a
        // SAFE promo reads as Spam here exactly as it does on the list.
        val label = result.optString("label")
        val band = rowBadge(verdict, categoryFor(label))
        val bandColor = color(rowBadgeColorRes(verdict, categoryFor(label)))

        // What it means, before any number: the same headline the dashboard shows.
        bandView.text = band.say
        bandView.setTextColor(bandColor)
        findViewById<MaterialCardView>(R.id.bandCard).strokeColor = bandColor

        findViewById<TextView>(R.id.detailLabelPill).apply {
            text = label
            setTextColor(bandColor)
            backgroundTintList =
                ColorStateList.valueOf(ColorUtils.setAlphaComponent(bandColor, 38))
        }
        findViewById<TextView>(R.id.detailKind).text = kindSentence(label)

        renderPoints(plainPoints(result))
        renderTechnical(result, band.label, bandColor)
    }

    private fun renderPoints(points: List<Point>) {
        val container = findViewById<LinearLayout>(R.id.pointsContainer)
        for (point in points) {
            val row = inflater.inflate(R.layout.item_point, container, false)
            val tint = when (point.kind) {
                PointKind.GOOD -> color(R.color.band_safe)
                PointKind.BAD -> color(R.color.band_phishing)
                PointKind.NOTE -> color(R.color.text_secondary)
            }
            row.findViewById<TextView>(R.id.pointMarker).apply {
                text = when (point.kind) {
                    PointKind.GOOD -> "✓"
                    PointKind.BAD -> "!"
                    PointKind.NOTE -> "?"
                }
                setTextColor(tint)
            }
            row.findViewById<TextView>(R.id.pointText).apply {
                text = point.text
                setTextColor(
                    if (point.kind == PointKind.NOTE) color(R.color.text_secondary)
                    else color(R.color.text_primary)
                )
            }
            if (point.terms.isNotEmpty()) {
                val chips = row.findViewById<ChipGroup>(R.id.pointChips)
                chips.visibility = View.VISIBLE
                for (term in point.terms) {
                    val chip = inflater.inflate(R.layout.item_term_chip, chips, false) as Chip
                    chip.text = term
                    chips.addView(chip)
                }
            }
            container.addView(row)
        }
    }

    private fun renderTechnical(result: JSONObject, bandLabel: String, bandColor: Int) {
        val toggle = findViewById<View>(R.id.technicalToggle)
        val content = findViewById<View>(R.id.technicalContent)
        val chevron = findViewById<TextView>(R.id.technicalChevron)
        toggle.setOnClickListener {
            val open = content.visibility != View.VISIBLE
            content.visibility = if (open) View.VISIBLE else View.GONE
            chevron.text = if (open) "▴" else "▾"
        }

        val risk = result.optDouble("risk", 0.0)
        findViewById<TextView>(R.id.detailRisk).apply {
            text = percent(risk)
            setTextColor(bandColor)
        }
        findViewById<TextView>(R.id.detailRiskCaption).text =
            "risk · bands 35% / 70% · verdict $bandLabel"

        val parts = result.optJSONObject("parts")
        val weights = result.optJSONObject("weights")
        if (parts != null) {
            val container = findViewById<LinearLayout>(R.id.partsContainer)
            for (key in parts.keys()) {
                val weight = weights?.optDouble(key)
                val suffix = if (weight == null || weight.isNaN()) ""
                else "   weight ${percent(weight)}"
                addBar(container, channelLabel(key) + suffix, parts.optDouble(key))
            }
        }

        // Entropy stays a plain number: it is an information measure, not a share.
        val confidence = result.optDouble("confidence", Double.NaN)
        findViewById<TextView>(R.id.detailConfidence).text =
            // Never .format() a string that already holds a percent sign — the
            // "%" from percent() would be read as a format specifier.
            if (confidence.isNaN()) ""
            else "model confidence ${percent(confidence)} · entropy " +
                "%.3f".format(result.optDouble("entropy", 0.0))

        val classes = result.optJSONObject("classes")
        if (classes != null) {
            val container = findViewById<LinearLayout>(R.id.classesContainer)
            for (key in classes.keys()) addBar(container, key, classes.optDouble(key))
        }

        val links = result.optJSONArray("links")
        if (links != null && links.length() > 0) {
            findViewById<View>(R.id.linksCard).visibility = View.VISIBLE
            val container = findViewById<LinearLayout>(R.id.linksContainer)
            for (i in 0 until links.length()) {
                val link = links.getJSONObject(i)
                addBar(container, link.optString("url"), link.optDouble("score"))
            }
        }
    }

    private fun addBar(container: LinearLayout, label: String, value: Double) {
        val row = inflater.inflate(R.layout.item_bar, container, false)
        row.findViewById<TextView>(R.id.barLabel).text = label
        row.findViewById<TextView>(R.id.barValue).text = percent(value)
        row.findViewById<ProgressBar>(R.id.barProgress).apply {
            progress = (value * 1000).toInt()
            progressTintList = ColorStateList.valueOf(
                color(if (value >= 0.7) R.color.band_phishing else R.color.accent)
            )
        }
        container.addView(row)
    }

    private fun color(res: Int): Int = ContextCompat.getColor(this, res)
}
