package com.sentinel.app

import kotlin.math.ln
import kotlin.math.min

/**
 * `fusion.py` (§8), on the phone — one verdict from content, link and sender.
 *
 * Adaptive weighting is the §8 rule verbatim: when the content model is unsure,
 * trust structure instead. That is why the behaviour channel is worth having
 * offline at all — it carries 0.45 of the weight exactly when the content model
 * cannot decide.
 */
object Fusion {

    const val SUSPICIOUS = 0.35   // §3 bands
    const val PHISHING = 0.70
    private const val LOW_CONF = 0.60      // §8 switch point
    private const val HIGH_ENTROPY = 0.80  // of max entropy; above this -> never SAFE
    private const val ESCALATE = 0.60

    private val W_SURE = listOf("nlp" to 0.50, "url" to 0.30, "behaviour" to 0.20)
    private val W_UNSURE = listOf("nlp" to 0.25, "url" to 0.30, "behaviour" to 0.45)

    /** One event -> the verdict fields. Channels that abstained are absent from
     * [parts] and [weights], never present as a zero. */
    data class Verdict(
        val verdict: String,
        val risk: Double,
        val confidence: Double,
        val entropy: Double,
        val weights: Map<String, Double>,
        val parts: Map<String, Double>,
        val reason: String,
    )

    fun band(risk: Double): String = when {
        risk >= PHISHING -> "PHISHING"
        risk >= SUSPICIOUS -> "SUSPICIOUS"
        else -> "SAFE"
    }

    /** Normalised prediction entropy in [0,1] — §8's uncertainty signal. */
    fun entropy(probs: FloatArray): Double {
        var sum = 0.0
        for (p in probs) {
            val q = p.toDouble().coerceIn(1e-9, 1.0)
            sum -= q * ln(q)
        }
        return sum / ln(probs.size.toDouble())
    }

    /**
     * @param url max link score in the message; null = no URL engine output at
     * all, which is NOT the same as 0.0 "scored and found benign".
     * @param behaviour §7's anomaly score; null = no sender to judge.
     */
    fun fuse(
        probs: FloatArray,
        url: Double?,
        behaviour: Double?,
        reasons: List<String>,
        maliciousIdx: Int = 2,
    ): Verdict {
        val mal = probs[maliciousIdx].toDouble()
        val confidence = probs.max().toDouble()
        val h = entropy(probs)

        // Renormalise over the channels that actually reported. §8's weights
        // assume all three always fire; spending a silent channel's share on a
        // zero drags every risk down by up to 45% and the fixed §3 bands stop
        // meaning anything.
        val parts = mapOf("nlp" to mal, "url" to url, "behaviour" to behaviour)
        val live = (if (confidence >= LOW_CONF) W_SURE else W_UNSURE)
            .filter { (k, _) -> parts[k] != null }
        val total = live.sumOf { it.second }
        val weights = live.associate { (k, w) -> k to w / total }
        val risk = weights.entries.sumOf { (k, w) -> w * parts[k]!! }

        var verdict = band(risk)
        // §8: high variance routes to SUSPICIOUS — never to SAFE.
        val routed = verdict == "SAFE" && h >= HIGH_ENTROPY
        // ...and the measured addition: the weighted sum alone can never lift
        // content-clean mail off SAFE, which is the one job §8 wrote the
        // behaviour weight to do. SAFE -> SUSPICIOUS only, never to PHISHING on
        // structure alone.
        val escalated = verdict == "SAFE" && (behaviour ?: 0.0) >= ESCALATE
        if (routed || escalated) verdict = "SUSPICIOUS"

        val why = reasons.filter { it.isNotEmpty() }.toMutableList()
        if (confidence < LOW_CONF) why += "content model unsure, weighted towards sender behaviour"
        if (escalated) why += "content looks ordinary but the sender does not"
        if (routed) why += "model uncertain - held as suspicious rather than cleared"

        return Verdict(
            verdict = verdict,
            risk = round3(risk),
            confidence = round3(confidence),
            entropy = round3(h),
            weights = weights,
            parts = parts.filterValues { it != null }.mapValues { round3(it.value!!) },
            reason = why.joinToString("; ").ifEmpty { "no risk signal on any channel" },
        )
    }

    /**
     * Damp the malicious probability when the text asks for NOTHING.
     *
     * A damper, not a veto: a saturated p=0.99 survives it (0.99 * 0.5
     * renormalises to ~0.97), which is why smishing recall barely moves. Only
     * the borderline cases the content model was unsure about and got wrong
     * fall back under the §3 band. 0.5 is the measured knee — it takes the
     * whole transactional false-positive win (10.0% -> 2.5%) for 3.7 points of
     * recall, and everything below it buys the awareness column steeply.
     */
    fun gateByAct(probs: FloatArray, evidence: Double, damp: Double = 0.5,
                  maliciousIdx: Int = 2): FloatArray {
        if (evidence > 0) return probs
        val p = probs.copyOf()
        p[maliciousIdx] = (p[maliciousIdx] * damp).toFloat()
        val sum = maxOf(p.sum().toDouble(), 1e-9)
        return FloatArray(p.size) { (p[it] / sum).toFloat() }
    }

    /**
     * Clear a content false positive when three independent channels agree the
     * message is benign: a REGISTERED sender, no malicious link, and no scam act
     * in the text.
     *
     * THE ONE DEPENDENCY IS HEADER INTEGRITY, so `trustedSender` must never be
     * true for a sender nobody vouched for. On this phone it is false today: the
     * header does arrive from the carrier rather than from a text box, which is
     * exactly the §10 gateway case the gate was written for, but turning it on
     * is a decision with a measured cost (2 legitimate messages cleared, 14
     * smishing cleared with them on the transactional harness) and it has not
     * been taken.
     */
    fun deescalate(f: Verdict, evidence: Double, url: Double?, trustedSender: Boolean,
                   cap: Double = 0.30): Verdict {
        if (f.verdict == "SAFE") return f
        if (!(trustedSender && evidence == 0.0 && (url == null || url < 0.5))) return f
        val risk = min(f.risk, cap)
        return f.copy(
            risk = risk,
            verdict = band(risk),
            reason = f.reason + "; registered sender, clean link and no scam act - de-escalated",
        )
    }

    /** api.py rounds every score it returns to three places; the dashboard and
     * the phone have to agree digit for digit, so the port rounds too. */
    fun round3(x: Double): Double = Math.round(x * 1000.0) / 1000.0
}
