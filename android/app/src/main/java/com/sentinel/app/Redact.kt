package com.sentinel.app

import org.json.JSONArray
import org.json.JSONObject

// PROJECT_SPEC.md §10: "PII redaction runs on-device before serialization
// (OTP, account no., Aadhaar, PAN, amounts, emails)."
//
// The app holds RECEIVE_SMS, so it sees every message — including the bank OTP
// that arrived thirty seconds ago. A pipeline that quietly accumulates those is
// an account-takeover kit, not a scam detector. Nothing is uploaded any more;
// the one thing that can leave is the audit report the user chooses to share,
// and its text goes through here first.
//
// Values are replaced with typed placeholders rather than deleted, because the
// SHAPE is the detection signal and the VALUE is the private part: "your OTP is
// [code]" still reads as an OTP message, and "pay [amount] now" still reads as
// a demand for money.
object Redact {

    // URLs are held out and put back afterwards. A scam link is the evidence,
    // not the user's secret, and url_model scores the host and path — redacting
    // digits inside it would break the very features that catch the scam.
    private val URL = Regex("""https?://\S+|www\.\S+""", RegexOption.IGNORE_CASE)

    // Order matters: the most specific shape has to match before a looser one
    // eats its digits. Amounts carry their own currency prefix, so they come
    // before the bare digit runs.
    private val RULES = listOf(
        Regex("""\b[\w.+-]+@[\w-]+\.[\w.]+\b""") to "[email]",
        // A UPI id is an email with no dot after the @ ("name@oksbi"), so the
        // rule above misses it, and every credit SMS names the payer's.
        Regex("""\b[\w.-]+@[A-Za-z]{2,}\b""") to "[upi]",
        Regex("""\b[A-Z]{5}\d{4}[A-Z]\b""") to "[pan]",
        Regex("""(?i)(?:rs\.?|inr|₹)\s?\d[\d,]*(?:\.\d+)?""") to "[amount]",
        Regex("""\b\d{4}\s\d{4}\s\d{4}\b""") to "[aadhaar]",
        Regex("""\b(?:\d[ -]?){13,19}\b""") to "[card]",
        Regex("""\b\d{9,18}\b""") to "[account]",
        // Last, and deliberately broad: an OTP is just a short digit run, and
        // anything of that shape we have not already named is not ours to send.
        Regex("""\b\d{4,8}\b""") to "[code]",
    )

    fun apply(text: String): String {
        val held = mutableListOf<String>()
        var masked = URL.replace(text) { match ->
            held += match.value
            // The placeholder is letter-led so the [code] rule cannot match it.
            "\u0000U${held.size - 1}\u0000"
        }
        for ((pattern, tag) in RULES) masked = pattern.replace(masked, tag)
        held.forEachIndexed { i, url -> masked = masked.replace("\u0000U$i\u0000", url) }
        return masked
    }

    /**
     * The model's result with its `terms` redacted, for the audit report. The
     * model explains RAW text, so a term can be the OTP itself, which would undo
     * redacting the body next to it. Copied, not edited: the UI holds `result`.
     */
    fun detail(result: JSONObject): JSONObject {
        val copy = JSONObject(result.toString())
        // `reason` quotes the same raw words ("driven by \"481920\"").
        if (copy.has("reason")) copy.put("reason", apply(copy.getString("reason")))
        val terms = copy.optJSONArray("terms") ?: return copy
        return copy.put("terms", JSONArray((0 until terms.length()).map { apply(terms.getString(it)) }))
    }
}

