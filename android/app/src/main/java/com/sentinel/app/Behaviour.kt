package com.sentinel.app

import android.content.Context
import java.io.InputStream
import kotlin.math.min

/**
 * `behaviour.sms_features` / `sms_score` (§7), on the phone.
 *
 * 0.85 of the 1.30 rule budget is STATELESS — a personal number claiming to be
 * an institution (0.60), a non-DLT header (0.15) and an odd hour (0.10) all read
 * the message and its sender and nothing else. The rest (first_contact 0.10,
 * url_first_contact 0.25, burst_1h 0.10) needs history, and on a phone that
 * history is `content://sms` itself rather than a sender-profile table: see
 * [Inbox]. No profile store ships here, so `circ_dev` and the email features
 * stay in Python where the mailbox is.
 *
 * These rules carry no corpus number — every public SMS set is bare text with no
 * sender field. On the phone they finally meet real metadata.
 */
object Behaviour {

    // A DLT header's value is that an OPERATOR assigned it to a registered
    // entity, not that it is two letters and a hyphen. HEADER matches the SHAPE,
    // which is all the anomaly features need; anything granting TRUST must also
    // find the entity in the registry.
    private val HEADER = Regex("""^(?:[A-Z]{2}-)?([A-Z0-9_]{3,13})(?:-[PSTG])?$""")

    // TRUST gets the strict form. The two-letter operator prefix is the one part
    // of a header nobody can choose: the access provider stamps it on in transit
    // during DLT scrubbing, along with the category suffix. Since 6 May 2025 a
    // carrier-delivered message arrives as AX-ICICIT-S or DM-AXISBK-T, never as
    // a bare ICICIT — so a bare entity is not a weaker form of a real header, it
    // is a form a real header no longer takes. [HEADER] stays lenient for the
    // anomaly rule; only [registered] is strict, because only it grants anything.
    private val CARRIER_HEADER = Regex("""^[A-Z]{2}-([A-Z0-9_]{3,13})(?:-[PSTG])?$""")
    private val TEN_DIGIT = Regex("""^\+?(?:91)?[6-9]\d{9}$""")
    private val INSTITUTION = Regex(
        """\b(bank|kyc|account|otp|upi|netbanking|card|loan|""" +
            """insurance|refund|customs|courier|electricity)\b""",
        RegexOption.IGNORE_CASE,
    )

    /** behaviour.SMS_W — the weights, and they are the calibration knob. */
    val WEIGHTS = listOf(
        "personal_number_as_institution" to 0.60,
        "non_dlt_header" to 0.15,
        "first_contact" to 0.10,
        "url_first_contact" to 0.25,
        "burst_1h" to 0.10,
        "odd_hour" to 0.10,
    )

    // behaviour.SMS_TAGS, in the order a reason prefers them — strongest first,
    // and at most three are ever said out loud.
    private val TAGS = listOf(
        "personal_number_as_institution" to "personal number claiming to be an institution",
        "url_first_contact" to "link in a first-ever message from this sender",
        "non_dlt_header" to "sender ID does not follow the DLT header format",
        "burst_1h" to "burst of messages from this sender",
        "odd_hour" to "delivered at an odd hour",
        "first_contact" to "no prior history with this sender",
    )

    private var registry: Set<String>? = null

    @Synchronized
    fun ensureLoaded(context: Context) {
        if (registry == null) {
            context.applicationContext.assets.open("dlt_registry.txt").use { loadRegistry(it) }
        }
    }

    @Synchronized
    fun loadRegistry(stream: InputStream) {
        registry = stream.bufferedReader().readLines()
            .map { it.trim() }.filter { it.isNotEmpty() }.toSet()
    }

    /**
     * Normalise a sender the way a person actually types one. Only one rewrite,
     * and it is the one that matters: a phone number typed with spaces, brackets
     * or hyphens is still a phone number. Without it "+91 98765 43210" misses
     * TEN_DIGIT and silently loses personal_number_as_institution — 0.60 of the
     * SMS score — to a space bar.
     */
    fun sid(sender: String?): String {
        val s = (sender ?: "").trim().uppercase()
        val bare = s.replace(Regex("""[\s()-]"""), "")
        return if (TEN_DIGIT.matches(bare)) bare else s
    }

    /** 'AD-AXISBK-S', 'AX-HDFCBK', 'HDFCBK' -> the registered-entity part. */
    fun entity(sender: String?): String {
        val s = sid(sender)
        return HEADER.matchEntire(s)?.groupValues?.get(1) ?: s
    }

    /**
     * -> True only for a CARRIER-STAMPED header whose entity is in the registry.
     *
     * Two independent things, both required: an access provider put its prefix
     * on the message in transit, and the entity inside is one the registry
     * names. A format match alone is not trust — treating shape as trust let any
     * typed string earn a de-escalation.
     */
    fun registered(sender: String?): Boolean {
        val entity = CARRIER_HEADER.matchEntire(sid(sender))?.groupValues?.get(1)
        return entity != null && entity in (registry ?: emptySet())
    }

    /**
     * §7's six SMS rules -> name to value in [0,1].
     *
     * @param firstContact null when the inbox cannot answer. BOTH first_contact
     * and url_first_contact then abstain — absence from a store we could not
     * read is not evidence of a first contact, and url_first_contact carries
     * 0.25 on that same claim. See [Inbox.firstFrom].
     * @param hour the hour the message was SENT, or null. The scan clock is not
     * the message clock; a default of noon would be the same lie with a
     * friendlier sign.
     */
    fun smsFeatures(
        sender: String?,
        text: String,
        firstContact: Boolean?,
        hasUrl: Boolean,
        hour: Int?,
        burst: Int = 0,
    ): Map<String, Double> {
        val s = sid(sender)
        val isNumber = TEN_DIGIT.matches(s)
        return mapOf(
            // §7's very strong signal: under Indian DLT rules a personal number
            // claiming to be an institution can never be legitimate.
            "personal_number_as_institution" to
                bit(isNumber && INSTITUTION.containsMatchIn(text)),
            "non_dlt_header" to bit(!HEADER.matches(s) && !isNumber),
            "first_contact" to if (firstContact == null) 0.0 else bit(firstContact),
            "url_first_contact" to
                if (firstContact == null) 0.0 else bit(firstContact && hasUrl),
            "burst_1h" to min(burst / 5.0, 1.0),
            "odd_hour" to if (hour == null) 0.0 else bit(hour < 7 || hour >= 23),
        )
    }

    /** -> anomaly score in [0,1] and at most three tags, strongest first. */
    fun smsScore(f: Map<String, Double>): Pair<Double, List<String>> =
        WEIGHTS.sumOf { (k, w) -> (f[k] ?: 0.0) * w }.coerceIn(0.0, 1.0) to
            TAGS.filter { (k, _) -> (f[k] ?: 0.0) >= 0.5 }.map { it.second }.take(3)

    private fun bit(b: Boolean) = if (b) 1.0 else 0.0
}

/**
 * `voice_model.evidence_risk` — the act layer (§8's act gate).
 *
 * Content models key on TOPIC: a message is scored as smishing because it is
 * about banking, OTPs and blocked accounts. A bank's own fraud advisory is about
 * all three, which is why 83% of real awareness SMS were being flagged. This
 * asks the different question — did the message ASK for something, threaten, or
 * demand payment — and scores 0.00 on every one of them.
 */
object Acts {

    private fun rx(pattern: String) = Regex(pattern, RegexOption.IGNORE_CASE)

    // The verb list gained update/verify/activate/complete and the object list
    // gained kyc/account/card on 2026-08-22: the original was written for spoken
    // calls, where the ask is "read me the OTP", but the Indian SMS ask is
    // "update your KYC".
    private val ASK = rx(
        """\b(share|tell|give|send|read|confirm|provide|type|enter|dictate|""" +
            """update|verify|activate|complete|revalidate)""" +
            """\b[^.?!]{0,40}\b(otp|cvv|pin|code|password|aadha?ar|pan|kyc|""" +
            """account|debit card|credit card|net ?banking)\b""",
    )
    private val THREAT = rx(
        """\b(account[^.?!]{0,20}(blocked|suspended|freez|clos)|""" +
            """arrest|police|cbi|legal\s+action|case\s+(filed|registered)|""" +
            """disconnect|expire[sd]?\s+today|within\s+\d+\s+(minute|hour))""",
    )
    private val REMOTE = rx(
        """\b(anydesk|teamviewer|remote\s+access|screen\s+shar|""" +
            """install[^.?!]{0,20}app|gift\s+card|google\s+play\s+code)\b""",
    )
    // "transfer" dropped to match voice_model.MONEY (fired on 13.3% of bank ham).
    private val MONEY = rx(
        """\b(scan[^.?!]{0,30}(qr|code)|(send|pay|deposit|arrange)""" +
            """[^.?!]{0,40}\b(rupees|amount|money|account|upi|qr|gpay|google pay|""" +
            """paytm|thousand|lakh)|processing (fee|charge)|refundable)\b""",
    )
    private val SECRECY = rx(
        """\b(do not (tell|discuss|disclose|inform|speak to)|don't tell|""" +
            """do not disconnect|keep (it |this )?(confidential|secret)|""" +
            """without (telling|informing)|no ?one should know)\b""",
    )

    // ENTITY (the mere mention of otp/kyc/aadhaar/card) is deliberately ABSENT:
    // it is a topic, not an act, and it fires on 20% of legitimate transactional
    // SMS against 3.5% of the malicious half — the message most likely to
    // contain the word "OTP" is a real OTP. It keeps its 0.25 in the voice
    // channel's scoring; it must never count as evidence that an act occurred.
    private val EVIDENCE = listOf(0.65 to ASK, 0.45 to THREAT, 0.65 to REMOTE,
                                  0.45 to MONEY, 0.45 to SECRECY)

    // Warning words only — never / not / avoid / beware. NOT fraud/fake/scam,
    // which look like warning cues and are not: "fraud detected on your account,
    // call now" is one of the commonest scam openers there is.
    private val NEG = rx("""\b(never|not|n't|avoid|beware|refrain|no need|instead of)\b""")
    private const val NEG_WINDOW = 45

    /** -> act evidence in [0,1]. Fires on 2.5% of legitimate transactional SMS
     * against 37.3% of credential-style smishing. */
    fun evidenceRisk(text: String): Double =
        min(1.0, EVIDENCE.filter { (_, rx) -> fires(rx, text) }.sumOf { it.first })

    /**
     * True if the pattern matches somewhere it is not being WARNED against.
     *
     * A bank's fraud advisory contains the exact words of the attack it warns
     * about. On text the negation window is what tells them apart (0.0% of bank
     * awareness SMS against 50.0% without it, for 2.4 points of smishing recall)
     * — and it is a window, not a lookbehind, because Python lookbehinds must be
     * fixed width and "Do not click on links asking you to update your KYC"
     * slipped through when only the one preceding word was checked.
     */
    private fun fires(pattern: Regex, text: String): Boolean =
        pattern.findAll(text).any { m ->
            !NEG.containsMatchIn(text.substring(maxOf(0, m.range.first - NEG_WINDOW), m.range.first))
        }
}
