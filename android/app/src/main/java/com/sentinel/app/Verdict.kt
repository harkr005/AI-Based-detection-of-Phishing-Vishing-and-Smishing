package com.sentinel.app

import org.json.JSONObject

// The plain-language layer, ported from web/components/verdict-panel.jsx so the
// phone explains a verdict in the same words the dashboard does. Nothing here
// re-decides anything — it only says out loud what the verdict was built from.
//
// The email/voice-only branches of the web version (solicitation, persuasion,
// synthetic-voice) are left out: api.py never returns those on /sms.

data class BandCopy(val label: String, val say: String)

/** §3's three bands, and nothing else is ever rendered as a verdict. */
fun bandCopy(verdict: String?): BandCopy = when (verdict) {
    "PHISHING" -> BandCopy("Malicious", "This looks like a scam")
    "SUSPICIOUS" -> BandCopy("Suspicious", "Be careful with this one")
    else -> BandCopy("Safe", "Nothing dangerous found")
}

/**
 * The badge on a message row: the three bands, plus one.
 *
 * A SAFE message the content model read as marketing now says "Spam" instead
 * of "Safe" — same verdict, said more usefully. It is NOT a fourth band and it
 * cannot become one: only the spam label qualifies, and only under SAFE.
 *
 * A smishing label under a SAFE verdict deliberately gets nothing. That pair
 * means fusion overruled the content model on the evidence, and printing
 * "Scam" over a Safe verdict would contradict the verdict rather than explain
 * it — which is the mistake the `category` column invites and §3 forbids.
 */
fun rowBadge(verdict: String?, category: String): BandCopy =
    if (verdict == "SAFE" && category == CATEGORY_SPAM) {
        BandCopy("Spam", "Unwanted marketing, not a scam")
    } else {
        bandCopy(verdict)
    }

fun rowBadgeColorRes(verdict: String?, category: String): Int =
    if (verdict == "SAFE" && category == CATEGORY_SPAM) R.color.band_spam
    else bandColorRes(verdict)

fun bandColorRes(verdict: String?): Int = when (verdict) {
    "PHISHING" -> R.color.band_phishing
    "SUSPICIOUS" -> R.color.band_suspicious
    "SAFE" -> R.color.band_safe
    else -> R.color.text_secondary
}

/** What the class label means, for someone who has never heard of smishing. */
fun kindSentence(label: String): String = when (label) {
    "ham" -> "It reads like an ordinary, genuine message."
    "spam" -> "It reads like unwanted marketing — junk mail rather than a scam."
    "smishing" -> "It reads like a scam text: it pretends to be your bank, a " +
        "delivery service or a government office to get your details or your money."
    "phishing" -> "It reads like phishing: a message that pretends to be someone " +
        "you trust so you hand over a password, a card number or money."
    else -> ""
}

/**
 * 0.998 -> "99.8%", 1.0 -> "100%", 0.5 -> "50%". Display only: every score
 * underneath, and §3's bands, stay the 0-1 values the spec fixes them at.
 * Not for entropy — that is an information measure (0 to ln 3), not a share.
 */
fun percent(value: Double): String {
    val text = "%.1f".format(value * 100)
    return (if (text.endsWith(".0")) text.dropLast(2) else text) + "%"
}

fun channelLabel(key: String): String = when (key) {
    "nlp" -> "Content (NLP)"
    "url" -> "Link (XGBoost)"
    "behaviour" -> "Sender behaviour"
    "spoof" -> "Synthetic voice"
    else -> key
}

// url_model's own feature names (url_model.PHRASE) said plainly. Anything not
// listed falls through unchanged, so a new feature degrades to its short name
// rather than disappearing.
private val LINK_SAY = mapOf(
    "rare top-level domain" to
        "its ending (.tk, .top, .xyz and the like) is one scammers buy because it is cheap",
    "unusual TLD length" to "the part after the last dot is an odd length for a real site",
    "hyphen-stuffed host" to "the address is padded with hyphens to look like a real brand",
    "raw IP address instead of a domain" to "it is a bare numeric address, not a company's domain",
    "punycode (look-alike) domain" to "it uses look-alike foreign characters to imitate a real name",
    "bank/payment lure words in the host" to
        "the address itself contains bank or payment words to look official",
    "digits in the host" to "there are digits inside the address a real brand would not use",
    "host is mostly digits" to "the address is mostly numbers",
    "deep subdomain nesting" to "the real domain is hidden behind extra sub-parts",
    "unusually long host name" to "the address is unusually long",
    "one very long host token" to "one chunk of the address is unusually long",
    "many dot-separated parts" to "the address is split into unusually many parts",
    "odd characters in the host" to "there are characters in it a normal address would not have",
    "random-looking host string" to "the address looks randomly generated",
    "host reads as non-word letters" to "the address is not made of real words",
)

enum class PointKind { GOOD, BAD, NOTE }

data class Point(
    val kind: PointKind,
    val text: String,
    val terms: List<String> = emptyList(),
)

/** The whole plain-language layer: structured fields -> sentences. */
fun plainPoints(result: JSONObject): List<Point> {
    val points = mutableListOf<Point>()
    val label = result.optString("label")

    // The exact words the content model leaned on (sms_model.explain returns
    // them as data). This is the question a new user actually asks — "what did
    // it see?"
    val terms = result.optJSONArray("terms")
    if (terms != null && terms.length() > 0) {
        val words = (0 until terms.length()).map { terms.getString(it) }
        points += if (label == "ham") {
            Point(
                PointKind.GOOD,
                "These everyday words are what made it read as an ordinary message:",
                words,
            )
        } else {
            Point(PointKind.BAD, "These words in it carried the most weight:", words)
        }
    }

    // ...and WHY the link is wrong, in words, not feature names.
    val links = result.optJSONArray("links")
    val worst = if (links != null && links.length() > 0) links.optJSONObject(0) else null
    if (worst != null) {
        val why = worst.optJSONArray("why")
        val said = if (why == null) emptyList() else (0 until why.length())
            .map { why.getString(it) }
            .map { LINK_SAY[it] ?: it }
        val tail = if (said.isEmpty()) "" else " — " + said.joinToString("; ")
        val url = worst.optString("url")
        val score = worst.optDouble("score", 0.0)
        points += when {
            score >= 0.7 -> Point(
                PointKind.BAD,
                "The link ($url) does not look like a real company's website$tail.",
            )
            score >= 0.35 -> Point(
                PointKind.BAD,
                "The link ($url) looks a little odd$tail — check it before you tap it.",
            )
            else -> Point(PointKind.GOOD, "The link in it looks like a normal website address.")
        }
    } else {
        points += Point(PointKind.GOOD, "There is no link in it to tap.")
    }

    // fusion.fuse drops a channel from `parts` when it abstained, so a missing
    // key means "not judged", never "judged harmless".
    val parts = result.optJSONObject("parts")
    if (parts != null && parts.has("behaviour")) {
        points += if (parts.optDouble("behaviour", 0.0) >= 0.5) {
            Point(
                PointKind.BAD,
                "The sender does not behave like the people you normally hear from: " +
                    "new number, unusual hour or unusual wording.",
            )
        } else {
            Point(PointKind.GOOD, "The sender behaves like someone you normally hear from.")
        }
    } else {
        points += Point(
            PointKind.NOTE,
            "The sender could not be judged yet — this app has not seen enough of your " +
                "past messages to know who is normal for you.",
        )
    }

    if (result.optDouble("evidence", -1.0) == 0.0) {
        points += Point(
            PointKind.GOOD,
            "It does not actually ask you to do anything: no code to share, no payment, no threat.",
        )
    }

    return points
}
