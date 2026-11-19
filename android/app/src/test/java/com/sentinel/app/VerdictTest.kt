package com.sentinel.app

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// plainPoints decides what a non-technical person is told about a message, so
// a flipped threshold here is a real harm, not a cosmetic bug.
class VerdictTest {

    private fun points(json: String) = plainPoints(JSONObject(json))

    @Test
    fun `a phishing link is called out, in words, not feature names`() {
        val result = points(
            """{"label":"smishing","evidence":1.0,"parts":{"nlp":0.998,"url":0.998},
                "terms":["incomplete","Update"],
                "links":[{"url":"sbi-kyc-verify.tk/update","score":0.998,
                          "why":["rare top-level domain","hyphen-stuffed host"]}]}"""
        )
        val link = result.single { it.text.startsWith("The link") }
        assertEquals(PointKind.BAD, link.kind)
        assertTrue(link.text.contains("does not look like a real company's website"))
        // Translated through LINK_SAY, so the raw feature name never reaches the user.
        assertTrue(link.text.contains("scammers buy because it is cheap"))
        assertTrue(link.text.contains("padded with hyphens"))

        val terms = result.single { it.terms.isNotEmpty() }
        assertEquals(PointKind.BAD, terms.kind)
        assertEquals(listOf("incomplete", "Update"), terms.terms)
    }

    @Test
    fun `an ordinary message reads as reassurance throughout`() {
        val result = points(
            """{"label":"ham","evidence":0.0,"parts":{"nlp":0.002,"behaviour":0.1},
                "terms":["lunch"],"links":[]}"""
        )
        assertTrue(result.all { it.kind == PointKind.GOOD })
        assertTrue(result.any { it.text == "There is no link in it to tap." })
        assertTrue(result.any { it.text.startsWith("The sender behaves like") })
        assertTrue(result.any { it.text.startsWith("It does not actually ask you to do anything") })
        assertTrue(result.any { it.text.contains("everyday words") })
    }

    @Test
    fun `a middling link is a warning, not an accusation`() {
        val link = points("""{"label":"spam","links":[{"url":"bit.ly/x","score":0.5}]}""")
            .single { it.text.startsWith("The link") }
        assertEquals(PointKind.BAD, link.kind)
        assertTrue(link.text.endsWith("check it before you tap it."))
    }

    @Test
    fun `a benign link is not dressed up as a threat`() {
        val link = points("""{"label":"ham","links":[{"url":"hdfcbank.com","score":0.02}]}""")
            .single { it.text.startsWith("The link") }
        assertEquals(PointKind.GOOD, link.kind)
        assertTrue(link.text.contains("normal website address"))
    }

    @Test
    fun `an abstaining behaviour channel says so, instead of claiming the sender is fine`() {
        // fusion.fuse drops abstaining channels from `parts` entirely.
        val point = points("""{"label":"smishing","parts":{"nlp":0.9}}""")
            .single { it.text.startsWith("The sender") }
        assertEquals(PointKind.NOTE, point.kind)
        assertTrue(point.text.contains("could not be judged yet"))
    }

    @Test
    fun `a badly behaved sender is flagged`() {
        val point = points("""{"label":"smishing","parts":{"nlp":0.9,"behaviour":0.6}}""")
            .single { it.text.startsWith("The sender") }
        assertEquals(PointKind.BAD, point.kind)
        assertTrue(point.text.contains("does not behave like"))
    }

    @Test
    fun `scores read as percentages without trailing noise`() {
        assertEquals("99.8%", percent(0.998))
        assertEquals("0.1%", percent(0.001))
        assertEquals("0%", percent(0.0))
        assertEquals("100%", percent(1.0))
        assertEquals("50%", percent(0.5))
        assertEquals("82.9%", percent(0.829))
    }

    @Test
    fun `bands map to the words the dashboard uses`() {
        assertEquals("This looks like a scam", bandCopy("PHISHING").say)
        assertEquals("Be careful with this one", bandCopy("SUSPICIOUS").say)
        assertEquals("Nothing dangerous found", bandCopy("SAFE").say)
        assertEquals("Malicious", bandCopy("PHISHING").label)
    }

    // Spam used to be invisible on the list: fusion says SAFE for it, so an
    // unsolicited promo and a genuine bank alert wore the same green "Safe".
    @Test
    fun `a safe promo says Spam, and nothing else gains a badge`() {
        assertEquals("Spam", rowBadge("SAFE", CATEGORY_SPAM).label)
        assertEquals(R.color.band_spam, rowBadgeColorRes("SAFE", CATEGORY_SPAM))

        // The bands are untouched, and this is the one that matters: a
        // smishing label under a SAFE verdict means fusion overruled the
        // content model. Printing "Scam" there would contradict the verdict.
        assertEquals("Safe", rowBadge("SAFE", CATEGORY_SCAM).label)
        assertEquals("Safe", rowBadge("SAFE", CATEGORY_SAFE).label)
        assertEquals("Malicious", rowBadge("PHISHING", CATEGORY_SPAM).label)
        assertEquals("Suspicious", rowBadge("SUSPICIOUS", CATEGORY_SPAM).label)
    }
}

// The samples are what a new install shows before any real SMS arrives, and
// they carry their own verdict instead of being scored — so nothing but this
// catches a sample whose verdict and label disagree, or one missing a field
// the detail screen reads.
class SamplesTest {

    @Test
    fun `every sample fills one section, and its verdict matches its label`() {
        val sections = SAMPLE_MESSAGES.map { (sender, body, result) ->
            assertTrue("sender is blank", sender.isNotBlank())
            assertTrue("body is blank", body.isNotBlank())

            val label = result.getString("label")
            val expected = when (label) {
                "ham" -> "SAFE"
                "spam" -> "SUSPICIOUS"
                "smishing" -> "PHISHING"
                else -> throw AssertionError("unknown label $label")
            }
            assertEquals(expected, result.getString("verdict"))
            // The detail screen reads these off every result; a missing one
            // silently renders as 0.
            for (key in listOf("risk", "confidence", "entropy", "evidence"))
                assertTrue("$label is missing $key", result.has(key))
            for (key in listOf("classes", "parts", "weights"))
                assertTrue("$label is missing $key", result.optJSONObject(key) != null)

            categoryFor(label)
        }
        assertEquals(
            listOf(CATEGORY_SAFE, CATEGORY_SPAM, CATEGORY_SCAM).sorted(),
            sections.sorted(),
        )
    }

    @Test
    fun `plainPoints renders every sample without a placeholder leaking through`() {
        for ((_, _, result) in SAMPLE_MESSAGES) {
            val points = plainPoints(result)
            assertTrue("no points for ${result.getString("label")}", points.isNotEmpty())
            // LINK_SAY translates every reason; an untranslated one reaches the
            // user as a raw feature name.
            assertTrue(points.none { it.text.contains("rare top-level domain") })
        }
    }
}
