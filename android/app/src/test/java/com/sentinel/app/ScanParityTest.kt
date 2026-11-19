package com.sentinel.app

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The on-device SMS pipeline against the Python one it was ported from, case by
 * case, from `scan_parity.json` (`python export_android.py`).
 *
 * The end-to-end cases were produced by calling `api.sms` itself, not by
 * re-assembling the steps in the export script — a fixture that repeated the
 * composition under test would happily agree with a port that got the
 * composition wrong.
 */
class ScanParityTest {

    companion object {
        private val ASSETS = File("src/main/assets")
        private lateinit var fixture: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            assertTrue("run `python export_android.py`", ASSETS.resolve("url_xgb.bin").exists())
            ASSETS.resolve("sms_charlr.bin").inputStream().use { SmsModel.load(it) }
            ASSETS.resolve("url_xgb.bin").inputStream().use { UrlModel.loadTrees(it) }
            ASSETS.resolve("verified_domains.txt").inputStream().use { UrlModel.loadRegistry(it) }
            ASSETS.resolve("dlt_registry.txt").inputStream().use { Behaviour.loadRegistry(it) }
            fixture = JSONObject(File("src/test/resources/scan_parity.json").readText())
        }

        private fun cases(name: String): List<JSONObject> {
            val array = fixture.getJSONArray(name)
            return (0 until array.length()).map { array.getJSONObject(it) }
        }

        private fun strings(array: JSONArray?): List<String> =
            if (array == null) emptyList() else (0 until array.length()).map { array.getString(it) }
    }

    @Test
    fun urlFeaturesAndScores() {
        for (case in cases("url_cases")) {
            val url = case.getString("url")
            val expected = case.getJSONObject("features")
            val got = UrlModel.features(url)
            UrlModel.FEATURES.forEachIndexed { i, name ->
                assertEquals("$name of $url", expected.getDouble(name), got[i], 1e-9)
            }
            assertEquals("verified $url", case.getBoolean("verified"), UrlModel.isVerified(url))
            // 1e-5, not 1e-9: xgboost accumulates its margin in float32, so the
            // last places of a 400-tree sum are its rounding, not a porting gap.
            assertEquals("score $url", case.getDouble("score"), UrlModel.score(url), 1e-5)

            // Exact TreeSHAP, against xgboost's own pred_contribs. Same float32
            // caveat, and then the thing that actually reaches the screen: two
            // contribution vectors differing in the sixth place still have to
            // pick the same phrases in the same order.
            val contributions = UrlModel.contributions(got)
            val expectedContribs = case.getJSONArray("contribs")
            for (i in UrlModel.FEATURES.indices) {
                assertEquals("contribution ${UrlModel.FEATURES[i]} of $url",
                    expectedContribs.getDouble(i), contributions[i], 1e-5)
            }
            val explained = UrlModel.explain(url)
            assertEquals("named for $url", strings(case.getJSONArray("named")), explained.named)
            assertEquals("reason for $url", case.getString("reason"), explained.reason)
        }
    }

    @Test
    fun behaviourRules() {
        for (case in cases("behaviour_cases")) {
            val sender = case.getString("sender")
            assertEquals("sid $sender", case.getString("sid"), Behaviour.sid(sender))
            assertEquals("entity $sender", case.getString("entity"), Behaviour.entity(sender))
            assertEquals("registered $sender",
                case.getBoolean("registered"), Behaviour.registered(sender))

            val features = Behaviour.smsFeatures(
                sender = sender,
                text = case.getString("text"),
                firstContact = if (case.isNull("first_contact")) null
                               else case.getBoolean("first_contact"),
                hasUrl = case.getBoolean("has_url"),
                hour = if (case.isNull("hour")) null else case.getInt("hour"),
                burst = case.getInt("burst"),
            )
            val expected = case.getJSONObject("features")
            for ((name, _) in Behaviour.WEIGHTS) {
                assertEquals("$name of $sender", expected.getDouble(name), features[name]!!, 1e-12)
            }
            val (score, tags) = Behaviour.smsScore(features)
            assertEquals("score of $sender", case.getDouble("score"), score, 1e-12)
            assertEquals("tags of $sender", strings(case.getJSONArray("tags")), tags)
        }
    }

    @Test
    fun actEvidence() {
        for (case in cases("evidence_cases")) {
            val text = case.getString("text")
            assertEquals(text.take(50), case.getDouble("evidence"), Acts.evidenceRisk(text), 1e-12)
        }
    }

    @Test
    fun wholeVerdicts() {
        for (case in cases("scan_cases")) {
            val sender = case.getString("sender")
            val expected = case.getJSONObject("expected")
            val got = LocalScan.scan(
                sender = sender,
                body = case.getString("text"),
                hour = case.getInt("hour"),
                firstContact = case.getBoolean("first_contact"),
            )

            assertEquals("verdict for $sender", expected.getString("verdict"), got.getString("verdict"))
            assertEquals("label for $sender", expected.getString("label"), got.getString("label"))
            assertEquals("terms for $sender",
                strings(expected.getJSONArray("terms")), strings(got.getJSONArray("terms")))
            assertEquals("reason for $sender",
                expected.getString("reason"), got.getString("reason"))

            // Everything api.py rounds to three places. One ulp of that rounding
            // is tolerated and nothing more: a real porting error is orders of
            // magnitude bigger than the last digit.
            for (key in listOf("risk", "confidence", "entropy", "label_prob", "evidence")) {
                assertEquals("$key for $sender",
                    expected.getDouble(key), got.getDouble(key), 1.1e-3)
            }
            for (group in listOf("classes", "parts", "weights")) {
                val want = expected.getJSONObject(group)
                val have = got.getJSONObject(group)
                assertEquals("$group keys for $sender", want.keys().asSequence().toSet(),
                    have.keys().asSequence().toSet())
                for (key in want.keys()) {
                    assertEquals("$group.$key for $sender",
                        want.getDouble(key), have.getDouble(key), 1.1e-3)
                }
            }

            val wantLinks = expected.getJSONArray("links")
            val haveLinks = got.getJSONArray("links")
            assertEquals("link count for $sender", wantLinks.length(), haveLinks.length())
            for (i in 0 until wantLinks.length()) {
                assertEquals(wantLinks.getJSONObject(i).getString("url"),
                    haveLinks.getJSONObject(i).getString("url"))
                assertEquals("link score for $sender",
                    wantLinks.getJSONObject(i).getDouble("score"),
                    haveLinks.getJSONObject(i).getDouble("score"), 1.1e-3)
                assertEquals("link why for $sender",
                    strings(wantLinks.getJSONObject(i).optJSONArray("why")),
                    strings(haveLinks.getJSONObject(i).optJSONArray("why")))
            }
        }
    }

    /**
     * One message, four headers, one verdict.
     *
     * This is the invariant the de-escalation gate broke while it was armed: the
     * identical text read SAFE at risk 0.300 from VM-HDFCBK-S and SUSPICIOUS at
     * 0.611 from AX-ICICIT-S, because a 204-row file names ICICIB and not
     * ICICIT. The 0.300 was `cap`, not an assessment of anything.
     *
     * Membership of an incomplete registry must not decide a verdict, so the
     * rule is stated as a test rather than left to a comment. These four headers
     * are all DLT-shaped, so every behaviour feature is identical and only the
     * registry lookup could ever have separated them.
     */
    @Test
    fun theHeaderDoesNotDecideTheVerdictOfAnIdenticalMessage() {
        val body = "Dear Customer, your KYC verification is complete. No action needed."
        val scans = listOf("VM-HDFCBK-S", "AX-ICICIT-S", "HDFCBK", "AD-SWIGGY-S")
            .associateWith { LocalScan.scan(it, body, hour = 14, firstContact = false) }

        val verdicts = scans.mapValues { it.value.getString("verdict") }
        val risks = scans.mapValues { it.value.getDouble("risk") }
        assertEquals("one verdict for one message: $verdicts", 1, verdicts.values.toSet().size)
        assertEquals("one risk for one message: $risks", 1, risks.values.toSet().size)
        for ((header, scan) in scans) {
            assertTrue("$header must not be de-escalated",
                !scan.getString("reason").contains("de-escalated"))
        }
    }

    /** The phone's own wording for the case api.py cannot have: an inbox it was
     * not allowed to read. Both first-contact rules must abstain, not fire. */
    @Test
    fun unreadableInboxAbstains() {
        val body = "Your KYC is pending, verify at http://kyc-verify.paytm.login.tk"
        val unknown = LocalScan.scan("9876543210", body, hour = 14, firstContact = null)
        val first = LocalScan.scan("9876543210", body, hour = 14, firstContact = true)

        assertTrue(unknown.getString("reason").contains("sender history not checked"))
        // first_contact (0.10) + url_first_contact (0.25) are the difference.
        assertEquals(0.35,
            first.getJSONObject("parts").getDouble("behaviour") -
                unknown.getJSONObject("parts").getDouble("behaviour"), 1e-9)
    }

    /** burst_1h is a phone-only input: api.py has no field for it, because no
     * caller of the web form knows how many messages a sender sent this hour. */
    @Test
    fun burstIsCountedFromTheInbox() {
        val body = "Running late, see you at 8"
        val calm = LocalScan.scan("VM-HDFCBK", body, 14, firstContact = false, burst = 0)
        val flood = LocalScan.scan("VM-HDFCBK", body, 14, firstContact = false, burst = 6)
        assertEquals(0.0, calm.getJSONObject("parts").getDouble("behaviour"), 1e-9)
        // min(6/5, 1) * 0.10, capped at one.
        assertEquals(0.10, flood.getJSONObject("parts").getDouble("behaviour"), 1e-9)
        assertTrue(flood.getString("reason").contains("burst of messages from this sender"))
    }
}
