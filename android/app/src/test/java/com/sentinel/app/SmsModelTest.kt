package com.sentinel.app

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The port is only worth having if it IS the model, so this compares it against
 * sklearn's own probabilities on every case in `sms_parity.json` — authored
 * edge cases (short words, NFKC, zero-width, homoglyphs, all-OOV text) plus 30
 * random rows from the real corpus, because cases written by the same person
 * who wrote the port only measure that person.
 *
 * Regenerate both files with `python export_sms_model.py` after retraining.
 */
class SmsModelTest {

    companion object {
        // Gradle runs unit tests with the module directory as the working dir.
        private val ASSET = File("src/main/assets/sms_charlr.bin")
        private val FIXTURE = File("src/test/resources/sms_parity.json")

        @BeforeClass
        @JvmStatic
        fun load() {
            assertTrue("run `python export_sms_model.py`", ASSET.exists())
            ASSET.inputStream().use { SmsModel.load(it) }
        }
    }

    @Test
    fun matchesSklearn() {
        val fixture = JSONObject(FIXTURE.readText())
        assertEquals(SmsModel.LABELS, fixture.getJSONArray("labels").let {
            (0 until it.length()).map(it::getString)
        })
        val cases = fixture.getJSONArray("cases")
        var worst = 0.0
        var worstText = ""
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val text = case.getString("text")
            val expected = case.getJSONArray("probs")
            val got = SmsModel.score(text)
            for (c in got.indices) {
                val diff = abs(got[c] - expected.getDouble(c))
                if (diff > worst) { worst = diff; worstText = text }
            }
        }
        assertTrue(cases.length() >= 20)
        // 1e-4 is the agreed bar. float32 weights and a ~200-term dot product
        // land around 1e-7, so anything near this bound is a porting bug, not
        // arithmetic — and the message it failed on is the one to look at.
        assertTrue("worst diff $worst on ${worstText.take(60)}", worst < 1e-4)
    }

    /**
     * §4 requires a reason on every verdict, and the words are the same
     * evidence as data — the UI shows them as chips. Exact attribution over the
     * n-grams, so this has to agree with Python word for word, not approximately.
     */
    @Test
    fun explainsLikeSklearn() {
        val cases = JSONObject(FIXTURE.readText()).getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val text = case.getString("text")
            val got = SmsModel.explain(text)
            assertEquals("reason for ${text.take(40)}", case.getString("reason"), got.reason)
            val want = case.getJSONArray("terms")
            assertEquals("terms for ${text.take(40)}",
                (0 until want.length()).map(want::getString), got.terms)
        }
    }

    /** The two ends of the model, stated outright so a silently empty asset
     * (all-zero weights still normalise to 0.333) cannot pass the test above. */
    @Test
    fun classifiesTheObviousCases() {
        val smishing = SmsModel.score(
            "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked")
        val ham = SmsModel.score("Hey, are we still meeting at 6 for dinner?")
        assertTrue("smishing ${smishing.toList()}", smishing[2] > 0.5f)
        assertTrue("ham ${ham.toList()}", ham[0] > 0.5f)
    }

    /** §9.9: obfuscation normalize() can undo must not change the verdict.
     * This is the defence that bought 47 points of recall back on the SMS
     * model, and it lives in the port now, not in the server. */
    @Test
    fun obfuscationIsNormalisedAway() {
        val plain = SmsModel.score("Your account is blocked, verify now")
        val attacked = SmsModel.score("Yоur аcc​ount is blоcked, ver​ify now")
        for (c in plain.indices) assertEquals(plain[c].toDouble(), attacked[c].toDouble(), 1e-6)
    }

    /** No in-vocabulary n-gram means a zero row, and a zero row must not be
     * divided by its own norm. NaN here would reach fusion as a verdict. */
    @Test
    fun allOutOfVocabularyTextIsFinite() {
        for (p in SmsModel.score("🚨🚨🚨")) {
            assertTrue(p.isFinite())
        }
    }
}
