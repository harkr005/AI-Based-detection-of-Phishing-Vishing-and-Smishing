package com.sentinel.app

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

// This is what stands between a user's bank OTP and a shared report, so it
// gets a test with real message shapes rather than a smoke check.
class RedactTest {

    @Test
    fun `an OTP never survives redaction`() {
        val out = Redact.apply("481920 is your OTP for SBI netbanking. Do not share it.")
        assertFalse("the code itself must be gone", out.contains("481920"))
        assertTrue("but the message still reads as an OTP", out.contains("[code]"))
        assertTrue(out.contains("your OTP for SBI netbanking"))
    }

    @Test
    fun `account numbers, PAN, Aadhaar and amounts are all named, not deleted`() {
        val out = Redact.apply(
            "Rs 25,000 debited from A/c 912345678901 to ABCDE1234F. Aadhaar 1234 5678 9012."
        )
        assertFalse(out.contains("25,000"))
        assertFalse(out.contains("912345678901"))
        assertFalse(out.contains("ABCDE1234F"))
        assertFalse(out.contains("1234 5678 9012"))
        // The shape is the detection signal, so it has to survive.
        assertTrue(out.contains("[amount]"))
        assertTrue(out.contains("[pan]"))
        assertTrue(out.contains("[aadhaar]"))
        assertTrue(out.contains("debited from A/c"))
    }

    @Test
    fun `emails go`() {
        assertTrue(Redact.apply("Write to fraud@sbi.co.in now").contains("[email]"))
    }

    @Test
    fun `UPI ids go`() {
        assertEquals("[amount] credited from [upi] via UPI",
                     Redact.apply("Rs 500 credited from mohak.k@oksbi via UPI"))
    }

    @Test
    fun `a scam URL is evidence, not a secret, and survives intact`() {
        // url_model scores the host and path; redacting digits inside it would
        // break the features that catch the scam.
        val url = "http://sbi-kyc-verify123.tk/update?id=99887766"
        val out = Redact.apply("URGENT: verify at $url before 8888 hrs")
        assertTrue("the link must be untouched", out.contains(url))
        assertTrue("digits outside it must not be", out.contains("[code]"))
    }

    @Test
    fun `a message with nothing private is returned unchanged`() {
        val plain = "Hey, are we still on for lunch tomorrow?"
        assertEquals(plain, Redact.apply(plain))
    }

    /**
     * The model explains RAW text, so a term can be the OTP itself. The audit
     * report is shared from the phone, so its terms must be redacted too.
     */
    @Test
    fun `the report's highlighted words are redacted, the link kept, the original untouched`() {
        val result = JSONObject(
            """{"verdict":"PHISHING","terms":["481920","account","912345678901"],
                "reason":"smishing - driven by \"481920\", \"account\"",
                "links":[{"url":"http://fake-bank.tk/x","score":0.99}]}"""
        )
        val out = Redact.detail(result)
        val terms = out.getJSONArray("terms")
        assertEquals("[code]", terms.getString(0))
        assertEquals("account", terms.getString(1))
        assertEquals("[account]", terms.getString(2))
        assertFalse("the reason quotes the same words", out.getString("reason").contains("481920"))
        assertTrue("the link is the evidence", out.has("links"))
        assertEquals("PHISHING", out.getString("verdict"))
        assertEquals("the UI's copy must not change", "481920",
                     result.getJSONArray("terms").getString(0))
    }
}
