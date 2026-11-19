package com.sentinel.app

import org.json.JSONArray
import org.json.JSONObject

// Three worked examples, one per section, so a fresh install has something to
// look at before a single real SMS arrives.
//
// ponytail: they carry their verdict already rather than being pushed through
// Classifier — samples that need api.py running to show anything would defeat
// the point of having them. The shape is api.py's own /sms reply, so the detail
// screen renders them through exactly the same code as a real scan.
private fun sample(
    verdict: String,
    label: String,
    risk: Double,
    confidence: Double,
    entropy: Double,
    classes: Map<String, Double>,
    parts: Map<String, Double>,
    weights: Map<String, Double>,
    terms: List<String>,
    link: Triple<String, Double, List<String>>?,
    evidence: Double,
    reason: String,
): JSONObject = JSONObject().apply {
    put("verdict", verdict)
    put("label", label)
    put("risk", risk)
    put("confidence", confidence)
    put("entropy", entropy)
    put("evidence", evidence)
    put("reason", reason)
    put("classes", JSONObject(classes))
    put("parts", JSONObject(parts))
    put("weights", JSONObject(weights))
    put("terms", JSONArray(terms))
    put(
        "links",
        JSONArray().apply {
            link?.let { (url, score, why) ->
                put(
                    JSONObject()
                        .put("url", url)
                        .put("score", score)
                        .put("why", JSONArray(why)),
                )
            }
        },
    )
    put("sample", true)
}

val SAMPLE_MESSAGES: List<Triple<String, String, JSONObject>> = listOf(
    Triple(
        "HDFCBK",
        "Dear Customer, your account ending 4417 has been credited with Rs.24,500.00 on 12-09. " +
            "Available balance Rs.61,208.33. Not you? Call 18002026161.",
        sample(
            verdict = "SAFE", label = "ham", risk = 0.06, confidence = 0.97, entropy = 0.142,
            classes = mapOf("ham" to 0.97, "spam" to 0.02, "smishing" to 0.01),
            parts = mapOf("nlp" to 0.03, "behaviour" to 0.11),
            weights = mapOf("nlp" to 0.7, "behaviour" to 0.3),
            terms = listOf("credited", "available balance", "dear customer"),
            link = null, evidence = 0.0,
            reason = "Reads as a routine transaction alert and asks nothing of you.",
        ),
    ),
    Triple(
        "VM-OFFERZ",
        "MEGA SALE is LIVE! Flat 70% OFF on everything + extra 10% on prepaid orders. " +
            "Shop now: shopdealz.top/sale2 Reply STOP to opt out.",
        sample(
            verdict = "SUSPICIOUS", label = "spam", risk = 0.48, confidence = 0.88, entropy = 0.398,
            classes = mapOf("ham" to 0.05, "spam" to 0.88, "smishing" to 0.07),
            parts = mapOf("nlp" to 0.41, "url" to 0.62, "behaviour" to 0.35),
            weights = mapOf("nlp" to 0.45, "url" to 0.35, "behaviour" to 0.2),
            terms = listOf("mega sale", "flat 70% off", "shop now", "prepaid"),
            link = Triple(
                "shopdealz.top/sale2", 0.62,
                listOf("rare top-level domain", "digits in the host"),
            ),
            evidence = 0.0,
            reason = "Unwanted marketing on a cheap throwaway domain — junk, not a scam.",
        ),
    ),
    Triple(
        "+91 98213 40071",
        "URGENT: Your SBI account will be BLOCKED today. Complete KYC immediately at " +
            "http://sbi-kyc-verify.xyz/update or your funds will be frozen. Do not share this SMS.",
        sample(
            verdict = "PHISHING", label = "smishing", risk = 0.94, confidence = 0.96, entropy = 0.171,
            classes = mapOf("ham" to 0.01, "spam" to 0.03, "smishing" to 0.96),
            parts = mapOf("nlp" to 0.93, "url" to 0.97, "behaviour" to 0.81),
            weights = mapOf("nlp" to 0.4, "url" to 0.35, "behaviour" to 0.25),
            terms = listOf("blocked", "kyc", "immediately", "funds will be frozen", "urgent"),
            link = Triple(
                "http://sbi-kyc-verify.xyz/update", 0.97,
                listOf(
                    "rare top-level domain",
                    "hyphen-stuffed host",
                    "bank/payment lure words in the host",
                ),
            ),
            evidence = 1.0,
            reason = "Impersonates a bank, invents a deadline and sends you to a look-alike site.",
        ),
    ),
)
