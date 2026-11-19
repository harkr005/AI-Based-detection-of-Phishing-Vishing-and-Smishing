package com.sentinel.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * `api.py`'s `/sms` route, on the phone — §10 Mode 1.
 *
 * The SMS pipeline moves on-device because SMS is the only channel whose content
 * arrives unbidden and continuously; email already lives on Google's servers and
 * has no on-device arrival property to preserve. Nothing here re-decides
 * anything: it wires [SmsModel], [UrlModel], [Behaviour], [Acts] and [Fusion]
 * together in exactly the order the Python route does, so a verdict on the phone
 * is the verdict the dashboard would have given, and `ScanParityTest` holds it
 * to that on ten whole messages.
 *
 * The body no longer leaves the phone to be scored. §2's "content never leaves
 * the phone, only the verdict does" is true of SMS from here on — and the
 * models now see RAW text, which is worth 0.006-0.013 macro-F1 that redaction
 * was costing them.
 *
 * ONE DELIBERATE DIFFERENCE from the route, and it is a difference of INPUT,
 * not of code: `fusion.deescalate` is armed here. See the call below.
 */
object LocalScan {

    @Synchronized
    fun ensureLoaded(context: Context) {
        SmsModel.ensureLoaded(context)
        UrlModel.ensureLoaded(context)
        Behaviour.ensureLoaded(context)
    }

    val ready: Boolean get() = SmsModel.ready && UrlModel.ready

    /**
     * @param hour the hour the message was SENT, from the SMSC's own timestamp,
     * or null. Never the hour of the scan — see the odd-hour rule.
     * @param firstContact null when [Inbox] could not answer, and then both
     * first-contact rules abstain rather than assert a history nobody has.
     */
    fun scan(
        sender: String,
        body: String,
        hour: Int?,
        firstContact: Boolean?,
        burst: Int = 0,
    ): JSONObject {
        val explained = SmsModel.explain(body)
        val links = UrlModel.links(body)
        val url = links.firstOrNull()?.second

        val tags = mutableListOf<String>()
        var behaviour: Double? = null
        if (sender.isNotEmpty()) {
            val features = Behaviour.smsFeatures(
                sender, body, firstContact, hasUrl = links.isNotEmpty(),
                hour = hour, burst = burst,
            )
            val (score, fired) = Behaviour.smsScore(features)
            behaviour = score
            tags += fired
            if (firstContact == null) {
                // api.py names how many senders ITS store knows. The phone has
                // no store to count — it has an inbox it could not read.
                tags += "sender history not checked - this app cannot read your past messages"
            }
        }

        // §8 act gate. Content scores topic; the act layer scores whether
        // anything was actually asked for, so a bank's own fraud warning — which
        // is maximally on-topic and asks for nothing — gets damped back down.
        val evidence = Acts.evidenceRisk(body)
        val probs = Fusion.gateByAct(SmsModel.score(body), evidence)
        if (evidence == 0.0) tags += "message contains no request, threat or payment demand"

        val link = if (url == null) null else UrlModel.explain(links.first().first)
        val reasons = listOf(explained.reason) + listOfNotNull(link?.reason) + tags
        var fused = Fusion.fuse(probs, url, behaviour, reasons)
        // DISARMED, after being armed and then measured (2026-09-22). The phone
        // is the one caller entitled to arm it — a carrier stamped this header,
        // not a text box — but entitlement is not justification:
        //
        //   40 real DLT template SMS   39/40 SAFE with it, 39/40 SAFE without
        //   200 smishing               97.0% caught without, 91.0% with
        //
        // Zero gain, twelve real scams cleared. The act gate landed after this
        // gate was designed and already removes the false positives it was
        // built for, so it now rescues a population that no longer exists.
        //
        // It also made the verdict depend on a 204-row file: the same message
        // read SAFE from VM-HDFCBK-S and SUSPICIOUS from AX-ICICIT-S, purely
        // because the registry names one entity and not the other. The SAFE was
        // `cap`, not an assessment.
        //
        // Re-arm by passing Behaviour.registered(sender) — but only against a
        // false-positive population that actually exists.
        fused = Fusion.deescalate(fused, evidence, url, trustedSender = false)

        return JSONObject().apply {
            put("verdict", fused.verdict)
            put("risk", fused.risk)
            put("confidence", fused.confidence)
            put("entropy", fused.entropy)
            put("weights", JSONObject(fused.weights.mapValues { it.value }))
            put("parts", JSONObject(fused.parts.mapValues { it.value }))
            put("reason", fused.reason)
            put("channel", "sms")
            put("label", explained.label)
            put("label_prob", Fusion.round3(explained.probability))
            put("evidence", Fusion.round3(evidence))
            put("terms", JSONArray(explained.terms))
            put("classes", JSONObject().apply {
                SmsModel.LABELS.forEachIndexed { i, name ->
                    put(name, Fusion.round3(probs[i].toDouble()))
                }
            })
            // `why` rides on the worst link only, which is the one the verdict
            // was built from — api.py explains that one and no other.
            put("links", JSONArray(links.mapIndexed { i, (url, score) ->
                JSONObject().put("url", url).put("score", Fusion.round3(score))
                    .apply { if (i == 0 && link != null) put("why", JSONArray(link.named)) }
            }))
            // Marks the verdict as this phone's own work rather than the
            // sidecar's, for anyone reading a stored scan later.
            put("engine", "on-device")
        }
    }
}
