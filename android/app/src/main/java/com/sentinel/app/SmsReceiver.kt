package com.sentinel.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony

// Fires even when the app is closed. Per PROJECT_SPEC.md §3/§10 this is where
// ingestion starts: BroadcastReceiver -> on-device model.
// ponytail: routes into the same in-memory MessageStore the UI reads, so a
// message captured while the app was backgrounded still shows up once it's
// reopened (as long as the process wasn't killed) — no offline queue yet.
class SmsReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        // One broadcast is one message, even when the network split it. An SMS
        // over 160 characters arrives as several parts, and scoring the parts
        // separately is how a long scam gets through: each half reads harmlessly
        // on its own while the lure lives in the whole. Observed on a digital
        // arrest text whose tail, "ing or a warrant will be issued.", came back
        // SAFE at 0.224. Joining also stops one SMS counting twice in §7's
        // sender profile and taking two slots in the phone's feed.
        val parts = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        if (parts.isNullOrEmpty()) return

        val message = MessageStore.add(
            // Every part carries the same sender and timestamp; the first is as
            // good as any, and the body is the parts in order.
            parts[0].originatingAddress ?: "unknown",
            parts.joinToString("") { it.messageBody ?: "" },
            parts[0].timestampMillis,
        )
        // Already on the page: the inbox sync got there first, or the broadcast
        // arrived twice. Nothing to scan, and nothing to send.
        if (message == null) return
        // goAsync keeps the process alive past onReceive returning — without it
        // Android is free to kill it the moment this method ends, and a
        // cold-start SMS could be scanned nowhere. finish() is called once the
        // scan has finished, not merely started.
        val pending = goAsync()

        Classifier.classify(context, message) { pending.finish() }
    }
}
