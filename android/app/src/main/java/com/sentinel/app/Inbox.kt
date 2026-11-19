package com.sentinel.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.Telephony
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * The phone's own message history, read from `content://sms` — §7's two
 * stateful SMS rules and nothing else.
 *
 * This replaces asking [MessageStore], which could only answer "have WE scanned
 * this sender before", not "have you heard from them before". On a fresh
 * install those differ for every sender alive: the app's table is empty, so
 * every message read as a first contact and armed first_contact (0.10) plus
 * url_first_contact (0.25) on ordinary mail. The inbox is the real history.
 *
 * No sender-profile table ships on the phone. These two numbers are the whole
 * of §7's state here.
 */
object Inbox {

    /**
     * @param priorCount messages from this sender strictly BEFORE [before] —
     * strictly, because the live message may not have been written to the
     * provider yet (this app is not the default SMS app, so the system writes
     * it on its own schedule) and "has it landed yet" must not change a verdict.
     * @param burst how many of those arrived in the preceding hour.
     */
    data class History(val priorCount: Int, val burst: Int) {
        val firstContact: Boolean get() = priorCount == 0
    }

    /**
     * -> null when the inbox cannot be read at all, which is NOT "no history".
     * `behaviour.known_sender` abstains for exactly this reason: absence from a
     * store we could not open is not evidence of a first contact, and
     * url_first_contact carries 0.25 on that claim.
     */
    fun history(context: Context, sender: String, before: Long): History? {
        if (sender.isEmpty()) return null
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_SMS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            return null
        }
        return try {
            // ponytail: one query answers both rules. 200 rows is plenty — a
            // sender past 5 in an hour already pins burst_1h at 1.0, and for
            // first contact only "any at all" matters.
            context.contentResolver.query(
                Telephony.Sms.CONTENT_URI,
                arrayOf(Telephony.Sms.DATE),
                "${Telephony.Sms.ADDRESS} = ? AND ${Telephony.Sms.DATE} < ?",
                arrayOf(sender, before.toString()),
                "${Telephony.Sms.DATE} DESC LIMIT 200",
            )?.use { cursor ->
                val date = cursor.getColumnIndexOrThrow(Telephony.Sms.DATE)
                var burst = 0
                while (cursor.moveToNext()) {
                    if (cursor.getLong(date) > before - 3_600_000L) burst++
                }
                History(cursor.count, burst)
            }
        } catch (e: Exception) {
            // A provider that throws is the same state as one we may not read.
            Log.w("Inbox", "could not read sender history", e)
            null
        }
    }
}
