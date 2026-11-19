package com.sentinel.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.Telephony
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.File
import java.util.Calendar

/**
 * Score the whole inbox in one pass and write a report, so the false positives
 * can be READ instead of waited for.
 *
 * The models have never seen a legitimate automated bank SMS — one of 5,306 ham
 * rows in the training corpus is a genuine automated message — so the words that
 * ought to reassure ("complete", "no action needed") were learned as scam
 * vocabulary. The only cure is examples, and a real inbox is full of them.
 *
 * Deliberately NOT part of scanning:
 *   - nothing is uploaded; the app has no network access at all.
 *   - nothing is written to [MessageStore], so a 1,000-message audit does not
 *     bury the user's actual message list.
 *
 * **Message text is REDACTED in the report**, through [Redact]. This file is meant to be sent to whoever is collecting the
 * data, and a false positive stays perfectly diagnosable without the digits:
 * what misfires is "Dear Customer, your KYC verification is complete", and
 * "[amount] debited from A/c [account]" reads the same to a human and to the
 * model, which was measured on redacted text at a cost of 0.006-0.013 macro-F1.
 * Nobody needs to mail their OTPs to a classmate to help with a project.
 *
 * It is written to the app's own external files directory, which from Android
 * 11 on NO file manager can browse — so the share sheet is how it leaves the
 * phone, and adb is the developer's way:
 *
 *     adb pull /sdcard/Android/data/com.sentinel.app/files/audit.jsonl
 *
 * Raw text never leaves: it stays in Room, on the phone, for whoever owns it.
 */
object Audit {

    const val REPORT = "audit.jsonl"

    /**
     * The audit's own thread, deliberately NOT the Activity's.
     *
     * A thousand real messages is about a minute of scoring, and an Activity
     * does not survive a minute reliably — a rotation, an incoming call or a
     * screen timeout destroys it, and its executor is shut down with it. The
     * report has to finish being written whatever the screen does; the UI
     * callbacks are the part that is allowed to be lost.
     */
    val worker: java.util.concurrent.ExecutorService =
        java.util.concurrent.Executors.newSingleThreadExecutor()

    data class Result(val scanned: Int, val byVerdict: Map<String, Int>, val report: File)

    private data class Row(val sender: String, val body: String, val sentAt: Long)

    /**
     * @param limit newest N messages. 500-1000 is a few seconds and covers
     * months of real traffic.
     * @param onProgress called off the main thread, roughly every 25 messages.
     */
    fun run(
        context: Context,
        limit: Int = 1000,
        onProgress: (Int, Int) -> Unit = { _, _ -> },
    ): Result {
        val app = context.applicationContext
        require(
            ContextCompat.checkSelfPermission(app, Manifest.permission.READ_SMS)
                == PackageManager.PERMISSION_GRANTED,
        ) { "READ_SMS is needed to audit the inbox" }

        LocalScan.ensureLoaded(app)
        val rows = read(app, limit)

        // History is computed from this window rather than by re-querying per
        // message: one pass over the inbox instead of a thousand provider
        // round-trips, and it makes first_contact/burst self-consistent with
        // what the audit actually looked at. Oldest first, so "prior" means
        // prior.
        val seen = HashMap<String, MutableList<Long>>()
        val report = File(app.getExternalFilesDir(null), REPORT)
        val byVerdict = LinkedHashMap<String, Int>()

        report.bufferedWriter().use { out ->
            rows.asReversed().forEachIndexed { i, row ->
                val before = seen.getOrPut(row.sender) { mutableListOf() }
                val hour = Calendar.getInstance()
                    .apply { timeInMillis = row.sentAt }.get(Calendar.HOUR_OF_DAY)
                val result = LocalScan.scan(
                    sender = row.sender,
                    body = row.body,
                    hour = hour,
                    firstContact = before.isEmpty(),
                    burst = before.count { it > row.sentAt - 3_600_000L },
                )
                before += row.sentAt

                val verdict = result.optString("verdict")
                byVerdict[verdict] = (byVerdict[verdict] ?: 0) + 1
                out.write(
                    JSONObject()
                        .put("sender", row.sender)
                        .put("sent_at", row.sentAt)
                        // Redacted, not raw: see the note above. The whole
                        // point of this file is that it can be sent to someone.
                        .put("body", Redact.apply(row.body))
                        // Left empty for the reader to fill in: "ok" when the
                        // verdict is right, anything else when it is not. A
                        // labelled file is the whole point of the exercise.
                        .put("correct", "")
                        // `terms` are words lifted from the RAW text and can
                        // be the OTP itself, which would undo redacting `body`.
                        .put("result", Redact.detail(result))
                        .toString(),
                )
                out.newLine()
                if (i % 25 == 0) onProgress(i, rows.size)
            }
        }
        onProgress(rows.size, rows.size)
        return Result(rows.size, byVerdict, report)
    }

    private fun read(context: Context, limit: Int): List<Row> {
        val rows = mutableListOf<Row>()
        context.contentResolver.query(
            Telephony.Sms.Inbox.CONTENT_URI,
            arrayOf(Telephony.Sms.ADDRESS, Telephony.Sms.BODY, Telephony.Sms.DATE),
            null, null,
            "${Telephony.Sms.DATE} DESC LIMIT $limit",
        )?.use { cursor ->
            val address = cursor.getColumnIndexOrThrow(Telephony.Sms.ADDRESS)
            val body = cursor.getColumnIndexOrThrow(Telephony.Sms.BODY)
            val date = cursor.getColumnIndexOrThrow(Telephony.Sms.DATE)
            while (cursor.moveToNext()) {
                rows += Row(
                    cursor.getString(address) ?: "unknown",
                    cursor.getString(body) ?: "",
                    cursor.getLong(date),
                )
            }
        }
        return rows
    }
}
