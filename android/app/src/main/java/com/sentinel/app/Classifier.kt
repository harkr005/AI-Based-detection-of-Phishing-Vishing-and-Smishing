package com.sentinel.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.Calendar
import java.util.concurrent.Executors

// Scores a captured message on this phone. Runs off the main thread; posts the
// result back onto it. Nothing about the message, not even its verdict, leaves
// the phone.
object Classifier {
    private val pool = Executors.newFixedThreadPool(2)
    private val mainHandler = Handler(Looper.getMainLooper())

    /** @param onDone run once scoring has finished, whether it worked or not. */
    fun classify(context: Context, message: ScannedMessage, onDone: (() -> Unit)? = null) {
        val app = context.applicationContext
        pool.execute {
            try {
                LocalScan.ensureLoaded(app)
                // The SMSC's own timestamp, in the phone's zone: the hour the
                // message was SENT. The scan clock is not the message clock.
                val hour = Calendar.getInstance()
                    .apply { timeInMillis = message.sentAt }
                    .get(Calendar.HOUR_OF_DAY)
                // Null when the inbox cannot be read, and then §7's two
                // first-contact rules abstain rather than accuse every sender
                // on a fresh install of being new.
                val history = Inbox.history(app, message.sender, message.sentAt)
                val result = LocalScan.scan(
                    sender = message.sender,
                    body = message.body,
                    hour = hour,
                    firstContact = history?.firstContact,
                    burst = history?.burst ?: 0,
                )
                val label = result.optString("label", "")
                mainHandler.post {
                    MessageStore.setResult(message, categoryFor(label), result)
                }
            } catch (e: Exception) {
                Log.e("Classifier", "scan failed for message ${message.id}", e)
                mainHandler.post {
                    MessageStore.setResult(message, CATEGORY_ERROR, null)
                }
            } finally {
                onDone?.invoke()
            }
        }
    }
}
