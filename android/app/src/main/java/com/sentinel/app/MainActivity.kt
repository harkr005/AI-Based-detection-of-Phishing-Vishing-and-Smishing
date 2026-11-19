package com.sentinel.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.os.Bundle
import android.util.Log
import android.os.Handler
import android.os.Looper
import android.provider.Telephony
import android.text.format.DateUtils
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.graphics.ColorUtils
import androidx.drawerlayout.widget.DrawerLayout
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.navigation.NavigationView
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        // Ask once per process, not on every recreate (rotation, etc.) — the
        // system dialog re-appearing on a rotation would be a real bug, not a
        // helpful reminder.
        private var permissionRequested = false

        /** How much of the existing inbox a fresh install scans and shows. */
        private const val BACKFILL = 30
    }

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var toolbar: MaterialToolbar
    private lateinit var navView: NavigationView
    private lateinit var settingsPanel: View
    private lateinit var messagesPanel: FrameLayout
    private lateinit var emptyState: TextView
    private lateinit var statusText: TextView
    private lateinit var messageList: LinearLayout
    private lateinit var swipeRefresh: SwipeRefreshLayout
    private var showingSettings = false

    private val pool = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    // Incoming SMS is captured by the manifest-registered SmsReceiver (it works
    // even when this Activity isn't open); this just repaints when that, or the
    // initial inbox load below, changes the shared MessageStore.
    private val onStoreChanged = { renderList() }

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            if (result.values.all { it }) {
                statusText.text = "Permission granted"
                MessageStore.afterOpen { syncInbox() }
            } else {
                statusText.text = "Permission denied — SMS scanning needs it to work"
                explainSmsBlock()
            }
        }

    /**
     * Android 13+ blocks SMS for an app installed from an APK file rather than
     * a store, until the user flips "Allow restricted settings" in App info —
     * and on newer builds the permission prompt is refused without ever being
     * shown, so the user cannot tell this apart from tapping Deny. Nothing an
     * app can call lifts it; this only points at the switch. The same App info
     * screen is also the only way back after an ordinary "Don't allow", so the
     * dialog is right in both cases.
     */
    private fun explainSmsBlock() {
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.TIRAMISU) return
        MaterialAlertDialogBuilder(this)
            .setTitle("Allow SMS access")
            .setMessage(
                "Android blocks SMS access for apps installed from an APK file.\n\n" +
                    "1. Tap Open App info\n" +
                    "2. Tap ⋮ (top right) → Allow restricted settings\n" +
                    "3. Permissions → SMS → Allow\n\n" +
                    "Then come back here."
            )
            .setPositiveButton("Open App info") { _, _ ->
                startActivity(
                    Intent(
                        android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        android.net.Uri.fromParts("package", packageName, null)
                    )
                )
            }
            .setNegativeButton("Not now", null)
            .show()
    }

    // Without this every recreation (rotation, theme change) strands a live
    // non-daemon thread for the lifetime of the process.
    override fun onDestroy() {
        pool.shutdown()
        super.onDestroy()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)
        // Hydrates the message list from SQLite, and seeds the worked examples
        // only if there is genuinely nothing there.
        MessageStore.open(this)

        drawerLayout = findViewById(R.id.drawerLayout)
        toolbar = findViewById(R.id.toolbar)
        navView = findViewById(R.id.navView)
        settingsPanel = findViewById(R.id.settingsPanel)
        messagesPanel = findViewById(R.id.messagesPanel)
        // The website's own red-on-black wash, behind the whole screen
        // rather than just the list: every panel above it is transparent.
        findViewById<View>(R.id.root).background = AmbientBackground()
        emptyState = findViewById(R.id.emptyState)
        statusText = findViewById(R.id.statusText)
        messageList = findViewById(R.id.messageList)
        swipeRefresh = findViewById(R.id.swipeRefresh)
        swipeRefresh.setColorSchemeColors(ContextCompat.getColor(this, R.color.accent))
        swipeRefresh.setOnRefreshListener { refreshNow() }

        setSupportActionBar(toolbar)
        val toggle = ActionBarDrawerToggle(
            this, drawerLayout, toolbar, R.string.open_drawer, R.string.close_drawer
        )
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState()

        navView.setNavigationItemSelectedListener { item ->
            showSettings(item.itemId == R.id.nav_settings)
            drawerLayout.closeDrawers()
            true
        }
        navView.setCheckedItem(R.id.nav_messages)
        showSettings(showingSettings)

        findViewById<Button>(R.id.grantButton).setOnClickListener {
            requestPermissions.launch(
                arrayOf(Manifest.permission.READ_SMS, Manifest.permission.RECEIVE_SMS)
            )
        }

        if (hasPermission()) {
            statusText.text = "Permission granted"
            // The inbox read itself belongs to onResume, which runs next:
            // doing it here as well would only be the same sweep twice.
        } else if (!permissionRequested) {
            // Ask immediately on open, like the system's own "Allow X to..."
            // prompts — no button tap needed first. The button in Settings
            // stays as a manual retry for anyone who dismissed it undecided.
            permissionRequested = true
            requestPermissions.launch(
                arrayOf(Manifest.permission.READ_SMS, Manifest.permission.RECEIVE_SMS)
            )
        }
    }

    override fun onResume() {
        super.onResume()
        MessageStore.addListener(onStoreChanged)
        renderAudit()
        // Granted from App info (see explainSmsBlock) returns here, not to
        // the permission callback, so the status line is re-read on the way in.
        if (hasPermission()) statusText.text = "Permission granted"
        // The broadcast is not a guarantee, and treating it as one is what let
        // messages go missing: an app the system has STOPPED receives no
        // SMS_RECEIVED at all until it is launched again (swiping it out of
        // recents is enough on several OEMs), and a process killed mid-scan
        // loses the one it was holding. The provider still has the message
        // either way, so re-reading the inbox on every open is what makes one
        // eventually arrive rather than never.
        MessageStore.afterOpen { syncInbox() }
        renderList()
    }

    /**
     * The same sweep [onResume] does, asked for out loud by pulling the list
     * down.
     */
    private fun refreshNow() {
        if (!hasPermission()) {
            swipeRefresh.isRefreshing = false
            Toast.makeText(this, "Checking needs the SMS permission.", Toast.LENGTH_SHORT).show()
            return
        }
        MessageStore.afterOpen {
            syncInbox { added ->
                swipeRefresh.isRefreshing = false
                Toast.makeText(
                    this,
                    when (added) {
                        0 -> "No new messages"
                        1 -> "1 message was missing — scanning it now"
                        else -> "$added messages were missing — scanning them now"
                    },
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }
    }

    private fun renderAudit() {
        findViewById<Button>(R.id.auditButton).setOnClickListener { runAudit() }
        findViewById<Button>(R.id.shareButton).apply {
            // Stateless: the report either exists on disk or it does not, so
            // there is nothing to remember across a restart.
            visibility = if (auditReport().exists()) View.VISIBLE else View.GONE
            setOnClickListener { shareReport() }
        }
    }

    /**
     * One pass over the inbox, to find what the models get WRONG on real
     * traffic. A public SMS corpus has almost no legitimate automated bank
     * messages in it, so this is the only place the false positives live.
     *
     * Runs on [pool], never the main thread: a thousand messages is a thousand
     * model scores. The button is disabled while it runs rather than guarded by
     * a flag, so a second tap is impossible instead of merely ignored.
     */
    private fun runAudit() {
        val button = findViewById<Button>(R.id.auditButton)
        val status = findViewById<TextView>(R.id.auditText)
        if (!hasPermission()) {
            status.text = "Reading your inbox needs the SMS permission."
            return
        }
        button.isEnabled = false
        // Audit.worker, not this Activity's pool: see Audit.worker.
        Audit.worker.execute {
            try {
                val result = Audit.run(this, onProgress = { done, total ->
                    runOnUiThread { status.text = "Checking $done of $total…" }
                })
                val counts = result.byVerdict.entries.joinToString(", ") { (v, n) ->
                    "${bandCopy(v).label.lowercase()} $n"
                }
                runOnUiThread {
                    status.text = "${result.scanned} messages checked — $counts.\n" +
                        "Codes, amounts and account numbers are removed from the report."
                    findViewById<Button>(R.id.shareButton).visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "inbox audit failed", e)
                runOnUiThread { status.text = "Could not check the inbox: ${e.message}" }
            } finally {
                runOnUiThread { button.isEnabled = true }
            }
        }
    }

    private fun auditReport() = java.io.File(getExternalFilesDir(null), Audit.REPORT)

    /**
     * The only way this file leaves a phone that is not plugged into a laptop:
     * from Android 11 on, no file manager can browse `Android/data`. A friend
     * taps this and picks WhatsApp or Gmail — no adb, no developer options.
     *
     * The chosen app gets a one-shot read grant through the FileProvider and
     * nothing else can open it.
     */
    private fun shareReport() {
        val report = auditReport()
        if (!report.exists()) {
            Toast.makeText(this, "Check the inbox first.", Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.files", report)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "Sentinel inbox report")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, "Send the report"))
    }

    override fun onPause() {
        super.onPause()
        MessageStore.removeListener(onStoreChanged)
    }

    private fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS) ==
            PackageManager.PERMISSION_GRANTED

    /**
     * Reconcile the page against `content://sms` — the last [BACKFILL] messages
     * on every open, with [MessageStore.add] dropping the ones already held.
     *
     * This used to run once EVER, which left the broadcast receiver as the only
     * way a message could ever appear. It is not reliable enough for that: an
     * app the system has stopped gets no broadcast until it is launched again,
     * which is why reinstalling "fixed" it and why the next SMS went missing
     * too. The provider, unlike the broadcast, still has the message.
     *
     * Not to be confused with [Audit], which reads a thousand and writes a
     * report without touching this list. This one is the page; that one is the
     * data collection.
     *
     * ponytail: no paging and no watermark — thirty rows re-read and deduped
     * beats a "last seen" timestamp that the two clocks (see
     * [MessageStore.add]) would disagree about anyway. Raise [BACKFILL] if a
     * phone can go long enough between opens to miss more than thirty.
     *
     * @param onDone how many were new, delivered on the main thread.
     */
    private fun syncInbox(onDone: ((Int) -> Unit)? = null) {
        if (!hasPermission()) {
            main.post { onDone?.invoke(0) }
            return
        }
        pool.execute {
            var added = 0
            runCatching {
                contentResolver.query(
                    Telephony.Sms.Inbox.CONTENT_URI,
                    arrayOf(Telephony.Sms.ADDRESS, Telephony.Sms.BODY, Telephony.Sms.DATE),
                    null, null,
                    "${Telephony.Sms.DATE} DESC LIMIT $BACKFILL",
                )?.use { cursor ->
                    val addressCol = cursor.getColumnIndexOrThrow(Telephony.Sms.ADDRESS)
                    val bodyCol = cursor.getColumnIndexOrThrow(Telephony.Sms.BODY)
                    val dateCol = cursor.getColumnIndexOrThrow(Telephony.Sms.DATE)
                    while (cursor.moveToNext()) {
                        val message = MessageStore.add(
                            cursor.getString(addressCol) ?: "unknown",
                            cursor.getString(bodyCol) ?: "",
                            cursor.getLong(dateCol),
                        ) ?: continue
                        added++
                        Classifier.classify(this, message)
                    }
                }
            }.onFailure { Log.e("MainActivity", "could not read the inbox", it) }
            main.post { onDone?.invoke(added) }
        }
    }

    // Two destinations now, so this is a boolean rather than a section name:
    // every message lives on one page, ordered by time.
    private fun showSettings(settings: Boolean) {
        showingSettings = settings
        toolbar.title = if (settings) "Settings" else "Messages"
        settingsPanel.visibility = if (settings) View.VISIBLE else View.GONE
        messagesPanel.visibility = if (settings) View.GONE else View.VISIBLE
        if (!settings) renderList()
    }

    private fun renderList() {
        if (showingSettings) return
        messageList.removeAllViews()
        val messages = MessageStore.all()
        emptyState.visibility = if (messages.isEmpty()) View.VISIBLE else View.GONE

        val inflater = LayoutInflater.from(this)
        for (message in messages) {
            val row = inflater.inflate(R.layout.item_message, messageList, false)
            row.findViewById<TextView>(R.id.rowSender).text = message.sender
            row.findViewById<TextView>(R.id.rowSnippet).text = message.body
            // The device's own record of when the SMS arrived — the same clock
            // the list is sorted by, so the order on screen is explained by what
            // is printed on it. DateUtils handles "5 minutes ago" vs "21 Sep"
            // and localises both.
            row.findViewById<TextView>(R.id.rowTime).text =
                DateUtils.getRelativeTimeSpanString(
                    message.sentAt, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS
                )

            // A failed scan is not a verdict, so it keeps the neutral colour
            // bandColorRes gives an unknown one rather than borrowing a band it
            // never earned.
            val verdict = if (message.failed) null else message.result?.optString("verdict")
            val color =
                ContextCompat.getColor(this, rowBadgeColorRes(verdict, message.category))
            row.findViewById<TextView>(R.id.rowStatus).apply {
                text = when {
                    message.failed -> "Scan failed"
                    verdict == null -> "Checking…"
                    else -> rowBadge(verdict, message.category).label
                }
                setTextColor(color)
                backgroundTintList =
                    ColorStateList.valueOf(ColorUtils.setAlphaComponent(color, 38))
            }

            row.findViewById<Button>(R.id.rowView).setOnClickListener {
                startActivity(
                    Intent(this, MessageDetailActivity::class.java)
                        .putExtra(EXTRA_MESSAGE_ID, message.id)
                )
            }
            messageList.addView(row)
        }
    }
}
