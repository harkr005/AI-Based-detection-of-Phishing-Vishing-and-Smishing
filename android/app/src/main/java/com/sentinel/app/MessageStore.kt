package com.sentinel.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import org.json.JSONObject
import java.util.concurrent.Executors
import kotlin.math.abs

data class ScannedMessage(
    val id: Long,
    val sender: String,
    val body: String,
    val sentAt: Long, // epoch millis, the DEVICE's own record of when the SMS arrived
    var category: String = CATEGORY_PENDING,
    var result: JSONObject? = null,
) {
    /**
     * Failed, as opposed to still running. `result` is null for both, so every
     * screen that renders an unscored message has to ask this to tell them
     * apart — without it a failed scan reads as "Checking…" forever, which is
     * a worse lie than being invisible was.
     */
    val failed: Boolean get() = category == CATEGORY_ERROR
}

const val CATEGORY_PENDING = "Pending"
const val CATEGORY_SAFE = "Safe"
const val CATEGORY_SPAM = "Spam"
const val CATEGORY_SCAM = "Scam"
const val CATEGORY_ERROR = "Error"

// sms_model's own three classes (see sms_model.LABELS), said in the words the
// user asked for rather than ham/spam/smishing.
fun categoryFor(label: String): String = when (label) {
    "ham" -> CATEGORY_SAFE
    "spam" -> CATEGORY_SPAM
    "smishing" -> CATEGORY_SCAM
    else -> CATEGORY_PENDING
}

/* ---- persistence --------------------------------------------------------- */

// This holds the RAW message, because it is what the user reads on their own
// screen. It lives in the app's private data directory, the same protection
// Android's own SMS provider has — and `allowBackup` is off in the manifest
// precisely so this file cannot be copied off the device by a cloud backup.
// Nothing here is uploaded: the app has no network access at all.
@Entity(tableName = "messages", indices = [Index(value = ["hash"], unique = true)])
data class StoredMessage(
    @PrimaryKey val id: Long,
    val sender: String,
    val body: String,
    val sentAt: Long,
    val category: String,
    val resultJson: String?,
    // sender|sentAt|body. Makes re-running the inbox backfill harmless: the
    // same physical SMS produces the same hash and the insert is ignored.
    val hash: String,
)

@Dao
interface MessageDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    fun insert(message: StoredMessage)

    @Query("SELECT * FROM messages ORDER BY sentAt DESC, id DESC LIMIT :limit")
    fun recent(limit: Int): List<StoredMessage>

    @Query("UPDATE messages SET category = :category, resultJson = :resultJson WHERE id = :id")
    fun setResult(id: Long, category: String, resultJson: String?)

    @Query(
        "DELETE FROM messages WHERE id NOT IN " +
            "(SELECT id FROM messages ORDER BY sentAt DESC, id DESC LIMIT :keep)"
    )
    fun trim(keep: Int)

    @Query("DELETE FROM messages")
    fun clear()
}

/* ---- the store ----------------------------------------------------------- */

/**
 * A write-through cache over Room: the list in memory is what every screen
 * reads, and every change to it is mirrored to SQLite on a background thread.
 *
 * Deliberately NOT a set of Room queries called from the UI. `renderList` runs
 * on the main thread, where Room refuses to be queried, and the alternatives
 * were `allowMainThreadQueries()` (an anti-pattern that happens to work at
 * these sizes) or rewriting every screen around Flow. This keeps the public
 * API identical to the in-memory version it replaces, so nothing above it
 * changed.
 *
 * [dao] stays null until [open] is called, and then the store is simply
 * in-memory — which is exactly what a JVM unit test wants, and why the tests
 * need no Android runtime.
 */
object MessageStore {
    private val messages = mutableListOf<ScannedMessage>()
    private val listeners = mutableListOf<() -> Unit>()

    private var dao: MessageDao? = null
    private val io = Executors.newSingleThreadExecutor()

    // Guarded by this object's monitor; see notifyChanged.
    private var repaintQueued = false

    /** Roughly a week of a busy inbox; the oldest fall off the end. */
    private const val MAX_KEPT = 500

    /** How far apart two copies of one SMS may be timestamped — see [add]. */
    private const val SAME_MESSAGE_MS = 24 * 3_600_000L

    // Lazy and nullable: a plain JVM unit test has no Looper at all, and an
    // eager Handler here took the whole object's initialiser down with it.
    private val mainThread: Handler? by lazy {
        runCatching { Handler(Looper.getMainLooper()) }.getOrNull()
    }

    // Ids come from the clock rather than a counter, so one can be handed out
    // synchronously without first knowing what is already on disk. That removes
    // a real startup race: an SMS arriving while [open] is still reading would
    // otherwise be given an id the database had already used.
    private var lastId = 0L

    @Synchronized
    private fun newId(): Long {
        lastId = maxOf(System.currentTimeMillis(), lastId + 1)
        return lastId
    }

    /**
     * Hydrate from disk. Call once, from the Activity — everything before this
     * behaves exactly as the old in-memory store did.
     */
    fun open(context: Context) {
        if (dao != null) return
        val app = context.applicationContext
        io.execute {
            val loaded = ScanDb.get(app).messages()
            val rows = loaded.recent(MAX_KEPT)
            synchronized(this) {
                dao = loaded
                messages.clear()
                rows.forEach { row ->
                    messages += ScannedMessage(
                        id = row.id,
                        sender = row.sender,
                        body = row.body,
                        sentAt = row.sentAt,
                        category = row.category,
                        result = row.resultJson?.let { runCatching { JSONObject(it) }.getOrNull() },
                    )
                }
                lastId = maxOf(lastId, rows.maxOfOrNull { it.id } ?: 0L)
            }
            // Only on a genuinely empty store, or the worked examples would
            // come back every launch alongside the ones already kept.
            if (rows.isEmpty()) seedSamples()
            notifyChanged()

            // A process killed mid-scan left its row on "Checking…" for good:
            // nothing else ever revisits one. CATEGORY_ERROR rows come back the
            // same way: a scan that failed once is worth one more try.
            val unscored = synchronized(this) { messages.filter { it.result == null } }
            unscored.forEach { Classifier.classify(app, it) }
        }
    }

    /**
     * Runs [block] on the store's own thread — which is to say, once [open] has
     * finished hydrating, whether it is still running or ran launches ago. The
     * inbox sync needs that: deciding what is new against a list that has not
     * been read off disk yet re-adds every message on it.
     */
    fun afterOpen(block: () -> Unit) {
        io.execute(block)
    }

    private fun hashOf(sender: String, sentAt: Long, body: String) = "$sender|$sentAt|$body"

    private fun persist(message: ScannedMessage) {
        val d = dao ?: return
        val row = StoredMessage(
            id = message.id,
            sender = message.sender,
            body = message.body,
            sentAt = message.sentAt,
            category = message.category,
            resultJson = message.result?.toString(),
            hash = hashOf(message.sender, message.sentAt, message.body),
        )
        io.execute {
            d.insert(row)
            d.trim(MAX_KEPT)
        }
    }

    /** The three worked examples from Samples.kt. */
    @Synchronized
    fun seedSamples() {
        if (messages.any { it.result != null && it.category != CATEGORY_PENDING } &&
            messages.size >= SAMPLE_MESSAGES.size
        ) {
            // Already populated; samples are a first-run courtesy, not content.
            return
        }
        val now = System.currentTimeMillis()
        SAMPLE_MESSAGES.forEachIndexed { index, (sender, body, result) ->
            val message = ScannedMessage(
                id = newId(),
                sender = sender,
                body = body,
                // Staggered an hour apart so they do not all claim the same
                // instant; real SMS arriving later still sorts above them.
                sentAt = now - (index + 1) * 3_600_000L,
                category = categoryFor(result.optString("label")),
                result = result,
            )
            messages += message
            persist(message)
        }
        notifyChanged()
    }

    /** Drops everything, on disk as well as in memory. Used by the tests. */
    @Synchronized
    fun clear() {
        messages.clear()
        val d = dao
        if (d != null) io.execute { d.clear() }
        notifyChanged()
    }

    /**
     * -> null when this message is already on the page.
     *
     * Both capture paths land here and they read two different clocks: the live
     * receiver takes the SMSC's timestamp straight off the PDU, while the inbox
     * sync reads the moment the phone wrote the provider row. The `hash` column
     * cannot settle it — one SMS, two hashes — so "same sender, same text, near
     * enough the same time" is the only key the two paths agree on.
     *
     * ponytail: a day of slack, which costs a sender who sends the identical
     * text twice in one day its second copy. Match on Telephony.Sms.DATE_SENT
     * (the SMSC clock, in the provider) if that ever shows up in a real inbox.
     */
    @Synchronized
    fun add(sender: String, body: String, sentAt: Long): ScannedMessage? {
        val duplicate = messages.any {
            it.sender == sender && it.body == body && abs(it.sentAt - sentAt) < SAME_MESSAGE_MS
        }
        if (duplicate) return null
        val message = ScannedMessage(newId(), sender, body, sentAt)
        // Appended, not inserted at the front: [all] sorts, so insertion order
        // carries no meaning and must not pretend to.
        messages += message
        if (messages.size > MAX_KEPT) {
            messages.sortBy { it.sentAt }
            repeat(messages.size - MAX_KEPT) { messages.removeAt(0) }
        }
        persist(message)
        notifyChanged()
        return message
    }

    /**
     * The verdict, once the model has one. Written through rather than left on
     * the object, or a restart would show every scanned message as pending
     * again.
     */
    @Synchronized
    fun setResult(message: ScannedMessage, category: String, result: JSONObject?) {
        message.category = category
        message.result = result
        val d = dao
        if (d != null) io.execute { d.setResult(message.id, category, result?.toString()) }
        notifyChanged()
    }

    /**
     * Every message, newest first — one page, not a section per verdict.
     *
     * Sorted here rather than trusted from insertion order, because the writers
     * disagree about it: the inbox backfill reads DATE DESC and the live
     * receiver arrives oldest-last. `id` breaks ties, so two SMS landing in the
     * same millisecond still get a stable order instead of swapping places on
     * every repaint.
     *
     * Filtering by category is deliberately gone: CATEGORY_ERROR had no section
     * of its own, so a failed scan was invisible. One list shows it by
     * construction.
     */
    @Synchronized
    fun all(): List<ScannedMessage> =
        messages.sortedWith(
            compareByDescending<ScannedMessage> { it.sentAt }.thenByDescending { it.id }
        )

    @Synchronized
    fun get(id: Long): ScannedMessage? = messages.find { it.id == id }

    fun addListener(listener: () -> Unit) {
        listeners.add(listener)
    }

    fun removeListener(listener: () -> Unit) {
        listeners.remove(listener)
    }

    /**
     * Always delivered on the main thread. Listeners repaint views, and [clear]
     * can be reached from a background thread (unpair talks to the server
     * first), so posting here is what stops a view being touched off the UI
     * thread — fixed once, rather than at every caller that has to remember.
     * `toList()` snapshots the listeners so one that unregisters itself cannot
     * trip the iterator.
     */
    fun notifyChanged() {
        val handler = mainThread ?: return listeners.toList().forEach { it() }
        // COALESCED, and it has to be. A listener repaints the whole list, and
        // the inbox backfill adds N messages and then resolves N verdicts, so
        // an un-batched notify posts 2N full re-inflations onto the main thread
        // — each of which is O(list) view work. Measured on a 30-message
        // backfill: "Skipped 465 frames", 8-second frames, GC thrash, and the
        // list crawling in one message at a time for minutes.
        //
        // One repaint per message is not worth anything either: nobody can see
        // thirty repaints, they see the last one. So while a repaint is already
        // queued, later changes ride along with it.
        synchronized(this) {
            if (repaintQueued) return
            repaintQueued = true
        }
        handler.post {
            synchronized(this) { repaintQueued = false }
            listeners.toList().forEach { it() }
        }
    }
}
