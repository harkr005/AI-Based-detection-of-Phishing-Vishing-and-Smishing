package com.sentinel.app

import android.content.Context
import java.io.InputStream
import java.nio.ByteOrder
import java.text.Normalizer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * `sms_model.py`, on the phone (PROJECT_SPEC.md §10 Mode 1).
 *
 * The Python model is TF-IDF char_wb (3,5) + multinomial LogisticRegression, so
 * there is nothing to convert: a vocabulary, three float vectors and one dot
 * product. `export_sms_model.py` writes them to `assets/sms_charlr.bin` (1.3 MB)
 * and this reads them back. No TFLite, no ONNX, no tokenizer — a transformer
 * was tried and lost on every axis (see the header of sms_model.py).
 *
 * Every step below mirrors sklearn exactly, including the parts that look like
 * mistakes, because a port that is merely reasonable is a different model:
 *   - char_wb pads each word with spaces and cuts n-grams inside the padding;
 *   - a word shorter than n yields ONE n-gram and no longer ones at all;
 *   - out-of-vocabulary n-grams are dropped BEFORE the L2 norm, so the norm is
 *     taken over the terms that survived, not over the message.
 * `SmsModelTest` holds it to 1e-4 against sklearn's own output on 50 texts.
 */
object SmsModel {

    /** sms_model.LABELS, in column order. [categoryFor] turns these into words. */
    val LABELS = listOf("ham", "spam", "smishing")

    private var vocab: HashMap<String, Int>? = null
    private lateinit var idf: FloatArray
    private lateinit var intercept: FloatArray
    private lateinit var coef: FloatArray // classes x vocab, row-major

    val ready: Boolean get() = vocab != null

    @Synchronized
    fun ensureLoaded(context: Context) {
        if (vocab == null) context.applicationContext.assets.open(ASSET).use { load(it) }
    }

    /** Also the unit tests' entry point, which have assets but no Context. */
    @Synchronized
    fun load(stream: InputStream) {
        val bytes = stream.readBytes()
        var pos = 0
        fun line(): String {
            // UTF-8 continuation bytes are all >= 0x80, so scanning for a raw
            // 0x0A cannot land inside a multi-byte n-gram.
            var end = pos
            while (bytes[end] != NEWLINE) end++
            val s = String(bytes, pos, end - pos, Charsets.UTF_8)
            pos = end + 1
            return s
        }

        val header = line().split(" ")
        require(header[0] == "smslr1") { "not an sms model asset: ${header[0]}" }
        val n = header[1].toInt()
        val classes = header[2].toInt()
        require(classes == LABELS.size) { "asset has $classes classes, expected ${LABELS.size}" }

        // ponytail: 62k boxed map entries, ~6 MB of heap, loaded once per
        // process. If that ever matters, the terms are already written in
        // column order — sort the bytes and binary-search them instead.
        val terms = HashMap<String, Int>(n * 2)
        for (i in 0 until n) terms[line()] = i

        val floats = java.nio.ByteBuffer.wrap(bytes, pos, bytes.size - pos)
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        idf = FloatArray(n).also { floats.get(it) }
        intercept = FloatArray(classes).also { floats.get(it) }
        coef = FloatArray(classes * n).also { floats.get(it) }
        require(!floats.hasRemaining()) { "${floats.remaining()} floats left over" }
        vocab = terms
    }

    /**
     * -> probabilities over [LABELS]. Column 2 (smishing) is what fusion reads.
     * Costs well under a millisecond once loaded; the Python model measured
     * 0.11 ms per SMS and this does the same arithmetic.
     */
    fun score(text: String): FloatArray = softmax(logits(transform(normalize(text))))

    /** TfidfVectorizer.transform for one document -> column -> tf-idf, L2-normalised. */
    private fun transform(normalised: String): HashMap<Int, Double> {
        val terms = checkNotNull(vocab) { "SmsModel.ensureLoaded() first" }

        // tf, per surviving column. sklearn counts n-grams, so does this.
        val counts = HashMap<Int, Int>()
        for (word in normalised.lowercase().split(SPACES)) {
            if (word.isEmpty()) continue
            ngrams(word) { g -> terms[g]?.let { counts[it] = (counts[it] ?: 0) + 1 } }
        }

        val value = HashMap<Int, Double>(counts.size * 2)
        var norm = 0.0
        for ((column, count) in counts) {
            // sublinear_tf=True, then idf, then L2 — TfidfVectorizer's order.
            val v = (1.0 + ln(count.toDouble())) * idf[column]
            value[column] = v
            norm += v * v
        }
        // An all-OOV message (emoji, a foreign script) has no column at all.
        // Dividing by its zero norm is how a port starts returning NaN.
        val scale = if (norm > 0.0) 1.0 / sqrt(norm) else 0.0
        for ((column, v) in value) value[column] = v * scale
        return value
    }

    private fun logits(row: Map<Int, Double>): DoubleArray {
        val out = DoubleArray(LABELS.size) { intercept[it].toDouble() }
        for ((column, x) in row) {
            for (c in out.indices) out[c] += x * coef[c * idf.size + column]
        }
        return out
    }

    /** What [explain] returns: the winning class, its probability, the §4 reason
     * sentence, and the same evidence as data for the UI to show as chips. */
    data class Explanation(
        val label: String,
        val probability: Double,
        val reason: String,
        val terms: List<String>,
    )

    /**
     * `sms_model.explain` — EXACT attribution, not perturbation. The model is
     * linear over char n-grams, so a word's contribution to the winning class is
     * the sum of its own n-gram weights times their tf-idf values.
     *
     * Leave-one-word-out was tried in Python first and is wrong for this model:
     * dropping a word deletes one n-gram per character, so long words always
     * look decisive. Do not "simplify" this back into a perturbation loop.
     *
     * Note it scores the first [MAX_WORDS] words re-joined by single spaces,
     * which is NOT what [score] sees — so `label` here can disagree with
     * `score`'s argmax on a long message. That is Python's behaviour and the
     * dashboard shows both, so the port keeps it.
     */
    fun explain(text: String, topK: Int = 3): Explanation {
        val words = normalize(text).split(SPACES).filter { it.isNotEmpty() }.take(MAX_WORDS)
        if (words.isEmpty()) return Explanation(LABELS[0], 1.0, "empty message", emptyList())

        val row = transform(words.joinToString(" "))
        val lg = logits(row)
        val k = lg.indices.maxBy { lg[it] }
        // Margin against the runner-up, so the evidence explains THIS verdict
        // rather than "not ham" in general.
        val rival = lg.indices.filter { it != k }.maxBy { lg[it] }
        val p = softmax(lg)[k].toDouble()

        val weight = row.mapValues { (column, x) ->
            x * (coef[k * idf.size + column] - coef[rival * idf.size + column])
        }
        val terms = checkNotNull(vocab)
        val drop = words.map { word ->
            val seen = HashSet<String>()
            var sum = 0.0
            ngrams(word.lowercase()) { g ->
                if (seen.add(g)) terms[g]?.let { sum += weight[it] ?: 0.0 }
            }
            sum
        }

        // 1% of the total positive evidence, so the cut scales with confidence.
        val floor = 0.01 * maxOf(drop.filter { it > 0 }.sum(), 1e-9)
        val top = drop.indices.sortedByDescending { drop[it] }.take(topK)
            .filter { drop[it] > floor }.sorted()
        if (top.isEmpty()) {
            return Explanation(LABELS[k], p, "${LABELS[k]}, no single word decides it", emptyList())
        }
        val picked = top.map { words[it].trim { c -> c in "\"'.,!?:;" } }
        return Explanation(LABELS[k], p,
            "${LABELS[k]} - driven by " + picked.joinToString(", ") { "\"$it\"" }, picked)
    }

    /** SMS are short; explanation cost is one cheap re-score per word. */
    private const val MAX_WORDS = 40

    /**
     * `adversarial.normalize` (§9.9), byte for byte: NFKC, strip zero-width,
     * fold homoglyphs. Applied at training time too, so skipping it here would
     * be train/serve skew rather than a shortcut — an attacker's Cyrillic "о"
     * would tokenize as garbage and the message would score as unknown text.
     */
    fun normalize(text: String): String {
        val folded = StringBuilder(text.length)
        for (ch in Normalizer.normalize(text, Normalizer.Form.NFKC)) {
            if (ch in ZERO_WIDTH) continue
            folded.append(UNHOMOGLYPH[ch] ?: ch)
        }
        return folded.toString()
    }

    /**
     * sklearn's `_char_wb_ngrams`. Indexes CODEPOINTS, not UTF-16 units: the
     * vocabulary was cut in Python, where an emoji is one character, so slicing
     * by `char` would cut a surrogate pair in half and match nothing.
     */
    private inline fun ngrams(word: String, emit: (String) -> Unit) {
        val w = " $word ".codePoints().toArray()
        for (n in 3..5) {
            var offset = 0
            emit(slice(w, 0, minOf(n, w.size)))
            while (offset + n < w.size) {
                offset += 1
                emit(slice(w, offset, n))
            }
            // The quirk that has to be copied: a word this short was counted
            // once, and n+1 would count it again.
            if (offset == 0) break
        }
    }

    private fun slice(codePoints: IntArray, from: Int, count: Int): String =
        buildString(count + 1) {
            for (i in from until from + count) appendCodePoint(codePoints[i])
        }

    private const val NEWLINE = '\n'.code.toByte()

    private fun softmax(logits: DoubleArray): FloatArray {
        val top = logits.max()
        val e = DoubleArray(logits.size) { exp(logits[it] - top) }
        val sum = e.sum()
        return FloatArray(e.size) { (e[it] / sum).toFloat() }
    }

    private const val ASSET = "sms_charlr.bin"

    // Python's str.split() has no argument here, so any run of whitespace
    // splits — and its idea of whitespace is Unicode's, not the six ASCII
    // characters Java's \s means. NFKC has already folded U+00A0 and friends
    // into plain spaces, but U+2028 and U+2029 survive it.
    //
    // Spelled out rather than written `(?U)\s+`: the unit tests run on OpenJDK's
    // regex engine and Android runs ICU's, which REFUSES that inline flag
    // outright — PatternSyntaxException on the device, green tests on the
    // desktop. Every class here means the same thing to both.
    private val SPACES =
        Regex("[\\s\\u001c-\\u001f\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]+")

    private const val ZERO_WIDTH = "​‌‍⁠﻿"

    // adversarial.HOMOGLYPHS, inverted: the lookalike an attacker substitutes
    // back to the ASCII it imitates. 'Ь' is the uppercase soft sign — the table
    // pairs it with lowercase 'b' on purpose, it is a shape, not a letter.
    private val UNHOMOGLYPH = mapOf(
        'а' to 'a', 'с' to 'c', 'е' to 'e', 'і' to 'i',
        'ј' to 'j', 'о' to 'o', 'р' to 'p', 'ѕ' to 's',
        'х' to 'x', 'у' to 'y', 'һ' to 'h', 'к' to 'k',
        'м' to 'm', 'т' to 't', 'Ь' to 'b', 'п' to 'n',
        'ԁ' to 'd',
    )
}
