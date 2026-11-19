package com.sentinel.app

import android.content.Context
import java.io.InputStream
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.log2
import kotlin.math.max

/**
 * `url_model.py`, on the phone: fifteen lexical features of the HOST and a
 * 400-tree XGBoost walk over them. `export_android.py` flattens the trees into
 * `assets/url_xgb.bin` (0.64 MB, smaller than the JSON it came from) and pins
 * the registry domains in `assets/verified_domains.txt`.
 *
 * Domain-level on purpose. Path and scheme features were REFUSED in Python:
 * PhiUSIIL's legitimate URLs are 100% HTTPS-with-no-path, so a model trained on
 * them learns the collection method and scores 0.331 AUC on live data — worse
 * than chance. Do not add path features here without a corpus that carries them
 * on both classes.
 */
object UrlModel {

    /** url_model.FEATURES, in the order the trees index them. */
    val FEATURES = listOf(
        "host_len", "n_dots", "n_hyphen", "n_digits", "digit_ratio", "n_special",
        "n_subdomain", "is_ip", "is_punycode", "tld_len", "tld_rare", "entropy",
        "longest_token", "n_lure", "n_vowel_ratio",
    )

    private var verified: Set<String>? = null
    private lateinit var treeStart: IntArray
    private lateinit var left: IntArray
    private lateinit var right: IntArray
    private lateinit var feature: IntArray
    private lateinit var threshold: FloatArray
    private lateinit var leaf: FloatArray
    private lateinit var cover: FloatArray
    private var maxDepth = 0
    private var baseMargin = 0.0

    val ready: Boolean get() = verified != null

    @Synchronized
    fun ensureLoaded(context: Context) {
        if (verified != null) return
        val assets = context.applicationContext.assets
        assets.open("url_xgb.bin").use { loadTrees(it) }
        assets.open("verified_domains.txt").use { loadRegistry(it) }
    }

    @Synchronized
    fun loadTrees(stream: InputStream) {
        val bytes = stream.readBytes()
        var pos = 0
        while (bytes[pos] != '\n'.code.toByte()) pos++
        val header = String(bytes, 0, pos, Charsets.UTF_8).split(" ")
        require(header[0] == "urlxgb2") { "not a url model asset: ${header[0]}" }
        val trees = header[1].toInt()
        val nodes = header[2].toInt()
        maxDepth = header[3].toInt()
        baseMargin = header[4].toDouble()

        val buffer = java.nio.ByteBuffer.wrap(bytes, pos + 1, bytes.size - pos - 1)
            .order(ByteOrder.LITTLE_ENDIAN)
        val ints = buffer.asIntBuffer()
        treeStart = IntArray(trees + 1).also { ints.get(it) }
        this.left = IntArray(nodes).also { ints.get(it) }
        this.right = IntArray(nodes).also { ints.get(it) }
        feature = IntArray(nodes).also { ints.get(it) }
        buffer.position(buffer.position() + ints.position() * 4)
        val floats = buffer.asFloatBuffer()
        threshold = FloatArray(nodes).also { floats.get(it) }
        leaf = FloatArray(nodes).also { floats.get(it) }
        cover = FloatArray(nodes).also { floats.get(it) }
        require(!floats.hasRemaining()) { "${floats.remaining()} floats left over" }
    }

    @Synchronized
    fun loadRegistry(stream: InputStream) {
        verified = stream.bufferedReader().readLines()
            .map { it.trim().lowercase() }.filter { it.isNotEmpty() }.toSet()
    }

    /** url_model.features(): one URL -> the fifteen numbers, in [FEATURES] order. */
    fun features(url: String): DoubleArray {
        val host = host(url)
        val tld = if ("." in host) host.substringAfterLast(".") else ""
        val digits = host.count { it.isDigit() }
        val letters = host.count { it.isLetter() }
        val lower = host.lowercase()
        return doubleArrayOf(
            host.length.toDouble(),
            host.count { it == '.' }.toDouble(),
            host.count { it == '-' }.toDouble(),
            digits.toDouble(),
            digits / max(host.length, 1).toDouble(),
            host.count { it in "@_~+%" }.toDouble(),
            max(host.count { it == '.' } - 1, 0).toDouble(),
            if (IP.matches(host)) 1.0 else 0.0,
            if ("xn--" in host) 1.0 else 0.0,
            tld.length.toDouble(),
            if (tld in COMMON_TLD) 0.0 else 1.0,
            entropy(host),
            host.split(TOKEN).maxOfOrNull { it.length }?.toDouble() ?: 0.0,
            LURE.count { it in lower }.toDouble(),
            lower.count { it in "aeiou" } / max(letters, 1).toDouble(),
        )
    }

    /**
     * -> phishing probability in [0,1]. This is what fusion consumes.
     *
     * Registry domains are pinned to 0.0, one guard here rather than at each
     * call site. ponytail (inherited from Python): a flat 0.0, not a discount —
     * an open redirect on a real bank domain sails through. The upgrade needs
     * path features, which need a corpus that has them.
     */
    fun score(url: String): Double = if (isVerified(url)) 0.0 else probability(url)

    /**
     * The model's own probability, WITHOUT the registry pin. `url_model.explain`
     * reads this rather than `score`, so a registry domain the model happens to
     * dislike is described as phishing while scoring 0.0 — a Python quirk the
     * port copies, because the dashboard and the phone must say the same thing
     * about the same link.
     */
    fun probability(url: String): Double = probabilityOf(features(url))

    private fun probabilityOf(x: DoubleArray): Double {
        var margin = baseMargin
        for (t in 0 until treeStart.size - 1) {
            var i = treeStart[t]
            // float32 on BOTH sides. xgboost casts the feature to float32 before
            // it tests a split, and on digit_ratio or entropy the double falls
            // the other side of a threshold often enough to move the margin by
            // 0.16 — measured while exporting, not guessed.
            while (left[i] != -1) {
                i = if (x[feature[i]].toFloat() < threshold[i]) left[i] else right[i]
            }
            margin += leaf[i]
        }
        return 1.0 / (1.0 + exp(-margin))
    }

    /**
     * True if the host IS a registry domain or a subdomain of one. Suffix match
     * on the HOST, never a substring on the URL: "hdfcbank.com.evil.tk" ends
     * with a registry string and must not be trusted, and "nothdfcbank.com"
     * ends with "hdfcbank.com" without the dot. Both fail here.
     */
    fun isVerified(url: String): Boolean {
        val registry = verified ?: return false
        val host = host(url)
        return registry.any { host == it || host.endsWith(".$it") }
    }

    /** [explain]'s two outputs: the §4 reason sentence, and the same content as
     * data so the UI can say WHY a link looks wrong instead of printing it. */
    data class LinkExplanation(val reason: String, val named: List<String>)

    /**
     * `url_model.explain` — the named contributions come from exact TreeSHAP,
     * the same numbers xgboost's own `pred_contribs=True` produces.
     *
     * A benign host has no positive evidence worth naming: the top
     * contributions are then just the least-negative ones, and printing them
     * reads as a false accusation. So below 0.5 it says nothing and says so.
     */
    fun explain(url: String, topK: Int = 3): LinkExplanation {
        val x = features(url)
        if (probabilityOf(x) < 0.5) {
            return LinkExplanation("link looks benign: nothing in the host name stands out",
                emptyList())
        }
        val contributions = contributions(x)
        val named = contributions.indices
            .sortedByDescending { contributions[it] }.take(topK)
            .filter { contributions[it] > 0.05 }
            .map { PHRASE[it] }
        return LinkExplanation(
            "link looks phishing" + if (named.isEmpty()) "" else ": " + named.joinToString(", "),
            named,
        )
    }

    /**
     * Exact TreeSHAP: how much each of the fifteen features moved THIS URL's
     * score, in log-odds. Lundberg's algorithm, transliterated from xgboost's
     * own C++ — the `extend`/`unwind` pair tracks, for every feature on the path
     * so far, the fraction of subsets it is and is not in, so a contribution can
     * be read off each leaf in one pass instead of enumerating 2^15 subsets.
     *
     * This walks BOTH children at every split, so it visits the whole forest
     * rather than one path per tree: measured at 2.8 ms against the score's
     * 26 us on a desktop JVM, i.e. about a hundred times the cost. That is why
     * it runs for the one link a message is judged on, only when that link is
     * already over 0.5, and never on the main thread.
     *
     * The cheap alternative was measured and refused: per-path (Saabas)
     * contributions pick the same top-three phrases as this on only 51.5% of
     * phishing-side URLs, which makes them a different explanation rather than
     * a cheaper one.
     */
    fun contributions(x: DoubleArray): DoubleArray {
        val phi = DoubleArray(FEATURES.size)
        // One flat slab, indexed by offset, exactly as the C++ does it with
        // pointer arithmetic: a child's path starts where its parent's ended.
        // Offsets are triangular in the depth, hence the size.
        val size = (maxDepth + 2) * (maxDepth + 3) / 2
        val path = ShapPath(IntArray(size), DoubleArray(size), DoubleArray(size),
            DoubleArray(size))
        for (t in 0 until treeStart.size - 1) {
            shap(x, phi, path, treeStart[t], 0, 0, 1.0, 1.0, -1)
        }
        return phi
    }

    /** The bookkeeping TreeSHAP carries down the tree: for each feature split on
     * so far, the fraction of subsets it is in (`one`) and is not in (`zero`),
     * and the combinatorial weight of the path to here. */
    private class ShapPath(
        val feature: IntArray,
        val zero: DoubleArray,
        val one: DoubleArray,
        val weight: DoubleArray,
    ) {
        fun extend(at: Int, depth: Int, zeroFraction: Double, oneFraction: Double, index: Int) {
            feature[at + depth] = index
            zero[at + depth] = zeroFraction
            one[at + depth] = oneFraction
            weight[at + depth] = if (depth == 0) 1.0 else 0.0
            for (i in depth - 1 downTo 0) {
                weight[at + i + 1] += oneFraction * weight[at + i] * (i + 1) / (depth + 1.0)
                weight[at + i] = zeroFraction * weight[at + i] * (depth - i) / (depth + 1.0)
            }
        }

        /** Undo [extend] for one feature — used when a feature is split on twice
         * down the same path, so it is not counted twice. */
        fun unwind(at: Int, depth: Int, index: Int) {
            val oneFraction = one[at + index]
            val zeroFraction = zero[at + index]
            var nextOne = weight[at + depth]
            for (i in depth - 1 downTo 0) {
                if (oneFraction != 0.0) {
                    val previous = weight[at + i]
                    weight[at + i] = nextOne * (depth + 1.0) / ((i + 1) * oneFraction)
                    nextOne = previous -
                        weight[at + i] * zeroFraction * (depth - i) / (depth + 1.0)
                } else {
                    weight[at + i] = weight[at + i] * (depth + 1.0) / (zeroFraction * (depth - i))
                }
            }
            for (i in index until depth) {
                feature[at + i] = feature[at + i + 1]
                zero[at + i] = zero[at + i + 1]
                one[at + i] = one[at + i + 1]
            }
        }

        /** What the path would weigh with `index` unwound, without unwinding it. */
        fun unwoundSum(at: Int, depth: Int, index: Int): Double {
            val oneFraction = one[at + index]
            val zeroFraction = zero[at + index]
            var nextOne = weight[at + depth]
            var total = 0.0
            for (i in depth - 1 downTo 0) {
                if (oneFraction != 0.0) {
                    val term = nextOne * (depth + 1.0) / ((i + 1) * oneFraction)
                    total += term
                    nextOne = weight[at + i] -
                        term * zeroFraction * (depth - i) / (depth + 1.0)
                } else if (zeroFraction != 0.0) {
                    total += (weight[at + i] / zeroFraction) / ((depth - i) / (depth + 1.0))
                }
            }
            return total
        }
    }

    private fun shap(
        x: DoubleArray,
        phi: DoubleArray,
        path: ShapPath,
        node: Int,
        uniqueDepth: Int,
        parentAt: Int,
        parentZero: Double,
        parentOne: Double,
        parentFeature: Int,
    ) {
        val at = parentAt + uniqueDepth + 1
        path.feature.copyInto(path.feature, at, parentAt, parentAt + uniqueDepth + 1)
        path.zero.copyInto(path.zero, at, parentAt, parentAt + uniqueDepth + 1)
        path.one.copyInto(path.one, at, parentAt, parentAt + uniqueDepth + 1)
        path.weight.copyInto(path.weight, at, parentAt, parentAt + uniqueDepth + 1)
        path.extend(at, uniqueDepth, parentZero, parentOne, parentFeature)

        if (left[node] == -1) {
            for (i in 1..uniqueDepth) {
                val w = path.unwoundSum(at, uniqueDepth, i)
                phi[path.feature[at + i]] += w * (path.one[at + i] - path.zero[at + i]) * leaf[node]
            }
            return
        }

        val hot = if (x[feature[node]].toFloat() < threshold[node]) left[node] else right[node]
        val cold = if (hot == left[node]) right[node] else left[node]
        val here = cover[node].toDouble()
        val hotZero = cover[hot] / here
        val coldZero = cover[cold] / here
        val split = feature[node]

        // Already split on this feature above? Then undo that split first, or it
        // is counted twice and the contributions stop summing to the margin.
        var depth = uniqueDepth
        var incomingZero = 1.0
        var incomingOne = 1.0
        var index = 0
        while (index <= depth && path.feature[at + index] != split) index++
        if (index != depth + 1) {
            incomingZero = path.zero[at + index]
            incomingOne = path.one[at + index]
            path.unwind(at, depth, index)
            depth -= 1
        }

        shap(x, phi, path, hot, depth + 1, at, hotZero * incomingZero, incomingOne, split)
        shap(x, phi, path, cold, depth + 1, at, coldZero * incomingZero, 0.0, split)
    }

    /** Every link in a message, worst first — `api._links` on the phone. */
    fun links(text: String): List<Pair<String, Double>> =
        URL_RE.findAll(text).map { it.value }.toList()
            .map { it to score(it) }.sortedByDescending { it.second }

    /** urlsplit(...).hostname: after the scheme, before the path, no userinfo,
     * no port, lowercased, and `www.` dropped the way url_model does it. */
    private fun host(url: String): String {
        val afterScheme = if ("://" in url) url.substringAfter("://") else url
        val netloc = afterScheme.takeWhile { it !in "/?#" }.substringAfterLast('@')
        val host = (if (netloc.startsWith("[")) netloc.substringAfter("[").substringBefore("]")
                    else netloc.substringBefore(":")).lowercase()
        return host.removePrefix("www.")
    }

    private fun entropy(s: String): Double {
        if (s.isEmpty()) return 0.0
        val n = s.length.toDouble()
        return -s.groupingBy { it }.eachCount().values.sumOf {
            val p = it / n
            p * log2(p)
        }
    }

    private val IP = Regex("""^\d{1,3}(\.\d{1,3}){3}$""")
    private val TOKEN = Regex("""[.\-_]""")

    // thread.URL_RE, with \w written out as ASCII. Python's \w is Unicode-aware,
    // but the two engines this has to run on disagree about it — OpenJDK (the
    // unit tests) reads \w as ASCII, Android's ICU reads it as Unicode — and the
    // inline (?U) that would settle it is a syntax error on ICU. Hosts are ASCII
    // or punycode in practice, and a non-ASCII domain written WITH a scheme
    // still matches the first branch.
    private val URL_RE = Regex(
        """https?://\S+|www\.\S+""" +
            """|\b[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.(?:com|in|net|org|co|io|tk|xyz|ml|ga|cf|top|link|info|ru)\b(?:/\S*)?""",
        RegexOption.IGNORE_CASE,
    )

    // url_model.PHRASE, in FEATURES order: one short phrase per feature, so a
    // verdict reads as English rather than as "n_lure=3".
    private val PHRASE = listOf(
        "unusually long host name", "many dot-separated parts", "hyphen-stuffed host",
        "digits in the host", "host is mostly digits", "odd characters in the host",
        "deep subdomain nesting", "raw IP address instead of a domain",
        "punycode (look-alike) domain", "unusual TLD length", "rare top-level domain",
        "random-looking host string", "one very long host token",
        "bank/payment lure words in the host", "host reads as non-word letters",
    )

    private val LURE = listOf(
        "login", "signin", "verify", "secure", "account", "update", "confirm",
        "bank", "otp", "kyc", "wallet", "free", "gift", "paytm", "upi",
    )
    private val COMMON_TLD = setOf(
        "com", "org", "net", "edu", "gov", "in", "uk", "de", "jp", "ru",
        "br", "it", "fr", "au", "ca", "nl", "es", "io", "co",
    )
}
