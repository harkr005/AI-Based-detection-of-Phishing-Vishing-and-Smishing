"""SMS channel: TF-IDF character n-grams + logistic regression, three classes.

This was MobileBERT until 2026-08-21. The transformer was replaced because it
lost its own justification. §4 picked MobileBERT for size and speed on-device
("~0.39s, ~74MB, no GPU"); measured on the SAME splits, the same augmented
training rows and the same §9.9 attacks:

  model                 cross-source   in-source   size      per SMS   worst attack
  MobileBERT                  0.878      0.980   98.5 MB   0.39s (spec)      -0.048
  char 3-5 + LogReg           0.878      0.975    2.6 MB      0.11 ms        -0.004

That table is the head-to-head, measured 2026-08-21 on the corpora as they
stood. The SHIPPED model has moved since: a real inbox was added to training on
2026-09-23 (see data.load_sms), which reads in-source 0.984 / cross-source
0.870 - +0.009 and -0.008 against the row above. What it bought is not in
either column, because neither is measured on traffic: 943 real, carrier-
stamped, legitimate Indian messages went from 128 false alarms to 1, at a cost
of one missed message out of 187 held-out smishing. See
scratch/_inbox_verdicts.py and scratch/_smish_verdicts.py, which print both
sides of that trade.

Since 2026-10-28 the word list comes from public rows only (see train()), so
no private string ships in the APK: in-source 0.980 / cross-source 0.859,
inbox false alarms 0 of 943, smishing caught 139 of 187.

Same honest score, 37x smaller, and MORE robust to obfuscation, not less: the
homoglyph/zero-width family costs it nothing once normalize() runs, and
swap/leet costs it 0.004 macro-F1. It is also the only variant here whose whole
training run fits in 3 seconds, which is why the §4 benchmark can be re-run on
demand instead of overnight.

Why character n-grams and not words: an SMS is ~15 words, and the attack surface
is inside the word - "kyc" written "k y c", "verify" written "ver1fy". Word
features see an unknown token; 3-5 character windows still see most of it.

Why LogisticRegression and not LinearSVC: fusion consumes calibrated
probabilities for the §3 bands, and LogReg emits them natively. A calibrated
LinearSVC scored the same but needed a CalibratedClassifierCV wrapper and came
out worse calibrated (log-loss 0.170 vs 0.145).

The transformer is kept verbatim in `_sms_mobilebert.py` so the §4 comparison
stays reproducible.

Two numbers are always reported:
  in-source   held-out slice of the corpora it trained on   (the flattering one)
  cross-source  a corpus it has never seen at all           (the honest one)
The gap is how much of the score is the dataset rather than the smishing.

No temporal split here: none of the SMS corpora carry timestamps. Email and URL
do get temporal splits - see the report before quoting a protocol.
"""
import pickle
from functools import lru_cache
from pathlib import Path

import data

import adversarial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

OUT = Path(__file__).parent / "models" / "sms_charlr.pkl"
LABELS = ["ham", "spam", "smishing"]


def _clean(texts):
    """§9.9 normalisation, applied identically in training and in scoring.

    The transformer did this inside encode(); here it is the one place text
    enters the pipeline. Skip it in either path and train/serve skew silently.
    """
    return [adversarial.normalize(t) for t in texts]


def build():
    """The whole model. char_wb keeps n-grams inside word boundaries, so a
    3-gram never spans two words and the features stay interpretable."""
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                        ngram_range=(3, 5), min_df=3),
        LogisticRegression(max_iter=2000, C=10, n_jobs=-1))


def train(df):
    """The word list is fitted on PUBLIC rows only; the weights on every row.

    The vocabulary ships in plain text inside the APK. Fitted on the private
    inbox too, it held ~16k n-grams no public row has - names included. Now a
    term must occur in >= 3 public rows, and the inbox only teaches weights.
    Cost, measured in scratch/_sms_public_vocab.py: cross-source F1 0.870 ->
    0.859, held-out smishing caught 142 -> 139 of 187, inbox false alarms 1 -> 0.
    """
    pipe = build()
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    vec.fit(_clean(df[df.source != "inbox"].text.fillna("")))
    clf.fit(vec.transform(_clean(df.text.fillna(""))), df.label_3.map(LABELS.index).values)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as fh:
        pickle.dump(pipe, fh)
    return pipe


def report(name, model, df):
    p = model.predict_proba(_clean(df.text.fillna(""))).argmax(1)
    y = df.label_3.map(LABELS.index).values
    macro = f1_score(y, p, average="macro")
    print(f"\n--- {name}  (n={len(df):,})  macro-F1 {macro:.3f}")
    print(classification_report(y, p, target_names=LABELS, digits=3, zero_division=0))
    return macro


@lru_cache(maxsize=1)
def load():
    """Cached: voice scoring calls score() once per window, and unpickling per
    call dominated the runtime. lru_cache is the whole cache layer."""
    with OUT.open("rb") as fh:
        return pickle.load(fh)


def score(texts, raw=False):
    """-> (n, 3) probabilities over ham/spam/smishing. Fusion consumes column 2.

    raw=True returns the decision function instead - the log-odds this model
    emits before the softmax, which is what explain() needs. For a multinomial
    LogisticRegression softmax(decision_function) IS predict_proba, so the two
    paths cannot disagree.
    """
    model = load()
    t = _clean(list(texts))
    return model.decision_function(t) if raw else model.predict_proba(t)


MAX_WORDS = 40  # SMS are short; explanation cost is one cheap re-score per word


def explain(text, top_k=3):
    """-> (label, probability, reason string, evidence words). §4 requires a
    reason per verdict; the words are the same evidence as DATA, so the UI can
    show what was flagged instead of a sentence the reader has to parse.

    Leave-one-word-out: drop each word, see how far the predicted class falls.
    The words that cost the most ARE the evidence, measured on the model itself.

    EXACT attribution, not perturbation. The model is linear over char n-grams,
    so a word's contribution to the winning class is just the sum of its own
    n-gram weights times their tf-idf values - the same move url_model makes
    with TreeSHAP, and for the same reason: when the exact answer is one dot
    product, approximating it is theatre.

    Leave-one-word-out was tried here first and is WRONG for this model. Removing
    a word deletes one n-gram per character, so long words always look decisive:
    on the fake-KYC sanity message it blamed "Your", "Update", "account" and
    never "KYC" or "blocked". Exact attribution has no length bias.
    """
    model = load()
    vec, clf = model.steps[0][1], model.steps[1][1]
    words = adversarial.normalize(text).split()[:MAX_WORDS]
    if not words:
        return LABELS[0], 1.0, "empty message", []

    x = vec.transform([" ".join(words)])
    lg = clf.decision_function(x)[0]
    k = int(lg.argmax())
    # margin against the runner-up class, so the evidence explains THIS verdict
    # rather than "not ham" in general
    rival = int(max((i for i in range(len(lg)) if i != k), key=lambda i: lg[i]))
    w = clf.coef_[k] - clf.coef_[rival]
    p = float(np.exp(lg[k] - np.logaddexp.reduce(lg)))

    analyzer, vocab = vec.build_analyzer(), vec.vocabulary_
    cols, vals = x.indices, x.data
    weight = {j: v * w[j] for j, v in zip(cols, vals)}
    drop = np.array([sum(weight.get(vocab[g], 0.0)
                         for g in set(analyzer(word)) if g in vocab)
                     for word in words])

    # 1% of the total positive evidence, so the cut scales with confidence
    floor = 0.01 * max(drop[drop > 0].sum(), 1e-9)
    top = [i for i in sorted(range(len(words)), key=lambda i: -drop[i])[:top_k]
           if drop[i] > floor]
    if not top:
        return LABELS[k], p, f"{LABELS[k]}, no single word decides it", []
    terms = [words[i].strip('"\'.,!?:;') for i in sorted(top)]
    quoted = ", ".join('"%s"' % t for t in terms)
    return LABELS[k], p, f"{LABELS[k]} - driven by {quoted}", terms


if __name__ == "__main__":
    s = data.load_sms()

    # Dataset_5971 is held out entirely - a corpus the model never sees.
    held = s[s.source == "Dataset_5971"]
    pool = s[s.source != "Dataset_5971"]
    # Split by group, not at random: the Indian rows are template-generated, so a
    # random split leaks near-twins into the test set. Costs the stratification -
    # groups win, the classes are large enough that the split stays close anyway.
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
                      .split(pool, groups=pool.group))
    tr, te = pool.iloc[i_tr], pool.iloc[i_te]

    # §9.9: half the training rows get an obfuscated twin. Test sets stay clean -
    # adversarial.py is what measures the attacked numbers.
    tr = data.pd.concat([tr, adversarial.augment(tr)], ignore_index=True)

    print(f"train {len(tr):,}  in-source test {len(te):,}  cross-source test {len(held):,}")
    model = train(tr)
    load.cache_clear()

    in_src = report("in-source (same corpora)", model, te)
    cross = report("cross-source (unseen corpus)", model, held)
    print(f"\ngap: {in_src - cross:+.3f} macro-F1 - that is the dataset, not the model")
    print(f"model on disk: {OUT.stat().st_size / 1e6:.2f} MB")

    p = score([
        "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked",
        "Hey, are we still meeting at 6 for dinner?",
    ])
    smish, ham = p[0][2], p[1][0]
    print(f"\nsanity: fake-KYC smishing={smish:.2f}  dinner-text ham={ham:.2f}")
    assert smish > 0.5 and ham > 0.5, "obvious cases must land in the right class"

    lab, prob, why, _ = explain(
        "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked")
    print(f"reason: {why}  ({prob:.2f})")
    assert lab == "smishing" and prob > 0.5
    # the explanation must point at the lure, not at "your" or "now"
    assert any(w in why.lower() for w in ("kyc", "sbi", "http", "blocked", "expired")), why

    # softmax(decision_function) must equal predict_proba, or explain() and
    # fusion are reading two different models
    a = score(["Update your KYC now"])[0]
    r = score(["Update your KYC now"], raw=True)[0]
    assert np.allclose(a, np.exp(r - np.logaddexp.reduce(r)), atol=1e-6)
    print("ok")
