"""Adversarial augmentation (spec §9.9).

Two halves, because they solve different halves of the problem:

  normalize()  - a 6-line preprocessing fix. Zero-width chars, homoglyphs and
                 NFKC lookalikes all collapse back to plain ASCII before the
                 tokenizer sees them. Cheaper and more reliable than teaching a
                 transformer to ignore U+200B, and it protects the URL channel
                 too. Applied at BOTH train and score time, so no train/serve
                 skew.
  perturb()    - the attacks normalization cannot undo: adjacent character
                 swaps, leetspeak, punctuation injection inside words, synonym
                 substitution. These survive any cleaning, so the only defence
                 is seeing them in training -> augment().

Attack model: the attacker rewrites the message, not the label. Every op keeps
the message human-readable ("Y0ur acc0unt is bl0cked") - that is the constraint
a real smisher works under too.

URLs are left untouched: obfuscating a link is the URL channel's problem
(punycode + entropy features already cover it), and mangling the URL here would
measure the wrong model.

ponytail: no LLM paraphrase (§9.9 item 6) - needs an API key and a generation
budget, and it is the only op that cannot be verified offline. Add it by
appending paraphrase pairs to a CSV and unioning them in augment().
"""
import random
import re
import unicodedata

import pandas as pd

# Cyrillic/Greek lookalikes an attacker substitutes for ASCII. Same glyph on
# screen, different codepoint, so "аccount" (Cyrillic а) tokenizes as garbage.
HOMOGLYPHS = {"a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о",
              "p": "р", "s": "ѕ", "x": "х", "y": "у", "h": "һ", "k": "к",
              "m": "м", "t": "т", "b": "Ь", "n": "п", "d": "ԁ"}
_UNHOMO = {v: k for k, v in HOMOGLYPHS.items()}
ZERO_WIDTH = "​‌‍⁠﻿"
LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "l": "1", "t": "7"}

# Scam vocabulary only. A general thesaurus (WordNet) swaps content words at
# random and changes what the message says; these pairs are the rewrites a
# smisher actually uses to dodge keyword filters.
SYNONYMS = {
    "verify": "confirm", "confirm": "verify", "update": "renew",
    "urgent": "immediate", "immediately": "right away", "account": "a/c",
    "blocked": "suspended", "suspended": "deactivated", "expired": "lapsed",
    "click": "tap", "link": "url", "free": "complimentary", "win": "receive",
    "won": "have won", "prize": "reward", "bank": "banking", "kyc": "e-kyc",
    "otp": "one time password", "password": "passcode", "cash": "funds",
    "money": "amount", "now": "at once", "today": "within the day",
}

_URLISH = re.compile(r"https?://|www\.|\.(com|in|tk|xyz|ml|ga|cf|net|org)\b", re.I)
_ZW_RE = re.compile(f"[{ZERO_WIDTH}]")


NORMALIZE = True  # flipped off by robustness() to measure what the defence buys


def normalize(text):
    """Undo the cleanable half of the attack. Run before every tokenizer call."""
    if not NORMALIZE:
        return str(text)
    text = _ZW_RE.sub("", unicodedata.normalize("NFKC", str(text)))
    return "".join(_UNHOMO.get(c, c) for c in text)


# --- attacks -----------------------------------------------------------------

def _swap(w, rng):
    """Transpose two adjacent inner characters: 'account' -> 'accuont'."""
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 2)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _homoglyph(w, rng):
    idx = [i for i, c in enumerate(w) if c.lower() in HOMOGLYPHS]
    if not idx:
        return w
    i = rng.choice(idx)
    return w[:i] + HOMOGLYPHS[w[i].lower()] + w[i + 1:]


def _zero_width(w, rng):
    if len(w) < 3:
        return w
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + rng.choice(ZERO_WIDTH) + w[i:]


def _inject(w, rng):
    """Whitespace/punctuation inside a word: 'OTP' -> 'O.T.P', 'bank' -> 'ba nk'."""
    if len(w) < 3:
        return w
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + rng.choice(". -_") + w[i:]


def _leet(w, rng):
    idx = [i for i, c in enumerate(w) if c.lower() in LEET]
    if not idx:
        return w
    i = rng.choice(idx)
    return w[:i] + LEET[w[i].lower()] + w[i + 1:]


def _synonym(w, rng):
    s = SYNONYMS.get(w.lower().strip(".,!:;"))
    return s.upper() if s and w.isupper() else (s or w)


OPS = [_swap, _homoglyph, _zero_width, _inject, _leet, _synonym]
UNICODE_OPS = [_homoglyph, _zero_width]   # what normalize() undoes completely
HARD_OPS = [_swap, _inject, _leet, _synonym]  # what only augmentation can teach


def perturb(text, rate=0.3, ops=OPS, seed=None):
    """Rewrite ~`rate` of the words with a random op. Deterministic given seed."""
    rng = random.Random(seed)
    out = []
    for w in str(text).split():
        out.append(rng.choice(ops)(w, rng)
                   if len(w) > 2 and rng.random() < rate and not _URLISH.search(w)
                   else w)
    return " ".join(out)


def attack(texts, rate=0.3, ops=OPS, seed=0):
    return [perturb(t, rate, ops, seed + i) for i, t in enumerate(texts)]


def augment(df, frac=0.5, rate=0.25, seed=0, col="text"):
    """Perturbed copies of `frac` of the rows, labels unchanged.

    All classes are perturbed, not just the malicious ones. Augmenting only
    smishing/phishing would teach the model "mangled spelling = attack", which
    is an artifact of the augmenter and misfires on typo-heavy real ham.
    """
    d = df.sample(frac=frac, random_state=seed).copy()
    d[col] = attack(d[col].tolist(), rate=rate, seed=seed)
    return d


# --- measurement -------------------------------------------------------------

def robustness(name, score_fn, texts, cls):
    """Print detection rate on clean text vs the same text attacked.

    score_fn(texts) -> (n, k) probabilities; `cls` is the malicious column.
    Every row passed in is malicious, so the number is recall.

    The two attack families are reported separately on purpose - mixing them
    dilutes both, and they have different fixes. Unicode obfuscation is run at
    rate 1.0 (an attacker who bothers, bothers with every word) and scored with
    normalize() off and on; the character/synonym attacks are run at two rates
    and normalize() cannot touch them.
    """
    global NORMALIZE
    texts = list(texts)
    base = (score_fn(texts).argmax(1) == cls).mean()
    print(f"\n{name}  (n={len(texts):,})   clean recall {base:6.1%}", flush=True)

    uni = attack(texts, rate=1.0, ops=UNICODE_OPS)
    for tag, flag in [("normalize OFF", False), ("normalize ON ", True)]:
        NORMALIZE = flag
        got = (score_fn(uni).argmax(1) == cls).mean()
        print(f"   homoglyph+zero-width  rate 1.00  {tag}  {got:6.1%}  ({got - base:+.1%})",
              flush=True)
    NORMALIZE = True

    for r in (0.25, 0.5):
        got = (score_fn(attack(texts, rate=r, ops=HARD_OPS)).argmax(1) == cls).mean()
        print(f"   swap/leet/inject/syn  rate {r:.2f}                 "
              f"{got:6.1%}  ({got - base:+.1%})", flush=True)
    return base


def main():
    import data  # before torch, see sms_model
    import sms_model
    import email_model

    assert normalize("аcc​ount") == "account"          # homoglyph + ZWSP
    assert normalize(perturb("verify your account now", 1.0, [_homoglyph, _zero_width], 3)) \
        == "verify your account now"                        # cleanable ops fully undone
    assert perturb("update your bank account immediately", 1.0, HARD_OPS, 2) \
        != "update your bank account immediately"
    assert perturb("pay at http://sbi-kyc.tk/x now", 1.0, seed=4).count("http://sbi-kyc.tk/x") == 1
    assert len(augment(pd.DataFrame({"text": ["a b c d e"] * 10}), frac=0.5)) == 5

    s = data.load_sms()
    smish = s[(s.source == "Dataset_5971") & (s.label_3 == "smishing")]
    robustness("SMS smishing (cross-source corpus)",
               lambda t: sms_model.score(t), smish.text.tolist(), cls=2)

    e = data.load_email()
    _, ph_te = email_model.phishing_split(e)
    robustness("EMAIL phishing (held-out by time)",
               lambda t: email_model.score(t)[0], ph_te.text.tolist(), cls=2)
    print("\nok")


if __name__ == "__main__":
    # Run through the imported module, not through __main__: sms_model does
    # `import adversarial`, and a script run would give it a SECOND copy of this
    # module whose NORMALIZE flag robustness() cannot reach.
    import adversarial
    adversarial.main()
