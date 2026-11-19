"""URL channel: lexical XGBoost scorer.

Same feature function is used for training and for scoring a live URL, so
there is no train/serve skew. Nothing here needs a page fetch - the model has
to run on a URL sitting inside an SMS before anyone clicks it.

Everything is scored at DOMAIN level. Measured on PhiUSIIL: 100% of its legit
URLs are https with no path, vs 49%/27% for phishing. Scheme and path there are
collection artifacts, not signal - a model that keeps them scores AUC 0.33 on
live OpenPhish data (worse than a coin flip). Dropping them is the fix.
"""
import email.utils
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

import data

MODEL = Path(__file__).parent / "models" / "url_xgb.json"
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
LURE = ("login", "signin", "verify", "secure", "account", "update", "confirm",
        "bank", "otp", "kyc", "wallet", "free", "gift", "paytm", "upi")
# ponytail: no path/scheme features - see module docstring. Add them back only
# with a dataset where both classes carry full URLs (PhishTank + Common Crawl).
FEATURES = ["host_len", "n_dots", "n_hyphen", "n_digits", "digit_ratio",
            "n_special", "n_subdomain", "is_ip", "is_punycode", "tld_len",
            "tld_rare", "entropy", "longest_token", "n_lure", "n_vowel_ratio"]
COMMON_TLD = {"com", "org", "net", "edu", "gov", "in", "uk", "de", "jp", "ru",
              "br", "it", "fr", "au", "ca", "nl", "es", "io", "co"}


def _entropy(s):
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values()) if n else 0.0


def _host(url):
    """urlsplit(url).hostname, lowercased, "" if none.

    "http://[broken" makes urlsplit raise, and phishing produces these - left
    alone it took the whole /sms or /email request down. Then fall back to
    exactly what the phone's UrlModel.host() does, so both still agree.
    """
    try:
        return (urlsplit(url if "://" in url else "http://" + url).hostname or "").lower()
    except ValueError:
        netloc = url.split("://", 1)[1] if "://" in url else url
        netloc = re.split(r"[/?#]", netloc, maxsplit=1)[0].rsplit("@", 1)[-1]
        return (netloc[1:].split("]", 1)[0] if netloc.startswith("[")
                else netloc.split(":", 1)[0]).lower()


def features(url):
    """One URL -> feature dict. Keep this the only place features are defined."""
    host = _host(url)
    host = host[4:] if host.startswith("www.") else host
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    digits = sum(c.isdigit() for c in host)
    letters = sum(c.isalpha() for c in host)
    return {
        "host_len": len(host),
        "n_dots": host.count("."),
        "n_hyphen": host.count("-"),
        "n_digits": digits,
        "digit_ratio": digits / max(len(host), 1),
        "n_special": sum(c in "@_~+%" for c in host),
        "n_subdomain": max(host.count(".") - 1, 0),
        "is_ip": int(bool(IP_RE.match(host))),
        "is_punycode": int("xn--" in host),
        "tld_len": len(tld),
        "tld_rare": int(tld not in COMMON_TLD),
        "entropy": _entropy(host),
        "longest_token": max((len(t) for t in re.split(r"[.\-_]", host)), default=0),
        "n_lure": sum(w in host.lower() for w in LURE),
        "n_vowel_ratio": sum(c in "aeiou" for c in host.lower()) / max(letters, 1),
    }


def featurize(urls):
    return pd.DataFrame([features(u) for u in urls], columns=FEATURES)


def train():
    d = data.load_urls()
    X, y = featurize(d.url), d.label.values

    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          eval_metric="logloss", n_jobs=-1)
    m.fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    m.save_model(MODEL)

    # Honest test: OpenPhish + Tranco are newer than PhiUSIIL = temporal split.
    feed = data.load_url_feed()
    Xf, yf = featurize(feed.url), feed.label.values
    p = m.predict_proba(Xf)[:, 1]

    print(f"train n={len(d):,}  holdout n={len(feed):,}")
    print(f"holdout AUC = {roc_auc_score(yf, p):.3f}")
    print(classification_report(yf, (p > 0.5).astype(int),
                                target_names=["benign", "phishing"], digits=3))
    top = sorted(zip(FEATURES, m.feature_importances_), key=lambda t: -t[1])[:6]
    print("top features:", ", ".join(f"{k} {v:.2f}" for k, v in top))
    return m


@lru_cache(maxsize=1)
def load():
    """Cached: the booster is re-read from disk otherwise, and SMS thread scoring
    calls score() once per conversation. lru_cache is the whole cache layer."""
    m = xgb.XGBClassifier()
    m.load_model(MODEL)
    return m


@lru_cache(maxsize=1)
def _verified():
    """124 verified Indian domains from the ScamShield registry, hosts only."""
    try:
        u = pd.read_csv(data.KB / "verified_urls.csv").url.str.strip().str.lower()
    except OSError:
        return frozenset()  # registry absent -> whitelist simply does not apply
    return frozenset(h[4:] if h.startswith("www.") else h for h in u)


def verified(url):
    """-> True if the host IS a registry domain or a subdomain of one.

    Suffix match on the host, never a substring on the URL: "hdfcbank.com.evil.tk"
    ends with a registry string and must not be trusted, and "nothdfcbank.com"
    ends with "hdfcbank.com" without the dot. Both fail here.
    """
    host = _host(url)
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in _verified())


def score(urls, mail=False, owner=None):
    """-> array of phishing probabilities in [0,1]. This is what fusion consumes.
    mail=True scores with the mail link model (see load_mail) - email only -
    plus its two layers: brand mismatch, and `owner` (sender_owner()) links
    scored as the authenticated sender's own domain.

    Registry domains are pinned to 0.0. One guard here rather than at each call
    site: thread.url_risk, api._links and fusion all read score(), so this is the
    only place that covers them all.

    ponytail: a flat 0.0, not a discount. An open redirect on a real bank domain
    would sail through - the upgrade is to keep the lexical score for URLs whose
    PATH looks like a redirect, once any corpus here carries paths (see the module
    docstring: today the model is domain-level on purpose).
    """
    if mail and load_mail() is not None:
        p = load_mail().predict_proba(mail_featurize([_own(u, owner) for u in urls]))[:, 1]
        p = np.array([max(v, BRAND_FLOOR) if brand_mismatch(u) else v for v, u in zip(p, urls)])
    else:
        p = load().predict_proba(featurize(urls))[:, 1]
    return np.where([verified(u) for u in urls], 0.0, p)


# --- the MAIL link model (url_xgb_v4), used for email only ------------------
# url_xgb learned benign = PhiUSIIL's bare https://www homepages, so any label
# in front of a domain reads as phishing: on real mail it separates genuine
# links from phishing-mail links at AUC 0.61, barely above chance, and flagged
# 19/38 of one user's genuine Gmail links (myaccount.google.com 0.87). v4 splits
# the host with the Public Suffix List and scores the REGISTRABLE domain with
# the same 15 features, the part in front as separate sub_* features, and
# trains with real mail link hosts as benign. Measured 2026-10-11
# (scratch/_train_url_v3.py --v4, log _url_v4.log): mail-vs-phishing-mail AUC
# 0.608 -> 0.685, unseen person's link false alarms 56.7% -> 30.5%, OpenPhish
# recall 80.3 -> 77.2, smishing link catch 44.0 -> 37.5. That last number is why
# SMS (here, thread.url_risk and the phone) stays on url_xgb.
# WEIGHTS DERIVE FROM PRIVATE MAIL (link hosts). Do not publish them.
MAIL_MODEL = MODEL.with_name("url_xgb_v4.json")
PSL_FILE = data.DATA / "URL" / "public_suffix_list.dat"   # version in its header
EMBEDDED_IP = re.compile(r"(?<!\d)\d{1,3}[-.]\d{1,3}[-.]\d{1,3}[-.]\d{1,3}(?!\d)")
SUB_SHAPE = ["sub_len", "sub_labels", "sub_digits", "sub_hyphen", "sub_entropy", "sub_longest"]
MAIL_FEATURES = FEATURES + ["sub_lure", "sub_ip"] + SUB_SHAPE


@lru_cache(maxsize=1)
def _psl():
    try:
        lines = PSL_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(s.split()[0].lower() for s in map(str.strip, lines)
                     if s and not s.startswith("//"))


def split(host):
    """host -> (sub, registrable domain) by the PSL algorithm: longest rule,
    '!' exceptions win, '*' wildcards, default rule '*'. PRIVATE suffixes count,
    so on github.io / s3.amazonaws.com the customer's label is the registrable
    domain. An IP, or a host that is itself a suffix, is all registrable."""
    host = host.lower().strip(".")
    host = host[4:] if host.startswith("www.") else host
    if IP_RE.match(host) or "." not in host:
        return "", host
    labels, rules = host.split("."), _psl()
    n, suffix = len(labels), 1
    for i in range(n):
        d = ".".join(labels[i:])
        if "!" + d in rules:
            suffix = n - i - 1
            break
        if d in rules or (i + 1 < n and "*." + ".".join(labels[i + 1:]) in rules):
            suffix = n - i
            break
    if suffix >= n:
        return "", host
    return ".".join(labels[:n - suffix - 1]), ".".join(labels[n - suffix - 1:])


def mail_features(url):
    """features() of the registrable domain, plus what is in front of it."""
    sub, reg = split(_host(url))
    f = features("http://" + reg)
    f.update(sub_lure=sum(w in sub for w in LURE),
             sub_ip=int(bool(EMBEDDED_IP.search(sub))),
             sub_len=len(sub), sub_labels=sub.count(".") + bool(sub),
             sub_digits=sum(c.isdigit() for c in sub), sub_hyphen=sub.count("-"),
             sub_entropy=_entropy(sub),
             sub_longest=max((len(t) for t in re.split(r"[.\-_]", sub)), default=0))
    return f


def mail_featurize(urls):
    return pd.DataFrame([mail_features(u) for u in urls], columns=MAIL_FEATURES)


# --- two mail-only layers on top of url_xgb_v4 ------------------------------
# Measured per mail (worst visible link) by scratch/_url_layers_eval.py, log
# _url_layers.log, 2026-10-11. Unseen person's genuine mail flagged 15.4% ->
# 3.8%; phishing_pot mail caught 25.2% -> 24.3%.
#  BRAND MISMATCH: a host token is a brand and the registrable domain is not
#   one of the brand's -> score >= BRAND_FLOOR (Kintis et al., CCS 2017).
#   Brands = Tranco top-1000 first labels + ScamShield registry entities; a
#   token used inside > GENERIC hyphenated top-1M labels is a word, not a brand
#   (login 56, bank 153 vs apple 28, paypal 7). Cutoff set from Tranco alone.
#  SENDER'S OWN LINK: DMARC passed AND the From domain is an established
#   (top-1M) domain AND the link goes to that registrable domain -> score the
#   domain alone, without the tracking part in front. The top-1M guard was
#   chosen after seeing that plain DMARC cost 2.2 points of catch (phishers
#   pass DMARC on domains they own); with it the cost is 1.7, and brand
#   mismatch wins back 0.8.
TRANCO = data.DATA / "URL" / "tranco_N2KLW.csv"
GENERIC, TOP_BRANDS, BRAND_FLOOR = 40, 1000, 0.9


@lru_cache(maxsize=1)
def _brands():
    """-> (brand token -> its registrable domains, top-1M registrable domains).
    Both empty when Tranco or the PSL is missing: the layers then abstain."""
    if not _psl() or not TRANCO.exists():
        return {}, frozenset()
    regs = [split(d)[1] for d in pd.read_csv(TRANCO, names=["rank", "d"]).d.str.lower()]
    first = [r.split(".")[0] for r in regs]
    compound = Counter(t for f in first if "-" in f for t in set(f.split("-")))
    by_token = {}
    for r, f in zip(regs, first):
        for t in {f, *f.split("-")}:
            by_token.setdefault(t, set()).add(r)
    brands = {f: by_token[f] for f in first[:TOP_BRANDS] if len(f) >= 3 and compound[f] <= GENERIC}
    try:
        kb = pd.read_csv(data.KB / "verified_urls.csv")
    except OSError:
        kb = pd.DataFrame(columns=["url", "entity_id"])
    for _, g in kb.groupby("entity_id"):
        own = {split(_host(u))[1] for u in g.url}
        for r in own:
            f = r.split(".")[0]
            if len(f) >= 3 and compound[f] <= GENERIC:
                brands.setdefault(f, set()).update(own)
    return brands, frozenset(regs)


def brand_mismatch(url):
    """-> the brand a host names but does not belong to (paypal-login.xyz ->
    'paypal'), else None."""
    brands = _brands()[0]
    sub, reg = split(_host(url))
    for t in set(re.split(r"[.\-]", sub)) | set(reg.split(".")[0].split("-")):
        if t in brands and reg not in brands[t]:
            return t
    return None


def sender_owner(sender, dmarc):
    """-> the From address's registrable domain when its links may be scored
    as the sender's own: DMARC passed and the domain is top-1M. Else None."""
    addr = email.utils.parseaddr(str(sender or ""))[1]
    if str(dmarc or "").lower() != "pass" or "@" not in addr:
        return None
    reg = split(addr.rsplit("@", 1)[1])[1]
    return reg if reg in _brands()[1] else None


def _own(url, owner):
    """The URL to score: the bare registrable domain when it is the sender's."""
    reg = split(_host(url))[1] if owner else None
    return f"https://{reg}" if owner and reg == owner else url


@lru_cache(maxsize=1)
def load_mail():
    """url_xgb_v4, or None when it or the PSL is missing - email then falls
    back to url_xgb rather than scoring PSL-less splits it never trained on."""
    if not MAIL_MODEL.exists() or not _psl():
        return None
    m = xgb.XGBClassifier()
    m.load_model(MAIL_MODEL)
    return m


# §4 reason strings: one short phrase per feature, so a verdict reads as English
# rather than as "n_lure=3". Only the features that fire get named.
PHRASE = {
    "host_len": "unusually long host name",
    "n_dots": "many dot-separated parts",
    "n_hyphen": "hyphen-stuffed host",
    "n_digits": "digits in the host",
    "digit_ratio": "host is mostly digits",
    "n_special": "odd characters in the host",
    "n_subdomain": "deep subdomain nesting",
    "is_ip": "raw IP address instead of a domain",
    "is_punycode": "punycode (look-alike) domain",
    "tld_len": "unusual TLD length",
    "tld_rare": "rare top-level domain",
    "entropy": "random-looking host string",
    "longest_token": "one very long host token",
    "n_lure": "bank/payment lure words in the host",
    "n_vowel_ratio": "host reads as non-word letters",
    # mail model only: the part in front of the registrable domain
    "sub_lure": "login/bank/payment words in front of the domain",
    "sub_ip": "an IP address hidden in front of the domain",
    "sub_len": "a long string in front of the domain",
    "sub_labels": "many parts in front of the domain",
    "sub_digits": "digits in front of the domain",
    "sub_hyphen": "hyphens in front of the domain",
    "sub_entropy": "a random-looking string in front of the domain",
    "sub_longest": "one very long chunk in front of the domain",
}


def explain(url, top_k=3, mail=False, owner=None):
    """-> (probability, reason string, named contributions). §4 requires a
    reason on every verdict; the named list is the same content as data, for a
    UI that wants to say WHY a link looks wrong rather than print a sentence.

    Contributions come from xgboost's own `pred_contribs=True`, which is exact
    TreeSHAP - same numbers as shap.TreeExplainer, without loading shap for a
    single row. Use the shap package for the report's summary plot, not here.
    """
    mail = mail and load_mail() is not None
    m, names = (load_mail(), MAIL_FEATURES) if mail else (load(), FEATURES)
    x = (mail_featurize if mail else featurize)([_own(url, owner) if mail else url])
    p = float(m.predict_proba(x)[0, 1])
    # The brand layer outranks the lexical reasons: it is the more specific
    # finding, and score() floors the risk on it.
    brand = brand_mismatch(url) if mail else None
    if brand:
        why = f"it names {brand} but is not one of {brand}'s websites"
        return max(p, BRAND_FLOOR), f"link looks phishing: {why}", [why]
    # A benign host has no positive evidence worth naming - the top contributions
    # are then just the least-negative ones, which reads as an accusation. Say
    # nothing instead; the reason string exists to justify flagging.
    if p < 0.5:
        return p, "link looks benign: nothing in the host name stands out", []

    # last column is the bias term, drop it
    contrib = m.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0][:-1]
    top = sorted(range(len(names)), key=lambda i: -contrib[i])[:top_k]
    named = [PHRASE[names[i]] for i in top if contrib[i] > 0.05]
    return (p, "link looks phishing" + (": " + ", ".join(named) if named else ""),
            named)


if __name__ == "__main__":
    train()

    assert features("http://192.168.1.4:8080/secure-login/verify")["is_ip"]
    assert features("http://kyc-verify.paytm.login.tk/x")["n_lure"] >= 3
    # path must not change the score - features are domain-level by design
    assert features("http://a.b.com/x/y?z=1") == features("https://www.a.b.com")

    # whitelist: the two ways a suffix match gets it wrong must both be rejected
    assert verified("https://hdfcbank.com/statement") and verified("https://net.hdfcbank.com/x")
    assert not verified("https://hdfcbank.com.evil.tk/login"), "subdomain-of-attacker"
    assert not verified("https://nothdfcbank.com"), "suffix without the dot"
    assert score(["https://hdfcbank.com/x"])[0] == 0.0
    assert score(["https://hdfcbank.com.evil.tk/login"])[0] > 0.5

    s = score(["https://www.google.com", "http://paytm-kyc-update.verify-now.tk/login"])
    print("google=%.2f  fake-kyc=%.2f" % tuple(s))
    assert s[1] > s[0], "obvious lure must outscore google"

    for u in ["http://paytm-kyc-update.verify-now.tk/login",
              "http://192.168.1.4:8080/secure-login", "https://www.google.com"]:
        p, why = explain(u)
        print(f"  {p:.2f}  {u}\n        {why}")
    # the two things that make that host obvious must be the things it names
    _, why = explain("http://paytm-kyc-update.verify-now.tk/login")
    assert "lure" in why or "rare top-level" in why, why
    # is_ip shares its credit with n_digits/digit_ratio - any of the three is the
    # same finding, so the assert follows the evidence rather than the feature name
    ip_why = explain("http://192.168.1.4:8080/secure-login")[1]
    assert "IP" in ip_why or "digit" in ip_why, ip_why
    assert "stands out" in explain("https://www.google.com")[1]
    print("ok")
