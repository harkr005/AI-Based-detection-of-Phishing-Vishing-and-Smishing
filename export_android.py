"""Export the SMS pipeline to the Android app's assets, and the parity fixtures
that prove the Kotlin port IS the Python model.

§10 Mode 1: SMS is the only channel whose content arrives on the device unbidden
and continuously, so it is the only one where on-device inference buys anything.
Nothing here is converted - the three SMS channels are a linear model, fifteen
lexical features and a set of regexes, so exporting them means writing the
numbers down.

    assets/sms_charlr.bin        TF-IDF char_wb (3,5) + LogisticRegression
    assets/url_xgb.bin           the 400-tree lexical URL model, flattened
    assets/dlt_registry.txt      behaviour.DLT_REGISTRY (the TRAI header registry)
    assets/verified_domains.txt  url_model._verified(), hosts pinned to 0.0
    test/resources/sms_parity.json    sklearn's own probabilities
    test/resources/scan_parity.json   features, scores and whole /sms verdicts

Run it after ANY retrain, or the phone quietly scores with last month's model:
    python export_android.py
"""
import datetime as dt
import json
import struct
from pathlib import Path

import numpy as np
import xgboost as xgb

import behaviour
import sms_model
import url_model
import voice_model

ROOT = Path(__file__).parent
ASSETS = ROOT / "android/app/src/main/assets"
FIXTURES = ROOT / "android/app/src/test/resources"

# Parity cases, chosen to hit what a re-implementation gets wrong rather than to
# look like inbox traffic: the short-word quirk (a word shorter than n yields
# ONE n-gram and no longer ones), NFKC folding, zero-width strip, homoglyphs,
# repeated whitespace, and text with no in-vocabulary n-gram at all (whose
# tf-idf row is all zeros, so the L2 norm must not divide by it).
CASES = [
    "",
    " ",
    "hi",
    "a b c",
    "ok",
    "Hey, are we still meeting at 6 for dinner?",
    "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked",
    "URGENT!! Your a/c will be SUSPENDED. Call 9876543210 immediately",
    "Congratulations!!! You have won Rs 25,00,000 in the KBC lottery draw",
    "481920 is your OTP for a transfer of Rs 25,000 from A/c 912345678901",
    "Dear customer [account], [amount] debited. Not you? [link]",
    "Yоur аccоunt is blоcked",          # homoglyphs
    "Your acc​ount is bl‌ocked﻿",             # zero-width
    "ＵＰＤＡＴＥ your KYC ①",  # NFKC: fullwidth + circled digit
    "Y0ur acc0unt is bl0cked - ver1fy n0w",                  # leet, survives normalize
    "K Y C  update\t\tneeded\nnow",                          # split/collapse
    "आपका खाता बंद हो जाएगा",  # Devanagari
    "\U0001f6a8\U0001f6a8\U0001f6a8",                        # emoji only, likely all-OOV
    "zzzzzzzzzzzz",                                          # long OOV word
    "Click http://bit.ly/3xK9qLm to claim",
]

# Senders, for the rules that read the sender rather than the text. Each line is
# a shape behaviour.sid/entity/registered has to agree about.
SENDERS = [
    # The first four are the shapes a carrier actually delivers since 6 May 2025
    # (Mohak's own inbox): operator prefix, entity, category suffix. "HDFCBK"
    # bare is kept deliberately - it must stay a non-anomaly and stop being
    # trusted. AX-ICICIT-S is real and NOT in our registry file, which is the
    # registry's incompleteness pinned as a test rather than argued about.
    "AX-ICICIT-S", "DM-AXISBK-T", "VM-HDFCBK-S", "JD-ATMSBI-T",
    "", "VM-HDFCBK", "AD-AXISBK-S", "HDFCBK", "VM-HDFCBNK", "AX-KYCUPD",
    "SUBWAY", "Zepto", "VM-2204", "BZ-CBSSBI",
    "9876543210", "+919876543210", "+91 98765 43210", "(987) 654-3210",
    "0123456789", "12345", "DM-654321", "VK-ICICIB", "TX-SBIINB-T", "unknown",
]

URLS = [
    "http://sbi-kyc-verify.tk/login",
    "https://hdfcbank.com/statement",
    "https://net.hdfcbank.com/x",
    "https://nothdfcbank.com/x",
    "https://hdfcbank.com.evil.tk/login",
    "http://192.168.1.4:8080/secure-login/verify",
    "http://xn--80ak6aa92e.com",
    "www.paytm-kyc-update.xyz",
    "https://www.google.com",
    "bit.ly/3xK9qLm",
    "http://a.b.com/x/y?z=1",
    "https://amazon.in/orders",
    "http://secure-login-verify-account-update.confirm-bank.top/a",
    "https://irctc.co.in",
    "flipkart.com",
]


def real_samples(n=30):
    """Real corpus rows too - the authored cases above share an author with the
    port, and that is the same trap the 50 vishing transcripts documented."""
    try:
        df = sms_model.data.load_sms()
    except Exception as e:  # dataset not on this machine; the authored cases stand
        print(f"corpus not loaded ({e}) - exporting authored cases only")
        return []
    # Public corpora only: this file is tracked in git, the inbox is private.
    s = df[df.source != "inbox"].text.dropna().astype(str)
    s = s[s.str.len() > 0]
    return s.sample(n=min(n, len(s)), random_state=0).tolist()


def write_floats(fh, *arrays):
    for a in arrays:
        a = np.asarray(a, np.float32).ravel()
        fh.write(struct.pack(f"<{a.size}f", *a))


def write_ints(fh, *arrays):
    for a in arrays:
        a = np.asarray(a, np.int32).ravel()
        fh.write(struct.pack(f"<{a.size}i", *a))


def export_sms():
    """Format (one file so the app opens one asset):
        "smslr1 <n_vocab> <n_classes>\\n"
        <n_vocab> vocabulary terms, one per line, in column order
        float32 LE: idf[n_vocab], intercept[n_classes], coef[n_classes][n_vocab]

    Terms never contain a newline: char_wb splits on whitespace before it cuts
    n-grams, so the only space a term can hold is the padding sklearn adds.
    """
    vec, clf = sms_model.load().steps[0][1], sms_model.load().steps[1][1]
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    assert not any("\n" in t for t in terms), "a term holds a newline; format broken"
    assert clf.coef_.shape == (len(clf.classes_), len(terms))

    out = ASSETS / "sms_charlr.bin"
    with out.open("wb") as fh:
        fh.write(f"smslr1 {len(terms)} {len(clf.classes_)}\n".encode())
        fh.write(("\n".join(terms) + "\n").encode())
        write_floats(fh, vec.idf_, clf.intercept_, clf.coef_)
    print(f"  sms_charlr.bin   {out.stat().st_size / 1e6:5.2f} MB  "
          f"{len(terms):,} n-grams x {len(clf.classes_)} classes")


def export_url():
    """Format:
        "urlxgb2 <n_trees> <n_nodes> <max_depth> <base_margin>\\n"
        int32 LE:  tree_start[n_trees + 1], left[n_nodes], right[n_nodes],
                   feature[n_nodes]
        float32 LE: threshold[n_nodes], leaf[n_nodes], cover[n_nodes]
    Children are GLOBAL node indices, so scoring is one flat loop; left == -1
    marks a leaf. Missing-value routing (`default_left`) is dropped: these
    fifteen features are computed from a string and none of them can be NaN.

    `cover` (the node's summed hessian) and `max_depth` are there for TreeSHAP,
    which needs to know how much of the training data went each way at every
    split, and a bound on the path buffer it walks with.

    `base_margin` is DERIVED, not read from `base_score`: xgboost has changed
    whether that field is stored in probability or margin space, and a wrong
    intercept is a silently shifted model rather than an error. Taken as
    (margin - sum of leaves) on real URLs and checked constant across them.
    """
    booster = url_model.load().get_booster()
    trees = json.loads(booster.save_raw("json"))[
        "learner"]["gradient_booster"]["model"]["trees"]

    start, left, right, feature, threshold, leaf, cover = [0], [], [], [], [], [], []
    depth = 0
    for t in trees:
        base = start[-1]
        n = len(t["left_children"])
        left += [-1 if c == -1 else c + base for c in t["left_children"]]
        right += [-1 if c == -1 else c + base for c in t["right_children"]]
        feature += list(t["split_indices"])
        threshold += list(t["split_conditions"])
        leaf += list(t["base_weights"])
        cover += list(t["sum_hessian"])
        start.append(base + n)
        # Depth of the deepest node, measured rather than taken from max_depth:
        # the phone sizes its SHAP path buffer from it.
        at = {0: 0}
        for i, (l, r) in enumerate(zip(t["left_children"], t["right_children"])):
            if l != -1:
                at[l] = at[r] = at[i] + 1
        depth = max(depth, max(at.values()))

    # float32 on BOTH sides of the comparison. xgboost casts features to float32
    # before it tests a split, and on a feature like digit_ratio or entropy the
    # float64 value falls the other side of a threshold often enough to move the
    # margin by 0.16 - which is what the constant-intercept assert first caught.
    # SmsModel's Kotlin twin therefore compares in Float, not Double.
    cut = np.asarray(threshold, np.float32)

    def walk(row):
        total = 0.0
        for t in range(len(trees)):
            i = start[t]
            while left[i] != -1:
                i = left[i] if row[feature[i]] < cut[i] else right[i]
            total += leaf[i]
        return total

    X = url_model.featurize(URLS)
    margin = booster.predict(xgb.DMatrix(X), output_margin=True)
    bias = [m - walk(row) for m, row in zip(margin, X.values.astype(np.float32))]
    assert np.ptp(bias) < 1e-4, f"intercept is not constant: {np.ptp(bias)}"
    base_margin = float(np.mean(bias))

    out = ASSETS / "url_xgb.bin"
    with out.open("wb") as fh:
        fh.write(f"urlxgb2 {len(trees)} {len(left)} {depth} {base_margin!r}\n".encode())
        write_ints(fh, start, left, right, feature)
        write_floats(fh, threshold, leaf, cover)
    print(f"  url_xgb.bin      {out.stat().st_size / 1e6:5.2f} MB  "
          f"{len(trees)} trees, {len(left):,} nodes, depth {depth}, "
          f"base {base_margin:+.4f}")


def export_registries():
    for name, values in (("dlt_registry.txt", behaviour.DLT_REGISTRY),
                         ("verified_domains.txt", url_model._verified())):
        (ASSETS / name).write_text("\n".join(sorted(values)) + "\n", encoding="utf-8")
        print(f"  {name:<16} {len(values):5,} entries")


def sms_fixture():
    texts = CASES + real_samples()
    # score() runs the same _clean() the training path did, so the fixture
    # measures the whole pipeline the phone has to reproduce - normalize
    # included, which is where a port silently drifts.
    probs = sms_model.score(texts)
    explained = [sms_model.explain(t) for t in texts]
    return {"labels": sms_model.LABELS,
            "cases": [{"text": t,
                       "probs": [round(float(p), 9) for p in row],
                       "reason": why, "terms": terms}
                      for t, row, (_, _, why, terms) in zip(texts, probs, explained)]}


def scan_fixture():
    """The three ported layers on their own, then the whole verdict.

    The whole-verdict cases call `api.sms` itself rather than re-assembling the
    steps here - a fixture that repeats the composition under test would agree
    with a port that got the composition wrong.
    """
    import api  # heavy (torch); only needed for the end-to-end cases

    # `contribs` is xgboost's own exact TreeSHAP, the thing the phone now walks
    # the trees a second time to reproduce. `named` and `reason` are what it
    # turns into on screen, and they are what actually has to match: two
    # contribution vectors that differ in the eighth place still have to pick
    # the same three phrases in the same order.
    booster = url_model.load().get_booster()
    shap = booster.predict(xgb.DMatrix(url_model.featurize(URLS)),
                           pred_contribs=True)[:, :-1]
    url_cases = []
    for u, contrib in zip(URLS, shap):
        _, reason, named = url_model.explain(u)
        url_cases.append({
            "url": u,
            "features": {k: round(float(v), 9)
                         for k, v in url_model.features(u).items()},
            "verified": bool(url_model.verified(u)),
            "score": round(float(url_model.score([u])[0]), 9),
            "contribs": [round(float(c), 9) for c in contrib],
            "named": named,
            "reason": reason,
        })

    # first_contact / has_url / hour / burst, spread over the senders so every
    # rule fires somewhere and abstains somewhere.
    behaviour_cases = []
    for i, s in enumerate(SENDERS):
        for text in ("Your KYC will be blocked, verify now",
                     "Running late, see you at 8"):
            first = [True, False, None][i % 3]
            behaviour_cases.append({
                "sender": s, "text": text,
                "first_contact": first, "has_url": i % 2 == 0,
                "hour": [None, 3, 14, 23][i % 4], "burst": [0, 1, 6][i % 3],
                "sid": behaviour.sid(s), "entity": behaviour.entity(s),
                "registered": bool(behaviour.registered(s)),
            })
    for c in behaviour_cases:
        f = behaviour.sms_features(c["sender"], c["text"],
                                   first_contact=c["first_contact"],
                                   has_url=c["has_url"], hour=c["hour"],
                                   burst=c["burst"])
        score, tags = behaviour.sms_score(f)
        c["features"] = {k: round(float(v), 9) for k, v in f.items()}
        c["score"], c["tags"] = round(float(score), 9), tags

    evidence_cases = [{"text": t, "evidence": round(float(voice_model.evidence_risk(t)), 9)}
                      for t in CASES + [
                          "Please share the OTP you received to complete the KYC",
                          "Never share your OTP with anyone, not even bank staff",
                          "We will never ask for your PIN or CVV over a call",
                          "Your account will be blocked within 2 hours",
                          "Install AnyDesk and share your screen with the officer",
                          "Transfer 50000 rupees to this account immediately",
                          "Do not tell anyone about this case, it is confidential",
                          "Scan this QR code to receive your refund",
                          "Your OTP is 481920. Do not share it with anyone.",
                      ]]

    # first_contact stays True/False here. On None, api.py's tag names how many
    # senders ITS store knows, which is a sentence about a server-side profile
    # the phone does not have - LocalScanTest covers the phone's own wording.
    scan_cases = []
    for i, (sender, text) in enumerate([
        ("VM-HDFCBK", "Your a/c is debited by Rs 2,500. Not you? Call 18002026161"),
        ("9876543210", "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk"),
        ("9876543210", "Mum is in hospital, please send Rs 15,000 to account 445566778899"),
        ("AD-SWIGGY-S", "Your order is on the way, arriving in 12 minutes"),
        ("VM-HDFCBK", "Never share your OTP, PIN or CVV with anyone"),
        ("AX-KYCUPD", "URGENT: KYC pending. Click https://kyc-verify.paytm.login.tk now"),
        ("+919812345678", "Congratulations! You won Rs 25,00,000. Send processing fee"),
        ("SUBWAY", "Buy one get one free on all subs this weekend"),
        ("unknown", "hi"),
        ("VK-ICICIB", "Dear customer, your account is blocked. Verify at http://icici-kyc.top"),
        # The case fusion.deescalate exists for, and the only one in this list
        # that exercises it: a registered header, no link, no scam act, and a
        # content model that flags it anyway (SUSPICIOUS 0.611 with the gate
        # disarmed). Without a case like this the gate is untested.
        ("VM-HDFCBK", "Dear Customer, your KYC verification is complete. No action needed."),
    ]):
        first = i % 2 == 0
        hour = [9, 2, 23, 14][i % 4]
        sent = dt.datetime(2026, 9, 22, hour, 30, tzinfo=dt.timezone.utc)
        api._STORES.clear()  # every case scored against a cold store, as the phone is
        # verified_sender=False, matching what LocalScan now passes. The phone
        # is entitled to True - a carrier stamped the header - but the gate was
        # measured on 2026-09-22 at zero gain and 12 of 200 smishing cleared, so
        # it is disarmed on every path. Flip both together or parity breaks.
        out = api.sms(api.SmsIn(text=text, sender=sender, first_contact=first,
                                sent_at=sent, user_id="parity",
                                verified_sender=False))
        scan_cases.append({"sender": sender, "text": text, "hour": hour,
                           "first_contact": first, "expected": out})

    return {"url_cases": url_cases, "behaviour_cases": behaviour_cases,
            "evidence_cases": evidence_cases, "scan_cases": scan_cases}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("assets:")
    export_sms()
    export_url()
    export_registries()

    print("fixtures:")
    for name, payload in (("sms_parity.json", sms_fixture()),
                          ("scan_parity.json", scan_fixture())):
        (FIXTURES / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8")
        n = sum(len(v) for v in payload.values() if isinstance(v, list))
        print(f"  {name:<16} {n:5,} cases")


if __name__ == "__main__":
    main()
