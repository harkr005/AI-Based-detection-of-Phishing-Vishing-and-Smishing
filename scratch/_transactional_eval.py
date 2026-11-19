"""Does anything fix the legitimate-transactional false positives?

Three moves, measured separately and together (see the memory note "instance #6
of the corpus trap"):

  1. de-escalation gate in fusion - evidence can currently never clear a
     saturated content model; nlp's 0.50 weight puts a floor at ~0.50
  2. act/evidence layer - score the ACT (an ask, a threat, a payment demand)
     instead of the topic, reusing voice_model's rules
  3. transactional ham eval set - Datasets/smishing/transactional_eval.json

CAVEAT that must travel with every smishing number below: both content models
were TRAINED on this corpus, so smishing recall here is in-sample and
optimistic. It is used only to price the RELATIVE cost of each move, never as a
detection score.
"""
import data  # noqa: F401  # before torch
import json
import re
from pathlib import Path

import numpy as np

import behaviour
import email_model
import fusion
import sms_model
import url_model
import voice_model

EVAL = Path(__file__).parent / "Datasets" / "smishing" / "transactional_eval.json"
RNG = np.random.default_rng(0)

# Smishing senders. Not all scammers use bare numbers - assuming a clean split
# would manufacture the result - so a quarter get look-alike headers.
FAKE_HEADERS = ["VM-HDFCBNK", "AX-KYCUPD", "AD-SBIALT", "VK-ICICIB"]


ACT_NAMES = ["ASK", "THREAT", "REMOTE", "ENTITY", "MONEY", "SECRECY"]

# ENTITY matches the mere MENTION of otp/kyc/aadhaar/card - a topic, not an act.
# Measured: it fires on 20% of legitimate transactional SMS and 3.5% of
# smishing, i.e. inverted, for the obvious reason that a real OTP message is
# the thing most likely to contain the word "OTP". Same shape as `urgency` in
# the 7B distillation. It stays in RULES for scoring; it must not be treated as
# EVIDENCE of an act.
EVIDENCE = {"ASK", "THREAT", "REMOTE", "MONEY", "SECRECY"}


def act_risk(text):
    """voice_model's rules on a whole message. §6's keyword/NER layer is
    channel-agnostic: an ask for an OTP is the same act in speech or in SMS."""
    return voice_model.rule_score(text), [
        n for n, (w, rx) in zip(ACT_NAMES, voice_model.RULES) if rx.search(text)]


def evidence_risk(text):
    """Act evidence only - ENTITY excluded. This is what the gate consults."""
    return min(1.0, sum(w for n, (w, rx) in zip(ACT_NAMES, voice_model.RULES)
                        if n in EVIDENCE and rx.search(text)))


def links_of(text):
    urls = voice_model_url_re.findall(text or "")
    if not urls:
        return None
    return float(max(url_model.score(urls)))


import thread as _thread  # noqa: E402
voice_model_url_re = _thread.URL_RE


def sms_behaviour(sender, text, has_url):
    if not sender:
        return None, []
    f = behaviour.sms_features(sender, text, first_contact=True,
                               has_url=has_url, hour=12)
    return behaviour.sms_score(f)


def trusted(sender):
    """Positive evidence of a legitimate sender, not merely absence of anomaly.
    A TRAI-registered DLT header is a traceable, accountable entity - which is
    exactly the property Gmail gets from DMARC alignment.

    Shape used to be enough here while the shipped gate already required registry
    membership, so this script measured a weaker gate than the one that runs:
    every look-alike below (VM-HDFCBNK, AX-KYCUPD) passed a shape test.

    Note what this harness now models: since 2026-08-21 api.sms also requires
    verified_sender, so the shipped web path never de-escalates at all. Every
    number below describes §10 GATEWAY mode, where a carrier supplies the
    header - not the form a user types into today."""
    return behaviour.registered(sender)


def deescalate(f, acts, url, sender, cap=0.30):
    """The move §8 is missing. Escalation exists (behaviour lifts content-clean
    mail); nothing symmetric lets evidence clear a content false positive.

    Fires only when EVERY independent channel is clean: a registered sender, no
    malicious link, and - the load-bearing one - no scam act anywhere in the
    text. Content saying "phishing" with no ask, no threat and no payment demand
    is the ACL paper's background-without-evidence case.
    """
    if f["verdict"] == "SAFE":
        return f, False
    if not (trusted(sender) and acts == 0.0 and (url is None or url < 0.5)):
        return f, False
    g = dict(f, risk=min(f["risk"], cap))
    g["verdict"] = fusion.band(g["risk"])
    g["reason"] = f["reason"] + "; registered sender, clean link, no scam act - de-escalated"
    return g, True


def score(text, sender, nlp_probs, mode):
    """One message -> verdict dict under one configuration."""
    url = links_of(text)
    beh, _ = sms_behaviour(sender, text, url is not None)
    a = evidence_risk(text)

    if mode in ("acts", "all_acts"):
        # Acts REPLACE the content model: p = [1-a, 0, a]
        probs = np.array([1 - a, 0.0, a])
    elif mode == "gated":
        # Content model damped when no act is present in the text at all.
        damp = 1.0 if a > 0 else 0.35
        p = np.asarray(nlp_probs, float).copy()
        p[2] *= damp
        p = p / p.sum()
        probs = p
    else:
        probs = np.asarray(nlp_probs, float)

    f = fusion.fuse(probs, url=url, behaviour=beh, malicious_idx=2)
    if mode in ("deesc", "all", "all_acts"):
        f, _ = deescalate(f, a, url, sender)
    return f


MODES = ["baseline", "deesc", "acts", "gated", "all"]


def run(items, nlp_probs, label):
    """-> {mode: flagged_fraction}. 'flagged' = not SAFE (the §3 band a user sees)."""
    out = {}
    for m in MODES:
        v = [score(t["text"], t.get("sender"), p, m)
             for t, p in zip(items, nlp_probs)]
        out[m] = np.mean([x["verdict"] != "SAFE" for x in v])
    return out


def main():
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    tx = ev["sms"]

    # --- the three populations -------------------------------------------
    corpus = data.load_sms()
    sm = corpus[corpus.label_3 == "smishing"].sample(200, random_state=0)
    ham = corpus[corpus.label_3 == "ham"].sample(200, random_state=0)

    smish = [{"text": t, "sender": (RNG.choice(FAKE_HEADERS) if RNG.random() < .25
                                    else "9" + "".join(RNG.choice(list("0123456789"), 9)))}
             for t in sm.text.astype(str)]
    conv = [{"text": t, "sender": "9" + "".join(RNG.choice(list("0123456789"), 9))}
            for t in ham.text.astype(str)]

    print("scoring content model on 3 populations...")
    p_tx = sms_model.score([t["text"] for t in tx])
    p_sm = sms_model.score([t["text"] for t in smish])
    p_cv = sms_model.score([t["text"] for t in conv])

    r_tx = run(tx, p_tx, "transactional")
    r_sm = run(smish, p_sm, "smishing")
    r_cv = run(conv, p_cv, "conversational")

    print(f"\n{'':10s} {'transactional':>14s} {'conversational':>15s} {'smishing':>10s}")
    print(f"{'':10s} {'ham FP %':>14s} {'ham FP %':>15s} {'recall %':>10s}")
    for m in MODES:
        print(f"{m:10s} {r_tx[m]*100:13.1f}% {r_cv[m]*100:14.1f}% {r_sm[m]*100:9.1f}%")

    # --- content layer alone, no sender: isolates the act layer -----------
    print("\nNo sender at all (behaviour silent) - isolates the content/act layer:")
    bare_tx = [{"text": t["text"]} for t in tx]
    bare_sm = [{"text": t["text"]} for t in smish]
    b_tx, b_sm = run(bare_tx, p_tx, "tx"), run(bare_sm, p_sm, "sm")
    print(f"{'':10s} {'transactional FP %':>20s} {'smishing recall %':>19s}")
    for m in MODES:
        print(f"{m:10s} {b_tx[m]*100:19.1f}% {b_sm[m]*100:18.1f}%")

    # --- what the act layer sees -----------------------------------------
    a_tx = np.array([act_risk(t["text"])[0] for t in tx])
    a_sm = np.array([act_risk(t["text"])[0] for t in smish])
    a_cv = np.array([act_risk(t["text"])[0] for t in conv])
    print(f"\nact layer alone:  transactional mean {a_tx.mean():.3f} "
          f"(fires on {np.mean(a_tx>0)*100:.0f}%) | smishing {a_sm.mean():.3f} "
          f"(fires on {np.mean(a_sm>0)*100:.0f}%) | conversational {a_cv.mean():.3f} "
          f"(fires on {np.mean(a_cv>0)*100:.0f}%)")

    # separability of each layer on transactional-vs-smishing
    def auc(pos, neg):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos, neg]
        o = np.argsort(s)
        r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    print(f"AUC smishing vs transactional-ham:  content {auc(p_sm[:,2], p_tx[:,2]):.3f}"
          f"  |  acts {auc(a_sm, a_tx):.3f}")
    print(f"AUC smishing vs conversational-ham: content {auc(p_sm[:,2], p_cv[:,2]):.3f}"
          f"  |  acts {auc(a_sm, a_cv):.3f}")

    # --- which acts carry it ---------------------------------------------
    names = ["ASK", "THREAT", "REMOTE", "ENTITY", "MONEY", "SECRECY"]
    print("\nper-act firing rate (smishing / transactional ham / conversational ham):")
    for n, (w, rx) in zip(names, voice_model.RULES):
        f_sm = np.mean([bool(rx.search(t["text"])) for t in smish])
        f_tx = np.mean([bool(rx.search(t["text"])) for t in tx])
        f_cv = np.mean([bool(rx.search(t["text"])) for t in conv])
        print(f"  {n:8s} w={w:.2f}  {f_sm*100:5.1f}%  {f_tx*100:5.1f}%  {f_cv*100:5.1f}%")


if __name__ == "__main__":
    main()
