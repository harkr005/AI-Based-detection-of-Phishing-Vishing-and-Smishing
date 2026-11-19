"""Fusion / risk scorer (spec §8) - one verdict from content, URL and behaviour.

Adaptive weighting is the §8 rule verbatim: when the content model is unsure,
trust structure instead. That is the whole Phase 2 thesis in three lines of
arithmetic - content changes every campaign, behaviour does not.

Two fusers, as §8 demands both be reported:
  fuse()    weighted sum, no fitting, ships as-is
  Stacker   logistic regression over the same inputs, needs labelled events

Uncertainty routing (§8): high prediction entropy never lands on SAFE. A model
that cannot decide is not evidence of innocence.

Run this file for the Phase 2 experiment: does the behaviour layer break the
0.609 spam-recall ceiling the content model pinned at in Phase 1?
"""
import numpy as np

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data  # noqa: F401

SUSPICIOUS, PHISHING = 0.35, 0.70    # §3 bands
LOW_CONF = 0.60                      # §8 switch point
HIGH_ENTROPY = 0.80                  # of max entropy; above this -> never SAFE

# §8, verbatim. Structure carries the decision when content is uncertain.
W_UNSURE = {"nlp": 0.25, "url": 0.30, "behaviour": 0.45}
W_SURE = {"nlp": 0.50, "url": 0.30, "behaviour": 0.20}
W_VOICE = {"nlp": 0.65, "spoof": 0.20, "behaviour": 0.15}
SPOOF_COUNTS = 0.5                # spoof slot counts only at or above this (fuse_voice)

# Measured deviation from §8, documented rather than silent. The weighted sum
# cannot escalate a message the content model already cleared: behaviour's whole
# budget is 0.45, and 0.45 * 1.0 sits above the 0.35 SUSPICIOUS line only when
# the content model is ALSO unsure - which on the CEAS test slice happens for 4
# messages in 7,338, because a DistilBERT trained on the same week is saturated.
# So a content-clean message with a damning sender profile stays SAFE, which is
# exactly the case §8 gave behaviour a weight for. This gate fixes that: a
# behaviour score at or above ESCALATE lifts SAFE to SUSPICIOUS, never higher,
# and never to PHISHING on structure alone. Threshold picked on the 1-7 Aug
# training window; the 8 Aug slice was scored once, after.
ESCALATE = 0.60


def entropy(probs):
    """Normalised prediction entropy in [0,1]. §8's uncertainty signal - one
    line, and no MC-dropout pass. ponytail: swap in MC-dropout variance only if
    entropy stops separating the SUSPICIOUS band."""
    p = np.clip(np.asarray(probs, float), 1e-9, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def band(risk):
    return "PHISHING" if risk >= PHISHING else \
           "SUSPICIOUS" if risk >= SUSPICIOUS else "SAFE"


def fuse(nlp_probs, url=None, behaviour=None, reasons=(), malicious_idx=None,
         escalate_at=ESCALATE):
    """One event -> verdict dict.

    nlp_probs     full class distribution (ham/spam/phishing or ham/spam/smishing)
    url           max URL score in the message; None = no URL engine output at
                  all (no link, or no link text available), which is NOT the same
                  as 0.0 "scored and found benign"
    behaviour     behaviour_anomaly_score from §7; None = no sender profile
    malicious_idx which class counts as malicious; default = every class but ham
    escalate_at   behaviour score that lifts a content-clean message off SAFE
    """
    p = np.asarray(nlp_probs, float)
    mal = float(p[malicious_idx]) if malicious_idx is not None else float(1 - p[0])
    conf = float(p.max())
    h = entropy(p)

    # Renormalise over the channels that actually reported. §8's weights assume
    # all three always fire; when one is silent, spending its share on a zero
    # drags every risk down by up to 45% and the fixed §3 bands stop meaning
    # anything. Measured on CEAS, where no URL strings ship at all: passing the
    # has-URL flag in that slot costs the weighted sum 0.22 AUC.
    parts = {"nlp": mal, "url": url, "behaviour": behaviour}
    w0 = W_SURE if conf >= LOW_CONF else W_UNSURE
    live = {k: v for k, v in w0.items() if parts[k] is not None}
    w = {k: v / sum(live.values()) for k, v in live.items()}
    risk = sum(w[k] * parts[k] for k in w)
    verdict = band(risk)

    # §8: high variance routes to SUSPICIOUS - never to SAFE
    routed = verdict == "SAFE" and h >= HIGH_ENTROPY
    # ...and the measured addition (see main(), and the note on ESCALATE): the
    # weighted sum alone can never lift content-clean mail off SAFE, which is the
    # one job §8 wrote the behaviour weight to do.
    escalated = verdict == "SAFE" and (behaviour or 0) >= escalate_at
    if routed or escalated:
        verdict = "SUSPICIOUS"

    why = [r for r in reasons if r]
    if conf < LOW_CONF:
        why.append("content model unsure, weighted towards sender behaviour")
    if escalated:
        why.append("content looks ordinary but the sender does not")
    if routed:
        why.append("model uncertain - held as suspicious rather than cleared")
    # The two overrides are returned as data as well as prose, so a UI can say
    # WHICH rule set the band instead of parsing the reason string for it.
    return {"verdict": verdict, "risk": round(risk, 3), "confidence": round(conf, 3),
            "entropy": round(h, 3), "weights": w,
            "parts": {k: round(v, 3) for k, v in parts.items() if v is not None},
            "escalated": escalated, "routed": routed,
            "reason": "; ".join(why) or "no risk signal on any channel"}


# Measured in scratch/_trust_eval.py - the table is in trust_damp's docstring.
TRUST_DAMP = 0.1


def trust_damp(probs, trusted, damp=TRUST_DAMP, malicious_idx=2):
    """Damp the content model's malicious column for an ESTABLISHED sender whose
    mail stays on its own domain (behaviour.trusted). Same shape as gate_by_act:
    a damper, not a veto, and the row is renormalised.

    Why: v6's remaining false alarms on the held-out third mailbox are repeat
    senders it has already delivered many times (one company mailer: 28 of its 358) - the
    content model judges every mail cold, and a sender's track record is the
    one thing it cannot see. The behaviour score cannot fix this through the
    weighted sum: at W_SURE it carries 0.20, under a 0.50 content vote.

    Ceiling: a COMPROMISED known account sending on-domain links passes. Every
    off-domain link, a failed auth check, or a crowd report disarms it.

    Measured 2026-10-09, scratch/_trust_eval.py, behaviour.TRUST_N = 3:
      COST  phishing_pot (8,614, streamed by date): 23.2% comes from a repeat
            sender, but auth-ok + every-link-on-domain leaves 2.0% (173) trusted,
            and v6 still catches 99.4% of THAT slice at every damp 1.0 -> 0.1.
            Phishing lost overall: 0 at every damp. The trusted slice is
            saturated phishing; a borderline one from a hijacked account is the
            case this table cannot see.
      GAIN  third mailbox (791 held-out ham), v6 phishing FP by damp:
            1.0 4.6% | 0.5 4.2% | 0.2 3.4% | 0.1 2.5%. UPPER bound: that corpus
            has its URLs stripped, so the link condition could not run.
      REACH personal ham: 82-85% is from established senders, but only 37-39%
            of it keeps every link on the sender's domain (tracking hosts), so
            the rule arms on ~31-33% of ham. That is the price of never
            overriding an off-domain link, and it shrinks GAIN accordingly.
    No knee - the cost column is flat at zero - so 0.1 ships.
    """
    p = np.asarray(probs, float).copy()
    if trusted:
        p[malicious_idx] *= damp
        p = p / max(p.sum(), 1e-9)
    return p


def gate_by_act(probs, evidence, damp=0.5, malicious_idx=2):
    """Damp the malicious probability when the text asks for NOTHING.

    The third documented deviation from §8, and the cheapest. Content models key
    on topic: a message is scored as smishing because it is ABOUT banking, OTPs
    and blocked accounts. A bank's own fraud advisory is about all three, which
    is why 83% of real awareness SMS ("Never share your OTP", "do not share your
    UPI PIN") were being flagged. The act layer asks the different question -
    did this message ASK for something, threaten, or demand payment - and scores
    0.00 on every one of them.

    So: when no act fires anywhere in the text, the content model's malicious
    column is multiplied by `damp` and the row renormalised. It is a damper, not
    a veto: a saturated p=0.99 survives it (0.99 * 0.35 renormalises to ~0.97),
    which is exactly why smishing recall does not move. Only the borderline
    cases - the ones the content model was unsure about and got wrong - fall
    back under the §3 band.

    UNLIKE deescalate(), this needs no sender and no infrastructure, so it works
    on the typed-sender web path where the de-escalation gate is disarmed.

    Measured end to end through api.sms, 2026-08-22. Held-out smishing (187),
    conversational ham (300), the 40 real DLT-template transactional SMS, and a
    6-message probe of real bank fraud-advisory phrasing:

      damp   smish R   conv ham FP   transact FP   awareness FP
      1.00     85.0%          0.7%         10.0%          66.7%   (gate off)
      0.50     81.3%          0.7%          2.5%          50.0%   <- shipped
      0.35     79.7%          0.3%          2.5%          50.0%
      0.20     77.0%          0.3%          2.5%          16.7%
      0.10     70.6%          0.0%          0.0%           0.0%

    0.5 IS THE KNEE AND THAT IS WHY IT SHIPS: it takes the whole transactional
    win (10.0% -> 2.5%) for the smallest recall cost (-3.7 points). Everything
    below 0.5 buys the awareness column and pays for it steeply in recall.

    The awareness probe was NOT used to pick the value, deliberately. Those six
    messages were written by the same author as this code, and an authored set
    measures its author - the same rule that keeps transcripts.jsonl out of
    training. Tuning damp to 0.10 would score 0.0% on them and cost 14 points of
    real recall to do it. When a genuine corpus of bank advisory SMS exists,
    re-cut this table on that instead.
    """
    p = np.asarray(probs, float).copy()
    if evidence <= 0:
        p[malicious_idx] *= damp
        p = p / max(p.sum(), 1e-9)
    return p


def deescalate(f, evidence, url=None, trusted_sender=False, cap=0.30):
    """The move §8 is missing, and a DOCUMENTED DEVIATION like ESCALATE above.

    §8 can lift a content-clean message (the escalation gate) but nothing can
    clear a content FALSE POSITIVE: with nlp saturated at 1.0, a perfect sender
    and a clean link still floor the risk at 0.504, because nlp's weight is
    0.50. That floor is arithmetic and still real, but it BITES ON ALMOST
    NOTHING today: 2 of 40 real DLT-template transactional SMS are flagged, and
    the content model scores that population at mean p(smishing) = 0.05.

    Fires only when three independent channels agree the message is benign - a
    registered sender, no malicious link, and no scam ACT in the text. The last
    is the load-bearing one and is the ACL paper's background-without-evidence
    case: content that reads as phishing but contains no ask, no threat and no
    payment demand is background narrative, not an attack.

    Measured 2026-08-21 by `_transactional_eval.py` (40 transactional ham / 200
    credential-style smishing / 200 conversational ham):
        transactional false positives    5.0% -> 0.0%   (2 of 40  -> 0)
        conversational false positives   5.0% -> 5.0%   (untouched)
        genuine smishing recall         99.0% -> 92.0%  (198 of 200 -> 184)

    ON THIS HARNESS THAT IS A LOSING TRADE: 2 legitimate messages cleared, 14
    smishing cleared with them. All 14 are exactly the rows the harness assigns
    VK-ICICIB, whose entity really is in DLT_REGISTRY; all 31 rows given
    look-alike headers (VM-HDFCBNK, AX-KYCUPD, AD-SBIALT) are correctly refused,
    where trusting header SHAPE instead of registry membership refused none of
    them and cost 11 more points of recall (81.0%). So the gate does what it
    says; its worth is set entirely by how often a scammer can send from a
    REGISTERED entity's header - which the harness fixes BY ASSUMPTION (see the
    _readme in transactional_eval.json), not by measurement. Under a typed
    sender that probability is 1.0 and the gate is worthless; under a
    carrier-supplied header (§10 gateway mode) it is small and the gate is sound.

    An earlier version of this docstring claimed 57.5% -> 0.0% on 40/322/300 and
    that nlp saturates at 1.0 on transactional templates. NEITHER REPRODUCES on
    anything on disk: models/sms_mobilebert and models/sms_mobilebert_prev give
    identical 5.0% -> 0.0%, and both score those 40 at mean p = 0.05. Whatever
    produced 57.5% - an older checkpoint, an older eval set, or an older sampler
    (this script now draws 200/200, not 322/300) - no longer exists. Do not quote
    it, in the report or anywhere else.

    THE ONE DEPENDENCY IS HEADER INTEGRITY. The gate trusts a TRAI-registered
    DLT header the way Gmail trusts DMARC alignment, so recall degrades exactly
    as fast as scammers can obtain or spoof a REGISTERED entity's header - a
    plausible-looking one no longer suffices: 5% spoofing -> 97.5% recall, 25% ->
    88.2%, 100% -> 50.9%. Never enable it for a channel where sender identity
    cannot be verified - which today means EMAIL, since no corpus or live feed
    here carries SPF/DKIM results.
    """
    if f["verdict"] == "SAFE":
        return f
    if not (trusted_sender and evidence == 0.0 and (url is None or url < 0.5)):
        return f
    g = dict(f, risk=min(f["risk"], cap), deescalated=True)
    g["verdict"] = band(g["risk"])
    g["reason"] = (f["reason"] +
                   "; registered sender, clean link and no scam act - de-escalated")
    return g


def fuse_voice(semantic, spoof=None, behaviour=None, reasons=()):
    """§8's voice split: 0.65 semantic + 0.20 synthetic-voice + 0.15 behaviour.
    spoof defaults to None, not 0.0: "no spoof evidence" must not be scored as
    "confirmed human voice".

    The spoof slot is ONE-SIDED: it counts only when it says synthetic
    (>= SPOOF_COUNTS) and abstains otherwise. Most scam calls are real people,
    so a human-sounding voice is not evidence of safety - in the weighted sum it
    was acquitting them. scratch/_deepfake_intent_eval.py, 400 calls: real
    human scams caught 85 -> 77% with the ASV19+ASV5 head two-sided, accuracy
    88.0 -> 84.2%; one-sided 87.5%. Found on that data (post-hoc), so it is a
    design rule here, not a measured gain. The raw score stays in `parts` for
    display, at weight 0."""
    counted = spoof if spoof is not None and spoof >= SPOOF_COUNTS else None
    parts = {"nlp": semantic, "spoof": counted, "behaviour": behaviour}
    live = {k: v for k, v in W_VOICE.items() if parts[k] is not None}
    w = {k: v / sum(live.values()) for k, v in live.items()}
    risk = sum(w[k] * parts[k] for k in w)
    if spoof is not None and counted is None:
        parts["spoof"], w["spoof"] = spoof, 0.0
    verdict = band(risk)
    # Same gate as fuse(): at 0.15 the caller can never move the band alone, so
    # a caller check at ESCALATE lifts SAFE to SUSPICIOUS, never higher.
    escalated = verdict == "SAFE" and (behaviour or 0) >= ESCALATE
    why = [r for r in reasons if r]
    if escalated:
        verdict = "SUSPICIOUS"
        why.append("what was said sounds ordinary but the caller's number does not")
    return {"verdict": verdict, "risk": round(risk, 3), "weights": w,
            "parts": {k: v for k, v in parts.items() if v is not None},
            "escalated": escalated, "reason": "; ".join(why) or "no risk signal"}


class Stacker:
    """§8's comparison: learn the weights instead of setting them.

    Needs labelled events, which the weighted sum does not - that is the honest
    cost, and why fuse() is what ships.
    """

    def __init__(self):
        self.lr = None

    @staticmethod
    def _X(nlp_mal, url, behaviour, conf):
        cols = [nlp_mal, behaviour, conf, nlp_mal * behaviour]  # the §8 interaction
        if url is not None:                  # same rule as fuse(): a silent channel
            cols.insert(1, url)              # contributes nothing rather than a zero
        return np.column_stack(cols)

    def fit(self, nlp_mal, url, behaviour, conf, y):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            self._X(nlp_mal, url, behaviour, conf), y)
        return self

    def predict(self, nlp_mal, url, behaviour, conf):
        return self.lr.predict_proba(self._X(nlp_mal, url, behaviour, conf))[:, 1]


# --- Phase 2 experiment ------------------------------------------------------

def recall_at_fpr(y, s, fpr=0.05):
    """Recall at a fixed false-positive rate. Comparing raw recall across models
    is meaningless when they sit at different operating points."""
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


def main():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import behaviour as bh
    import email_model

    # CEAS_08 is the only corpus with ham AND spam AND real senders, receivers and
    # dates - i.e. the only place a rolling sender profile can be built at all.
    # Its label means SPAM, not phishing (measured in Phase 1), so this experiment
    # is about the spam ceiling, which is exactly what behaviour was built for.
    e = data.load_email()
    c = e[(e.corpus == "CEAS_08") & e.date.notna() & e.receiver.notna()]
    c = c[c.date.dt.strftime("%Y-%m") == "2008-08"].sort_values("date")

    feats = bh.stream(c)                  # profiles built strictly from the past
    day = c.date.dt.day
    tr, te = (day <= 7).to_numpy(), (day == 8).to_numpy()   # content model's split
    y, y_tr = c.label.to_numpy()[te], c.label.to_numpy()[tr]
    print(f"stream {len(c):,} mails, {c.sender.nunique():,} senders  "
          f"train {tr.sum():,} / test {te.sum():,} ({y.mean():.0%} spam)")

    # §7 profile of NORMAL: the forest fits the benign half of the training week.
    model = bh.Behaviour().fit(feats[tr], normal=y_tr == 0)
    iso = model.score(feats[te])
    rule = bh.rule_score(feats)
    b, rule_te = rule[te], rule[te]

    p, _ = email_model.score(c.text[te].tolist())
    nlp_mal, conf = p[:, 1] + p[:, 2], p.max(1)
    # CEAS ships a has-URL flag, not the link text, so the URL engine cannot run
    # on this corpus at all. Kept only for the artifact check printed below.
    url = c.urls.to_numpy(float)[te]

    fused = [fuse(pi, None, bi) for pi, bi in zip(p, b)]
    weighted = np.array([f["risk"] for f in fused])
    verdicts = np.array([f["verdict"] for f in fused])

    # The stacker needs labels the content model never trained on, so it is fitted
    # on the first half of day 8 and scored on the second; both fusers are then
    # compared on that same untouched slice, on the same inputs.
    #
    # The has-URL flag is withheld from BOTH. Given it, the stacker reaches AUC
    # 0.986 and it is a lie: the flag scores AUC 0.395 on its own, because 63% of
    # CEAS ham carries a link against 43% of its spam (2008 spam was plain-text
    # pill adverts, the ham was mailing-list digests full of links). The stacker
    # learns "has a link -> probably ham", which inverts the moment it meets real
    # phishing. Third instance in this project of a model learning the collection
    # method instead of the task - see PhiUSIIL's HTTPS-with-no-path artifact.
    h = len(y) // 2
    st = Stacker().fit(nlp_mal[:h], None, b[:h], conf[:h], y[:h])
    stacked = st.predict(nlp_mal[h:], None, b[h:], conf[h:])

    print(f"\n{'':38}{'AUC':>7}{'rec@5%FP':>10}{'rec@1%FP':>10}")
    for name, yy, ss in [("content only (Phase 1)", y, nlp_mal),
                         ("behaviour only, hand rule", y, rule_te),
                         ("behaviour only, isolation forest", y, iso),
                         ("fusion, weighted sum (spec 8)", y, weighted),
                         ("content only, same 2nd half", y[h:], nlp_mal[h:]),
                         ("fusion, LR stacker (spec 8, 2nd half)", y[h:], stacked)]:
        print(f"{name:38}{roc_auc_score(yy, ss):>7.3f}{recall_at_fpr(yy, ss):>10.3f}"
              f"{recall_at_fpr(yy, ss, .01):>10.3f}")

    # --- the Phase 2 question: does behaviour break the 0.609 spam ceiling? ---
    # ESCALATE was fixed from the 1-7 Aug window (see the note on the constant);
    # this slice is scored with it, not tuned on it.
    cleared = nlp_mal < 0.5               # what the content model waves through
    esc = rule_te >= ESCALATE
    rec_spam = ((nlp_mal >= .5) | esc)[y == 1].mean()
    print(f"\ncontent cleared {cleared.sum():,} mails "
          f"({(y[cleared] == 0).sum():,} ham, {(y[cleared] == 1).sum():,} spam)")
    print(f"behaviour AUC on just that slice           {roc_auc_score(y[cleared], rule_te[cleared]):.3f}")
    print(f"of the {(y[cleared] == 1).sum():,} missed spam, escalated to SUSPICIOUS "
          f"{(esc & cleared & (y == 1)).sum():,}")
    print(f"ham escalated with them                    "
          f"{(esc & cleared & (y == 0)).sum():,} of {(y[cleared] == 0).sum():,} "
          f"({(esc & cleared & (y == 0)).sum() / (y[cleared] == 0).sum():.1%})")
    print(f"\nSPAM RECALL  content only {(nlp_mal >= .5)[y == 1].mean():.3f}"
          f"  ->  with behaviour {rec_spam:.3f}")
    print(f"verdict mix: {pd.Series(verdicts).value_counts().to_dict()}")
    print(f"(withheld has-URL flag scores AUC {roc_auc_score(y, url):.3f} alone - "
          f"{url[y == 0].mean():.0%} of ham carries a link vs {url[y == 1].mean():.0%} "
          f"of spam, so it is a corpus artifact, not a signal)")

    tags = [t for _, f in feats[te][esc & cleared & (y == 1)].iterrows()
            for t in bh.explain(f)]
    print("why they were escalated:", pd.Series(tags).value_counts().head(3).to_dict())


if __name__ == "__main__":
    # self-check first: the arithmetic must hold without loading 66M parameters
    sure_phish = fuse([0.02, 0.03, 0.95], url=0.9, behaviour=0.8)
    assert sure_phish["verdict"] == "PHISHING" and sure_phish["weights"] == W_SURE

    # §8's actual claim: content unsure + behaviour damning must still escalate
    unsure = fuse([0.40, 0.35, 0.25], url=0.2, behaviour=0.95)
    assert unsure["weights"] == W_UNSURE
    assert unsure["verdict"] != "SAFE", unsure
    assert unsure["risk"] > fuse([0.40, 0.35, 0.25], url=0.2, behaviour=0.0)["risk"]

    # a flat distribution with nothing else must not be cleared (§8 routing)
    flat = fuse([0.34, 0.33, 0.33], url=0.0, behaviour=0.0)
    assert flat["verdict"] == "SUSPICIOUS" and "uncertain" in flat["reason"]
    assert flat["routed"] and not flat["escalated"]
    lifted = fuse([0.99, 0.005, 0.005], behaviour=0.9)
    assert lifted["escalated"] and not lifted["routed"], lifted
    assert fuse([0.99, 0.005, 0.005])["verdict"] == "SAFE"

    assert round(entropy([1.0, 0, 0]), 3) == 0.0 and round(entropy([1 / 3] * 3), 3) == 1.0
    assert band(0.35) == "SUSPICIOUS" and band(0.70) == "PHISHING" and band(0.34) == "SAFE"
    assert fuse_voice(0.9, 0.8, 0.5)["verdict"] == "PHISHING"
    # a silent channel must not dilute the ones that spoke: no spoof head yet
    # cannot mean "20% evidence of a human voice"
    assert fuse_voice(0.9)["risk"] == 0.9
    assert fuse_voice(0.9, 0.9, 0.9)["risk"] > fuse_voice(0.9, 0.0, 0.0)["risk"]
    # a human-sounding voice never lowers the risk; it is shown, at weight 0
    v = fuse_voice(0.4, spoof=0.1)
    assert v["risk"] == fuse_voice(0.4)["risk"] == 0.4 and v["verdict"] == "SUSPICIOUS"
    assert v["parts"]["spoof"] == 0.1 and v["weights"]["spoof"] == 0.0
    assert fuse_voice(0.4, spoof=0.9)["risk"] > 0.4
    assert fuse([0.1, 0.1, 0.8])["risk"] == fuse([0.1, 0.1, 0.8], None, None)["risk"]
    # ...and behaviour alone must never reach PHISHING
    assert fuse([0.99, 0.005, 0.005], behaviour=1.0)["verdict"] == "SUSPICIOUS"
    print("self-check ok\n")

    main()
