"""Voice channel: audio -> transcript -> Phase 1 text model + keyword layer -> risk curve.

Spec §6: 3s sliding windows with 1s overlap, DistilBERT intent head on the
transcript, a keyword layer for high-risk entities, an EMA risk curve, and an
alert only after two consecutive windows above threshold.

Two deliberate simplifications, both marked below: no voice-specific head is
trained (the Phase 1 SMS model already knows these lures), and speaker
separation comes from the stereo channels rather than a diarization model.
"""
import re
import sys
import wave
from pathlib import Path

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data  # noqa: F401
import numpy as np

import sms_model

# §6 keyword/NER layer. Asking for a secret scores higher than merely naming
# one: a real bank says "never share your OTP", and the eval set contains that
# exact call.
# The lookbehinds are load-bearing, not tidying. "Do NOT share your OTP" is the
# opposite of an ask and is the most reliable marker of a LEGITIMATE sender -
# every real bank OTP carries it, no scammer does. Without them it scored as
# evidence of the very act it warns against. §5's legit half already held the
# real bank call saying "we will never ask for your OTP, PIN or CVV".
# The verb list gained update/verify/activate/complete and the object list gained
# kyc/account/card on 2026-08-22. The original was written for spoken calls, where
# the ask is "read me the OTP"; the Indian SMS ask is "update your KYC", and the
# act layer fired on 3.7% of held-out smishing before this.
ASK = re.compile(r"\b(share|tell|give|send|read|confirm|provide|type|enter|dictate|"
                 r"update|verify|activate|complete|revalidate)"
                 r"\b[^.?!]{0,40}\b(otp|cvv|pin|code|password|aadha?ar|pan|kyc|"
                 r"account|debit card|credit card|net ?banking)\b", re.I)

# Fixed-width lookbehinds used to carry the negation guard, which meant they only
# saw the ONE word before the verb. "Do not click on links asking you to update
# your KYC" slipped straight through: the word before "update" is "to". A window
# does what a lookbehind cannot, because Python lookbehinds must be fixed width.
#
# Warning words only - never / not / avoid / beware. NOT fraud/fake/scam, which
# look like warning cues and are not: "fraud detected on your account, call now"
# is one of the commonest scam openers there is.
NEG = re.compile(r"\b(never|not|n't|avoid|beware|refrain|no need|instead of)\b", re.I)
# The negation is looked for back to the start of the SENTENCE the act sits in,
# not over a fixed character window. A window was 45 characters until 2026-09-22
# and it missed the real thing: in Airtel's own advisory the ASK match starts at
# character 71 and the "Never" governing it is at character 0, so the guard never
# saw it and the act layer scored 0.65 on a fraud WARNING - which is how that
# message became the single PHISHING verdict in a 943-message real inbox.
#
# Measured over the 63 advisory messages in that inbox, 400 smishing and the 40
# transactional templates:
#     window 45   advisory fires 22.2%   smishing acts kept 15.0%
#     window 80+  advisory fires 12.7%   smishing acts kept 15.0%
#     sentence    advisory fires 12.7%   smishing acts kept 15.5%
# Sentence scoping matches the widest window on advisories and RECOVERS two
# smishing acts that a wide window wrongly suppressed, because a negation two
# sentences earlier does not govern this clause. It is also the honest rule: a
# warning governs its own sentence, and a scam's ask is its own sentence.
_SENTENCE_END = re.compile(r"[.!?]")


def _fires(rx, text, neg=True):
    """True if `rx` matches somewhere it is not being warned against.

    A bank's own fraud advisory contains the exact words of the attack it warns
    about, which is why the content model flags 83% of real awareness SMS as
    smishing. Telling those apart is the act layer's whole job, so on TEXT it has
    to read the negation rather than the vocabulary.

    ON FOR TEXT, OFF FOR SPEECH, and that is measured, not stylistic:

                            neg on   neg off
      bank awareness SMS      0.0%     50.0%    written warnings exist
      transactional ham       2.5%      7.5%
      Indian smishing        31.8%     34.2%    costs 2.4 points
      voice alert F1 (test)   0.38      0.56    speech is hurt badly

    Nobody telephones you to warn you not to share your OTP, so on a call the
    construction that survives is the scammer's own - "if you do NOT share the
    code your account will be blocked" - and suppressing it throws away a real
    ask. In writing the same construction is usually the bank warning you.
    """
    if not neg:
        return bool(rx.search(text))
    return any(not NEG.search(text[_sentence_start(text, m.start()):m.start()])
               for m in rx.finditer(text))


def _sentence_start(text, at):
    """Where the sentence containing `at` begins - 0 if it is the first."""
    return max((m.end() for m in _SENTENCE_END.finditer(text[:at])), default=0)
THREAT = re.compile(r"\b(account[^.?!]{0,20}(blocked|suspended|freez|clos)|"
                    r"arrest|police|cbi|legal\s+action|case\s+(filed|registered)|"
                    r"disconnect|expire[sd]?\s+today|within\s+\d+\s+(minute|hour))", re.I)
REMOTE = re.compile(r"\b(anydesk|teamviewer|remote\s+access|screen\s+shar|"
                    r"install[^.?!]{0,20}app|gift\s+card|google\s+play\s+code)\b", re.I)
ENTITY = re.compile(r"\b(otp|cvv|aadha?ar|pan\s*card|kyc|verification\s+code|"
                    r"debit\s+card|credit\s+card|net\s*banking)\b", re.I)

# Added after the eval set showed the original four rules covered only 14 of 25
# scam families. Both were written from DEV items only (army-UPI, relative-in-
# trouble, fake charity, digital-arrest follow-up) and fire on 8 of 13 dev scams
# and 0 of 13 dev legitimate calls. A third family the gaps suggested - the
# unsolicited prize - was deliberately NOT added: dev holds a real gym renewal
# and a real credit-card offer that such a rule would have flagged, so the split
# earned its keep by refusing a rule that looked obvious.
#
# "transfer" was REMOVED from the verbs 2026-10-10. It was 94 of the 96 false
# alarms on HarperValley half A - real customers saying "transfer money between
# my checking and savings" - and every other HVB false alarm was 2 ASK hits.
# Rules-only, flagged >= 0.35 (`scratch/_hvb_rules.py`; fixed on half A, read on B):
#                     HVB A   HVB B   scambait   27 scam   27 legit
#     with transfer   13.3%   10.8%    75.1%      50.0%     26.7%
#     without          0.6%    0.3%    74.7%      50.0%     26.7%
# An own-account guard ("checking|savings|between") was tried and did nothing:
# the match is "transfer amount", the words that make it innocent sit outside it.
# THE COST: authored test AUC 0.757 -> 0.719 - s24's "transfer to the school's
# account" no longer matches. Re-adding "transfer ... to (this|the|a...) account"
# restores it at no HVB cost but catches 0 extra scambait calls, and "the" was
# only chosen after seeing s24 - a test item. Refused as fitting the test set.
MONEY = re.compile(r"\b(scan[^.?!]{0,30}(qr|code)|(send|pay|deposit|arrange)"
                   r"[^.?!]{0,40}\b(rupees|amount|money|account|upi|qr|gpay|google pay|"
                   r"paytm|thousand|lakh)|processing (fee|charge)|refundable)\b", re.I)
SECRECY = re.compile(r"\b(do not (tell|discuss|disclose|inform|speak to)|don't tell|"
                     r"do not disconnect|keep (it |this )?(confidential|secret)|"
                     r"without (telling|informing)|no ?one should know)\b", re.I)

# ponytail: hand-set weights, summed and capped. These are THE calibration knob.
# The first four were fitted to two hand-written scripts; the last two and every
# threshold below were fitted on the 26-transcript DEV half and nothing else.
RULES = [(0.65, ASK), (0.45, THREAT), (0.65, REMOTE), (0.25, ENTITY),
         (0.45, MONEY), (0.45, SECRECY)]

WINDOW, OVERLAP = 3.0, 1.0   # §6 emission cadence: a score every 2 seconds
ALERT_THRESHOLD = 0.70       # the PHISHING band from §3 - unchanged
EMA_ALPHA = 0.6              # knob: higher = reacts faster, more single-window noise

# Each emission is scored on the trailing CONTEXT seconds, not on the 3s window
# alone. Measured: 3s of phone speech is 7.7 words, which is shorter than the
# phrases being detected ("please share the code you received"), so per-window
# scores collapsed and the alert fired on 2 of 25 scam calls. Scoring the recent
# conversation instead lifts dev recall from 0.08 to 0.69 without touching the
# emission rate, the threshold or the §3 bands. 10s and 20s work nearly as well;
# this is a plateau, not a tuned spike.
CONTEXT = 15.0

# Score scam acts per window (streaming) instead of once on the finished
# transcript. MEASURED AND REJECTED 2026-08-24, kept because the negative result
# is the useful part - see _act_stream for the numbers. Short calls look like a
# win (FTC catch 26.0% -> 41.4% at an unchanged 12.0% false-alarm rate on 1,446
# real bank calls) but on the 27 recorded two-party calls, median 913 words, it
# alerts on 60% of the LEGITIMATE half against 7% call-level, while peak-risk AUC
# does not move (0.703 either way). It is not detecting more, it is a maximum
# over ~150 fragments, so the absolute risk grows with call duration and walks
# through §3's fixed bands. Any future streaming build must score the PREFIX,
# not a 15 s fragment, so that end-of-call equals the call-level number by
# construction.
STREAM_ACTS = False
LATCH = 2                    # firing windows before a streamed act counts


def _whisper(model_size, _cache={}):
    """ponytail: cache the model. transcribe() used to build a fresh
    WhisperModel per call, which is invisible on one demo clip and means N model
    loads over a corpus of N."""
    if model_size not in _cache:
        from faster_whisper import WhisperModel  # heavy import, only when used
        # ponytail: ask torch, not another channel's model. sms_model used to be
        # a transformer and carried a DEV constant; it is now a 2.6 MB sklearn
        # pipeline with no torch in it at all.
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _cache[model_size] = WhisperModel(
            model_size, device=dev,
            compute_type="float16" if dev == "cuda" else "int8")
    return _cache[model_size]


def transcribe(path, model_size="small"):
    """-> [(start, end, word)]. Word timestamps let us cut clean text windows
    instead of chopping the audio, which would slice words in half.

    `path` may also be a float32 numpy array already at 16 kHz - two-party
    corpora arrive as one file per speaker and have to be mixed in memory.
    """
    audio = str(path) if isinstance(path, (str, Path)) else path
    segments, _ = _whisper(model_size).transcribe(
        audio, word_timestamps=True, vad_filter=True)
    return [(w.start, w.end, w.word.strip())
            for seg in segments for w in seg.words if w.word.strip()]


def windows(words, ctx=CONTEXT, hop=WINDOW - OVERLAP):
    """Emit a scoreable text every `hop` seconds, each covering the trailing
    `ctx` seconds of transcript.

    ponytail: O(n*windows) rescan - fine for a call (a few thousand words);
    bucket by time if you ever feed it an hour of audio.
    """
    if not words:
        return []
    out, t, end = [], hop, words[-1][1]
    while t < end + hop:
        text = " ".join(w for s, e, w in words if e > t - ctx and s < t)
        out.append((round(min(t, end), 2), text))
        t += hop
    return out


def rule_score(text):
    """Keyword/NER layer -> [0,1]. Weights sum, then cap."""
    # neg=False: this is the SPEECH path. See _fires - the negation window costs
    # this channel 18 points of alert F1 on the test half and buys it nothing.
    return min(1.0, sum(w for w, rx in RULES if _fires(rx, text, neg=False)))


# ENTITY matches the mere MENTION of otp/kyc/aadhaar/card - a topic, not an act.
# Measured on 40 real DLT-template SMS vs 322 credential-style smishing: ENTITY
# fires on 20% of the LEGITIMATE half and 3.5% of the malicious one, i.e. it is
# inverted, because the message most likely to contain the word "OTP" is a real
# OTP. Same shape as `urgency` in the 7B distillation. It keeps its 0.25 in
# RULES for scoring; it must never count as EVIDENCE that an act occurred.
# TWO OBVIOUS SMS ACTS WERE REFUSED, both on measurement over dev rows only:
#   "call this number"  Dataset_10191 smishing 58.7% / its ham 0.0%
#                       BUT Indian smishing 0.9% / Indian ham 4.3% - inverted
#   "contains a link"   Indian smishing 7.0% / Indian ham 17.1% - inverted
# Both describe the UK premium-rate era the older corpus was collected in, not an
# Indian scam. Same refusal as the unsolicited-prize rule on the voice side, for
# the same reason: a rule that fires on the legitimate half is worse than no rule.
EVIDENCE_ACTS = ("ASK", "THREAT", "REMOTE", "MONEY", "SECRECY")
_ACT_NAMES = ("ASK", "THREAT", "REMOTE", "ENTITY", "MONEY", "SECRECY")


def evidence_risk(text):
    """Act evidence only, ENTITY excluded -> [0,1].

    Answers "did this message ASK for something?" rather than "is it about
    banking?". Fires on 2.5% of legitimate transactional SMS vs 37.3% of
    credential-style smishing; with ENTITY included those numbers invert.
    """
    return min(1.0, sum(w for n, (w, rx) in zip(_ACT_NAMES, RULES)
                        if n in EVIDENCE_ACTS and _fires(rx, text)))


def window_risk(texts):
    """Per-window risk in [0,1]. Rules only - the SMS model was REMOVED from
    this channel on 2026-08-20 because on real recorded calls it is worse than
    nothing.

    Measured on 27 real two-party call recordings (fixed dev/test halves) and on
    the 50 authored transcripts:

                          real calls          authored
                        dev    test         dev    test
      rules only       0.702  0.708        0.911  0.663
      sms only         0.262  0.438        0.834  0.826
      noisy-OR (old)   0.405  0.542        0.876  0.826

    The two sets disagree completely, and the length column explains why: the
    authored transcripts are a median of 57 words, real calls a median of 913.
    The authored set is SMS-SHAPED TEXT, so an SMS model scores well on it; it
    never tested transfer to conversation. On real speech `sms_model` separates
    nothing (scam mean 0.597, legit 0.570) and noisy-OR drags a working 0.70
    signal down to chance.

    Why it fails is not a bug in the model - it is what it was trained on. In
    SMS an unsolicited "verify your account" IS smishing, because banks do not
    hold SMS conversations, so the model learned that transactional banking
    vocabulary is itself the alarm. On a call that vocabulary is what a
    LEGITIMATE agent says: a real bank customer-service call scores 0.994 with
    reason "scam-like phrasing, no rule fired" - the neural layer alone.

    ponytail: no combiner rescues both sets (gating sms behind rules>0 reads
    0.667 on real-call test and 0.681 on authored test - worse than rules alone
    on each). Two registers, two models: `sms_model` keeps the SMS channel and
    `thread.py`, and stays out of speech. The replacement for the missing
    generalisation is a primitive/evidence detector, not another whole-
    transcript classifier - see speech_model.py's docstring for the attempt that
    failed and why.
    """
    return np.array([rule_score(t) for t in texts])


def risk_curve(scores, alpha=EMA_ALPHA, decay=EMA_ALPHA / 3):
    """Exponential moving average - scam calls escalate, the climb is the signal.

    Asymmetric on purpose: rises at alpha, falls at decay. A symmetric EMA lets
    one window of small talk between two OTP asks drop the curve back under
    threshold, and a call that has already asked for your OTP is not safe again
    three seconds later. Risk ratchets; it does not forget.
    """
    out, prev = [], float(scores[0]) if len(scores) else 0.0
    for s in scores:
        a = alpha if s > prev else decay
        prev = a * s + (1 - a) * prev
        out.append(prev)
    return np.array(out)


def alert_at(curve, times, threshold=ALERT_THRESHOLD):
    """First time the curve holds above threshold for two consecutive windows.
    One window alone is a single-word false alarm; §6 requires two."""
    for i in range(1, len(curve)):
        if curve[i - 1] >= threshold and curve[i] >= threshold:
            return times[i]
    return None


def split_stereo(path):
    """-> (left, right) int16 arrays, or None if the file is mono.

    ponytail: this IS the speaker separation. A dual-channel recording (stereo
    mic setup, or what Exotel/Twilio hand back) puts each party on their own
    channel, so a numpy slice beats a diarization model that can still be wrong.
    Mono file and you genuinely need speakers? That is when pyannote earns its keep.
    """
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 2:
            return None
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return frames[0::2], frames[1::2]


def as_words(text, wps=2.5):
    """Text -> fake word timings, so the risk pipeline runs on a transcript with
    no audio. 2.5 words/second is ordinary Indian-English phone speech; the eval
    set is text, and ASR error is measured separately when recordings exist."""
    return [(i / wps, (i + 1) / wps, w) for i, w in enumerate(text.split())]


def analyse(path, caller=None, risky_sms=None):
    """Full pipeline on one recording -> the §8 voice verdict.

    0.65 semantic + 0.20 synthetic-voice + 0.15 behaviour. The spoof head is
    asked only if it has been trained; an untrained head passes None, which
    fusion treats as "no evidence" rather than "confirmed human" - the two are
    not the same and scoring them the same is how a missing model quietly
    becomes an acquittal.

    caller: the caller's number as the user typed it (or "withheld"), and
    risky_sms: store.risky_sms for it - behaviour.voice_features. A bare audio
    file has neither, and the behaviour slot abstains.
    """
    import behaviour
    import fusion
    import spoof_model

    words = transcribe(path)
    sem = verdict(windows(words), text=" ".join(w[2] for w in words))
    spoof = None
    if spoof_model.OUT.with_suffix(".pkl").exists():
        spoof = float(spoof_model.score([path])[0])

    text = " ".join(w[2] for w in words)
    beh, tags = behaviour.voice_score(behaviour.voice_features(caller, text, risky_sms))
    f = fusion.fuse_voice(sem["peak_risk"], spoof=spoof, behaviour=beh,
                          reasons=[sem["reason"]] +
                          ([f"voice is likely synthetic ({spoof:.2f})"]
                           if spoof is not None and spoof >= 0.5 else []) + tags)
    # transcript rides along: the web UI shows what was heard next to the score,
    # and rebuilding it from `windows` is wrong - they overlap by design (§6).
    return {**sem, **f, "semantic_risk": sem["peak_risk"], "transcript": text}


def score_transcript(text, wps=2.5):
    """Same verdict, from text instead of audio (§5 eval set, gateway replay)."""
    return verdict(windows(as_words(text, wps)), text=text)


def _act_risk(text):
    """-> (risk, acts fired) from the distilled act detector, (0.0, []) if it has
    not been trained. See act_model.py - it is the only layer here that survived
    contact with real recorded calls."""
    # No try/except: absence is the OUT check, anything else is a real failure
    # and used to be swallowed into "no acts", i.e. a silent SAFE.
    import act_model
    if not act_model.OUT.exists():
        return 0.0, []
    w = text.split()
    chunks = [" ".join(w[i:i + 250]) for i in range(0, len(w), 250)] or [text]
    acts = act_model.acts_of(act_model.load()[1])
    return act_model.fired_risk(acts, act_model.score(chunks).max(0))


# Where the call-level acts come from. "distilled" = act_model, the DistilBERT
# copy of the judge (milliseconds, 256 tokens at a time). "llm" = the judge
# itself, llm_primitives.call_risk (~30 s a call, ~5 GB of GPU; set
# LLM_PRIMITIVES_MODEL / _4BIT). The small copy existed for a 2-second live
# alert, and Android cannot record calls, so voice is upload-only on the website
# and the wait is affordable. Under measurement 2026-10-10: `_voice_llm_eval.py`.
ACTS_FROM = "distilled"
LLM_HYBRID = True            # llm_primitives.call_risk's hybrid prompt - see its docstring


def _llm_risk(text):
    import llm_primitives as lp
    return lp.call_risk(text, criteria=True, hybrid=LLM_HYBRID)[0]


def _act_stream(texts):
    """-> per-window act risk, ratcheted (an act does not un-happen), or None if
    the detector is absent.

    The streaming half of CHI 2025's turn-by-turn design: judging only the whole
    transcript means the verdict cannot exist before the caller hangs up, which
    is useless to the person on the call. Scoring each window lets the alert fire
    when the act is spoken. It is only viable on a detector trained with the
    ACL local-evidence rows - a window is ~40 words and the whole-transcript
    model never saw a fragment that short.

    OFF by default - this is a measured negative result, not a spare feature.
    Audio path, 26 of the 27 recorded calls, peak-risk AUC / scam alerts / legit
    alerts:

        v1 acts, call-level (was shipped)   0.703   27%    7%
        ACL acts, call-level (now shipped)  0.688   27%    0%
        v1 acts, streaming                  0.679   45%   13%
        ACL acts, streaming                 0.703   91%   60%

    Streaming does not rank better - the AUC is identical to call-level. What it
    does is inflate the absolute risk on long calls, because a maximum over ~150
    overlapping fragments almost always finds one. LATCH (requiring two firing
    windows) was added to fix that and did not: with 15 s of context on a 2 s hop
    a single phrase already lands in about seven consecutive windows, so the
    latch is satisfied by one utterance. The FTC catch it appeared to buy
    (26.0% -> 41.4%) is the same effect read from the other side - short calls,
    more fragments scored, more fires - not better detection.
    """
    import act_model
    if not act_model.OUT.exists():
        return None
    acts = act_model.acts_of(act_model.load()[1])
    r = np.array([act_model.fired_risk(acts, row)[0]
                  for row in act_model.score(list(texts))])
    return np.maximum.accumulate(np.where(np.cumsum(r > 0) >= LATCH, r, 0.0))


def verdict(win, text=None):
    """Windows -> verdict dict. Shared by the audio and transcript paths so the
    thing the eval set measures is the thing that runs on a real call."""
    if not win:
        return {"verdict": "SAFE", "reason": "no speech detected", "curve": []}
    times, texts = zip(*win)
    curve = risk_curve(window_risk(list(texts)))
    # Scam ACTS. Call-level by default - an act is evidence about the call and
    # does not decay - which took the 27 real calls 0.703 -> 0.772. STREAM_ACTS
    # scores them per window instead so the alert can fire mid-call; measured by
    # `_hvb_eval.py`. Silently skipped if the model is absent.
    if STREAM_ACTS:
        a = _act_stream(texts)
        if a is not None:
            curve = np.clip(curve + a, 0.0, 1.0)
    fired = []
    if not STREAM_ACTS:
        acts, fired = ((_llm_risk(text), []) if ACTS_FROM == "llm" else _act_risk(text)) if text else (0.0, [])
        if acts:
            curve = np.clip(curve + acts, 0.0, 1.0)
    hit = alert_at(curve, times)
    peak = float(curve.max())
    mv = moves(times, texts, fired, text)
    return {
        "verdict": "PHISHING" if hit is not None else
                   "SUSPICIOUS" if peak >= 0.35 else "SAFE",
        "alert_at_s": hit,
        "peak_risk": round(peak, 3),
        "reason": _reason(mv, peak),
        "moves": mv,   # what the caller did, when, in their words - the web panel lists these
        # plain floats, not np.float64 - this dict goes straight into the event schema
        "curve": [round(float(c), 3) for c in curve],
        "windows": win,
    }


def evaluate(weights=None):
    """Score the 50 authored transcripts (§5) -> per-item table + the numbers.

    The channel's first real measurement. Everything quoted for voice before
    this rested on two hand-written scripts, which is not evidence.
    """
    import data
    global RULES
    if weights is not None:
        RULES = [(w, rx) for w, (_, rx) in zip(weights, RULES)]

    d = data.load_transcripts()
    out = [score_transcript(t) for t in d.text]
    d = d.assign(peak=[r["peak_risk"] for r in out],
                 alert=[r["alert_at_s"] is not None for r in out],
                 said=[r["verdict"] for r in out],
                 why=[r["reason"] for r in out])

    tp = int((d.alert & (d.label == 1)).sum())
    fp = int((d.alert & (d.label == 0)).sum())
    fn = int((~d.alert & (d.label == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    d.attrs["scores"] = {"precision": prec, "recall": rec,
                         "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                         "tp": tp, "fp": fp, "fn": fn}
    return d


# Each scam move in plain words, for the reason string and the web panel alike.
# Each scam move in plain words, for the reason string and the web panel alike.
# Said of the CALL, not of "the caller": a mono recording holds both voices and
# nothing here can tell who said a line (scam-baiters threaten scammers too).
SAY = {"credential": "a request for a one-time code, PIN, password or card number",
       "payment":    "a demand for money or a payment",
       "remote":     "a request to install an app or share your screen",
       "giftcard":   "talk of gift cards, a common way scammers take payment",
       "threat":     "a threat of arrest, a fine or a blocked account",
       "secrecy":    "a request to keep the call secret or stay on the line",
       "urgency":    "pressure to act right away",
       "prize":      "a claim that you've won a prize, refund or compensation"}
_RULE_ACT = {"ASK": "credential", "THREAT": "threat", "REMOTE": "remote",
             "MONEY": "payment", "SECRECY": "secrecy"}   # ENTITY is a topic, not a move


def _rule_act(name, match):
    # REMOTE also carries gift / Play cards, which are a payment, not an app
    return "giftcard" if name == "REMOTE" and re.match(r"gift|google", match, re.I) else _RULE_ACT.get(name)


def _clip(text, n=30):
    w = text.split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def _sentence(text, phrase):
    """The whole sentence holding the first `phrase` in `text`."""
    m = re.search(re.escape(phrase), text)
    e = _SENTENCE_END.search(text, m.end() - 1)
    return _clip(text[_sentence_start(text, m.start()):e.end() if e else None].strip())


def moves(times, texts, fired=(), text=None):
    """-> [{act, say, t, quote}]: the scam moves the verdict counted, when they
    were heard and the sentence they were heard in, earliest first.

    A rule move is timed by the first window it matched in and quoted with that
    match's whole sentence from the full transcript (`text`; a window can end
    mid-sentence). A move only the act model found (`fired`) is quoted with the
    transcript sentence that model scores highest for it, timed by the first
    window holding that sentence's end."""
    found = {}
    for name, (_, rx) in zip(_ACT_NAMES, RULES):
        for t, x in zip(times, texts) if name in _RULE_ACT else ():
            for m in rx.finditer(x):   # = _fires(neg=False), the speech path
                q = m.group(0)
                found.setdefault(_rule_act(name, q), (t, _sentence(text if text and q in text else x, q)))
    rest = [a for a in fired if a not in found]
    if rest and text:   # fired acts only exist when a transcript was scored
        import act_model
        acts = act_model.acts_of(act_model.load()[1])
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        p = act_model.score(sents)
        for a in rest:
            best = sents[int(p[:, acts.index(a)].argmax())]
            tail = " ".join(best.split()[-4:])
            found[a] = (next((t for t, x in zip(times, texts) if tail in x), times[-1]), _clip(best))
    return sorted(({"act": a, "say": SAY.get(a, a), "t": round(float(t), 1), "quote": q}
                   for a, (t, q) in found.items()), key=lambda m: m["t"])


def _reason(mv, peak):
    """Human-readable sentence - §4 requires one on every verdict."""
    if mv:
        return f"call contains {' and '.join(m['say'] for m in mv)} ({peak:.2f} risk)"
    if peak < 0.35:   # SAFE band: nothing scam-like to name
        return f"no scam signs found ({peak:.2f} risk)"
    return f"scam-like phrasing, no rule fired ({peak:.2f} risk)"


if __name__ == "__main__":
    # Fake word timings so the risk pipeline is testable without any audio yet.
    def fake_words(script, wps=2.5):
        return [(i / wps, (i + 1) / wps, w) for i, w in enumerate(script.split())]

    scam = ("hello sir I am calling from your bank your account is blocked "
            "due to KYC we need to verify please share the OTP you received "
            "otherwise your account will be suspended today share the OTP now")
    calm = ("hi ma am I am calling about the parcel you ordered last week "
            "it will arrive tomorrow between four and six in the evening "
            "please let us know if nobody is home thank you have a good day")

    for name, script, expect in [("scam", scam, True), ("normal", calm, False)]:
        win = windows(fake_words(script))
        times, texts = zip(*win)
        curve = risk_curve(window_risk(list(texts)))
        hit = alert_at(curve, times)
        print(f"{name:7} peak {curve.max():.2f}  alert at {hit}  "
              f"| {_reason(moves(times, texts), float(curve.max()))}")
        assert (hit is not None) == expect, f"{name} alert should be {expect}"
        mv = {m["act"]: m for m in moves(times, texts)}
        assert (set(mv) >= {"credential", "threat"}) if expect else not mv, mv
        if expect:   # the quote is the sentence that matched, said when it was heard
            assert "share the otp" in mv["credential"]["quote"].lower(), mv
            assert 0 < mv["credential"]["t"] <= times[-1], mv

    # stereo split, proven on a file we write here rather than assumed
    tmp = Path(sys.argv[1] if len(sys.argv) > 1 else "_stereo_check.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(np.array([[1, -1]] * 100, dtype=np.int16).tobytes())
    left, right = split_stereo(tmp)
    assert (left == 1).all() and (right == -1).all(), "channels crossed"
    tmp.unlink()
    print("self-check ok\n")

    # §5 eval set. Weights, context length and the combiner were fitted on DEV
    # (26) and nothing else; TEST (24) was scored once, after.
    from sklearn.metrics import roc_auc_score
    d = evaluate()
    print("50 authored transcripts, dev-tuned / test-held-out")
    for name in ("dev", "test"):
        x = d[d.split == name]
        y, a = x.label.to_numpy(), x.alert.to_numpy()
        tp, fp, fn = int((a & (y == 1)).sum()), int((a & (y == 0)).sum()), int((~a & (y == 1)).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"  {name:5} n={len(x)}  peak-risk AUC {roc_auc_score(y, x.peak):.3f}  |  "
              f"alert P {pr:.2f} R {rc:.2f} F1 {2 * pr * rc / max(pr + rc, 1e-9):.2f}"
              f"  (tp {tp} fp {fp} fn {fn})")
    bad = d[(d.label == 0) & d.alert]
    print()
    print(f"false alarms: {dict(zip(bad.id, bad.type)) or 'none'}")
    miss = d[(d.label == 1) & ~d.alert]
    print(f"missed: {sorted(miss.type)}")
    print("Most misses are scams with no transactional vocabulary - a relationship "
          "being built rather than a transaction pushed - which is what the "
          "behaviour layer, not the text, has to catch. A spoken-register "
          "classifier was tried for the rest and FAILED (speech_model.py): "
          "trained scam-vs-legit it learns which corpus the text came from, not "
          "what the caller is doing. Acts, not topics - see act_model.py.")

    # External validation (§5's honesty check). The 50 transcripts above were
    # written by the same person who wrote the rules, so on their own they
    # measure the author. These are 1,353 real robocalls the FTC collected from
    # the cease-and-desist letters it sent to gateway carriers - calls that
    # actually reached American phones. Skipped when the corpus is absent.
    if (data.DATA / "vishing" / "robocall" / "metadata.csv").exists():
        r = data.load_robocalls()
        r["peak"] = [score_transcript(t)["peak_risk"] for t in r.transcript]
        r["words"] = r.transcript.str.split().str.len()
        caught = r.peak >= 0.35
        print(f"\n{len(r)} real FTC robocalls (English, deduped): "
              f"caught {caught.mean():.1%}")
        for lo, hi in [(0, 30), (30, 50), (50, 80), (80, 10 ** 6)]:
            g = r[(r.words >= lo) & (r.words < hi)]
            print(f"  {lo:>3}-{'+' if hi > 10 ** 5 else hi:<4} words  n={len(g):>4}  "
                  f"caught {caught[g.index].mean():>6.1%}")
        print("A CATCH RATE, NOT AN ACCURACY - every row is a robocall, the "
              "corpus has no benign half. The gradient is real: misses are "
              "recordings that captured too little speech to judge (voicemail "
              "fragments, IVR tails), not calls the text layer read and cleared.")
        print("AND SINCE 2026-08-20 THIS NUMBER IS IN-SOURCE: act_model was "
              "trained on LLM labels over THIS corpus, so it is no longer an "
              "external check on the act layer. The honest external numbers are "
              "in `scratch/_voice_llm_eval.py` (2026-10-25, combination rule on): "
              "27 recorded calls AUC 0.850, false alarms on 723 held-out real bank "
              "calls 0.3%, held-out scambait channels caught 84.2%, 105 calls from "
              "seven never-seen scambait channels caught 96.2%, 706 real "
              "Korean FSS calls (machine-translated) caught 61.9%. None of those "
              "is in any training set here.")

    # Conversational check (--calls). Real two-party calls, both classes, from
    # YouTube: 11 scam-call recordings vs 15 legitimate transactional calls.
    # Costs ~16 min of Whisper, hence the flag.
    #
    # THE RESULT IS NEGATIVE AND IT MATTERS: peak-risk AUC 0.433, WORSE THAN
    # CHANCE, alert P 0.27 R 0.27 against P 0.89 R 0.67 on the authored set. A
    # legitimate bank customer-service call scores 0.994 ("scam-like phrasing,
    # no rule fired" - the neural model, not the keyword layer) while a real
    # scam call scores 0.000. The content layer keys on TRANSACTIONAL BANKING
    # VOCABULARY, not on scam intent. §6's escalation is unsupported here too:
    # scam risk runs 0.079 -> 0.053 across the call, rising in 36% of calls,
    # peaking a median 45 s in.
    #
    # Both halves of the corpus are compromised and the number must not be
    # quoted bare: the legitimate half is scripted mock/training calls, dense
    # with transactional vocabulary in a way a real call is not, and the scam
    # half is scambait, diluted by the baiter's commentary and time-wasting -
    # which is why the longest scam calls score near zero. The failure is
    # systematic and interpretable, so the conclusion stands even though 0.433
    # is not a clean measurement of the model.
    if "--calls" in sys.argv and (data.DATA / "vishing" / "calls" / "index.json").exists():
        import json
        from sklearn.metrics import roc_auc_score
        c = json.load(open(data.DATA / "vishing" / "calls" / "index.json"))
        got = []
        for x in c:
            v = analyse(x["path"])
            txt = " ".join(t for _, t in (v.get("windows") or []))
            if sum(ord(ch) > 127 for ch in txt) / max(len(txt), 1) < 0.02:  # English only
                got.append({"label": x["label"], "peak": v["peak_risk"],
                            "alert": v["alert_at_s"] is not None, "title": x["title"]})
        y = np.array([r["label"] for r in got])
        pk = np.array([r["peak"] for r in got])
        al = np.array([r["alert"] for r in got])
        print(f"\n{len(got)} real two-party calls: {int(y.sum())} scam / "
              f"{int((1 - y).sum())} legit")
        print(f"  peak-risk AUC {roc_auc_score(y, pk):.3f}  "
              f"(the authored set reads 0.865 - this is the transfer gap)")
        for lab, name in [(1, "scam"), (0, "legit")]:
            m = y == lab
            print(f"  {name:<6} peak mean {pk[m].mean():.3f}  alerts {al[m].mean():.0%}")
