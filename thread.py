"""SMS threads: score the conversation, not each message (spec §6).

"Conversational scams start innocent and turn malicious later" - so the unit of
scoring is the thread, and a message that was fine on its own stops being fine
once the thread it lives in has asked for an OTP.

Nothing here is a new model. Three pieces, all reused:
  per-message risk = max(smishing prob, URL score)   §6: SMS text alone is weak,
                                                     the link carries the weight
  rolling curve    = voice_model.risk_curve          same asymmetric EMA ratchet
  bands            = §3 SAFE / SUSPICIOUS / PHISHING

The rolling window is the last WINDOW messages. Risk ratchets *inside* a
conversation (one polite message does not clear an OTP ask), but a thread does
not stay hot forever - old messages fall out of the window instead.

Sender metadata - DLT header format, first-contact, burst rate, odd hour (§7) -
is NOT here. That is the behaviour engine's input, and it needs fields the SMS
corpora do not carry. This module scores content over time only.
"""
import re

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data  # noqa: F401
import numpy as np

import sms_model
import url_model
from voice_model import risk_curve

# Bare-domain form matters: smishing SMS routinely drop the scheme ("sbi-kyc.tk/x").
URL_RE = re.compile(
    r"https?://\S+|www\.\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|in|net|org|co|io|tk|xyz|ml|ga|cf|top|link|info|ru)\b(?:/\S*)?",
    re.I)

# `\S+` cannot tell a URL's last character from the punctuation around it, and
# real SMS wrap links in brackets and end sentences with them. Found in a real
# inbox 2026-09-22: "<tinyurl.com/unblk>" was extracted as `tinyurl.com/unblk>`
# and scored as its own host. Stripped here rather than in the pattern, because
# the pattern has three branches and each would need the same guard.
_TRAILING = ".,;:!?'\"()[]{}<>"


def find_urls(text):
    """Every link in the text, without the punctuation that follows it.

    The one place URLs are pulled out of a message - api._links, thread.url_risk
    and the phone's UrlModel.links all go through this shape, so a URL scored on
    one path is the same string on every other.
    """
    return [u.rstrip(_TRAILING) for u in URL_RE.findall(text or "")]

WINDOW = 20      # messages kept in the rolling window
SUSPICIOUS, PHISHING = 0.35, 0.70  # §3 bands


def url_risk(texts):
    """-> per-message max URL score, 0.0 where the message has no link.

    One batched call for the whole thread; url_model.load() re-reads the booster
    on every call, so calling it per message is the expensive way to be wrong.
    """
    found = [find_urls(t) for t in texts]
    flat = [u for us in found for u in us]
    if not flat:
        return np.zeros(len(texts))
    scores, out, i = list(url_model.score(flat)), [], 0
    for us in found:
        out.append(max(scores[i:i + len(us)], default=0.0))
        i += len(us)
    return np.array(out, dtype=float)


def score_thread(messages, window=WINDOW):
    """Oldest-first list of message texts -> verdict dict for the conversation."""
    msgs = list(messages)[-window:]
    if not msgs:
        return {"verdict": "SAFE", "risk": 0.0, "escalation": 0.0,
                "reason": "empty thread", "curve": []}

    smish = sms_model.score(msgs)[:, sms_model.LABELS.index("smishing")]
    urls = url_risk(msgs)
    curve = risk_curve(np.maximum(smish, urls))

    risk = float(curve.max())
    return {
        "verdict": "PHISHING" if risk >= PHISHING else
                   "SUSPICIOUS" if risk >= SUSPICIOUS else "SAFE",
        "risk": round(risk, 3),
        # the climb itself is the signal - a thread that starts at 0.05 and ends
        # at 0.8 is the conversational scam shape §6 describes
        "escalation": round(float(curve[-1] - curve[0]), 3),
        "peak_at": int(curve.argmax()),
        "reason": _reason(msgs, smish, urls, curve),
        "curve": [round(float(c), 3) for c in curve],
    }


def _reason(msgs, smish, urls, curve):
    """Human-readable sentence - §4 requires one on every verdict.

    The word/feature-level explanation is computed for the peak message only:
    it costs a forward pass per word, and the rest of the thread did not drive
    the verdict.
    """
    i = int(curve.argmax())
    if curve[i] < SUSPICIOUS:  # nothing to justify, and explaining costs a pass per word
        return f"no risky content across {len(msgs)} message(s)"
    if urls[i] > smish[i]:
        link = max(find_urls(msgs[i]), key=lambda u: url_model.score([u])[0])
        driver = "%s (%s)" % (url_model.explain(link)[1], link)
    else:
        driver = sms_model.explain(msgs[i])[2]
    where = "message 1" if i == 0 else f"message {i + 1} of {len(msgs)}"
    if i > 0 and curve[i] - curve[0] >= 0.2:
        return (f"thread escalated: risk rose {curve[0]:.2f} -> {curve[i]:.2f} by "
                f"{where}; {driver}")
    return f"{where} carries the risk ({curve[i]:.2f}); {driver}"


if __name__ == "__main__":
    slow_burn = [
        "Hello, am I speaking with the account holder?",
        "This is Priya from HDFC Bank customer care, calling regarding your account.",
        "Sir your KYC verification is pending, it will take just 2 minutes.",
        "Please complete it here http://hdfc-kyc-verify.tk/update before 6pm or "
        "your account will be blocked.",
    ]
    normal = [
        "Hey, are we still meeting at 6 for dinner?",
        "Yeah, booked a table at the usual place.",
        "Cool, I'll come straight from work.",
        "See you there.",
    ]
    # the whole point of thread scoring: the scam message is followed by chit-chat
    cooled = slow_burn + ["Thank you for your time sir.", "Have a good day."]

    for name, thread, expect in [("slow-burn", slow_burn, "PHISHING"),
                                 ("normal", normal, "SAFE"),
                                 ("cooled-off", cooled, "PHISHING")]:
        r = score_thread(thread)
        print(f"{name:11} {r['verdict']:10} risk {r['risk']:.2f}  "
              f"escalation {r['escalation']:+.2f}\n            {r['reason']}")
        assert r["verdict"] == expect, f"{name} should be {expect}, got {r['verdict']}"

    # a benign tail must not clear the thread - per-message scoring would let it
    assert score_thread(cooled)["curve"][-1] >= SUSPICIOUS, "ratchet leaked"
    assert url_risk(["no link here"])[0] == 0.0
    assert find_urls("pay at hdfc-kyc.tk/x now") == ["hdfc-kyc.tk/x"]
    # the real-inbox case: a link wrapped in angle brackets, and one ending a
    # sentence. Both used to be scored with the punctuation glued on.
    assert find_urls("unblock <tinyurl.com/unblk> now") == ["tinyurl.com/unblk"]
    assert find_urls("verify at hdfc-kyc.tk/x.") == ["hdfc-kyc.tk/x"]
    print("ok")
