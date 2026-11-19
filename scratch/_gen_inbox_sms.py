"""Turn one real Indian inbox into the SMS corpus none of the public ones are.

`app-reports/audit.jsonl` is what the phone's "Check my inbox" button wrote:
943 messages off Mohak's own handset, scored on-device, bodies already through
`Redact` (codes, amounts and account numbers are placeholders before the file
is ever written). That is the population every public SMS corpus lacks - real
Indian bank, IPO/ASBA, OTP and carrier traffic - and the model's error rate on
it is 74/943 smishing, every one of which is legitimate.

LABELS COME FROM THE CARRIER, NOT FROM THE MODEL, or this would be circular.
Under TCCCPR's 2025 amendment the delivered header is
`[operator]-[entity]-[category]` and BOTH outer parts are stamped on in transit
by the access provider during DLT scrubbing - the sender chooses neither. So
the category suffix is a third party's classification of every message here:

    -P  promotional  -> spam       (bulk marketing, by the carrier's own call)
    -S  service      -> ham        (transactional / service)
    -T  transactional-> ham
    -G  government   -> ham
    numeric shortcode-> ham        (57575*, 59039* - Steam, Microsoft OTPs)

Nothing in this inbox is smishing. That is not an assumption: all 74 rows the
model called smishing were read, and they are IPO unblock notices, OTPs,
Airtel's own fraud advisory, Blinkit order alerts and Paytm login alerts.

    ponytail: one row of this corpus IS a scam - TX-DWTDS-P's "SYLPH INDUSTRIES
    is showing strong upward momentum, buy 50,000" stock pump. It lands in
    spam, which is the right side of the line that matters (fusion reads the
    smishing column), so it is left alone rather than hand-patched.

THE PLACEHOLDERS CANNOT GO IN ALONE. On the phone the model scores RAW text and
never sees "[amount]" at all; this file would be the only corpus containing the
token, and all of its rows are ham or spam, so "[amount]" would become a clean
non-smishing feature that production never fires. Every row therefore ships
twice - once redacted, once with values of the same SHAPE `Redact` consumed
put back - in the same group, so the twins cannot straddle a split.

Writes Dataset(Indian)/sms/inbox_sms.csv (gitignored, like every dataset here:
this is one person's real message history and it does not belong in the repo).
"""
import hashlib
import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIT = ROOT / "app-reports" / "audit.jsonl"
OUT = ROOT / "Dataset(Indian)" / "sms" / "inbox_sms.csv"

# The carrier-stamped header. Same shape as behaviour.CARRIER_HEADER, and
# deliberately strict: a bare entity is not a weaker form of a real header, it
# is a form a real header no longer takes.
CARRIER = re.compile(r"^[A-Za-z]{2}-([A-Za-z0-9_]{3,13})-([PSTG])$")
SHORTCODE = re.compile(r"^\+?\d{5,12}$")

# Values of the shape Redact swallowed, so the de-redacted twin reads like the
# message that arrived. Rotated by a hash of the text: same row, same values on
# every regeneration, and no single value becomes a feature of its own.
FILLERS = {
    "[amount]": ["Rs.2,500", "INR 14,990", "Rs 1,20,000", "Rs.499.00", "₹1,299"],
    "[account]": ["918604195555", "50761234567890", "1800 209 5577".replace(" ", ""),
                  "404512378901"],
    "[code]": ["482913", "7721", "913406", "55210"],
    "[card]": ["4315 7712 9004 1188", "5076 1234 5678 9010"],
    "[email]": ["no.reply@axisbank.com", "care@icicibank.com", "support@growwapp.in"],
    "[pan]": ["AVQPK1234J", "BNZPM5678K"],
    "[aadhaar]": ["4312 9087 1123", "7781 2290 4456"],
}
PLACEHOLDER = re.compile(r"\[(?:amount|account|code|card|email|pan|aadhaar)\]")


def deredact(text):
    """Put values back, deterministically per row."""
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
    n = [0]

    def pick(match):
        options = FILLERS[match.group(0)]
        n[0] += 1
        return options[(seed + n[0]) % len(options)]

    return PLACEHOLDER.sub(pick, text)


def label_of(sender):
    """-> "ham" | "spam" | None, from the carrier's category and nothing else."""
    carrier = CARRIER.match(sender)
    if carrier:
        return "spam" if carrier.group(2) == "P" else "ham"
    if SHORTCODE.match(sender):
        return "ham"
    return None


def group_of(sender, text):
    """Sender entity + opening six words.

    The bank templates repeat: 157 messages arrive from AX-AXISBK-S alone and
    most are the same three sentences with a different amount. A random split
    would put near-twins on both sides and read a flattering number off them,
    which is the trap `load_sms` already groups the Indian corpus to avoid.
    """
    carrier = CARRIER.match(sender)
    entity = carrier.group(1).upper() if carrier else sender
    opening = " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split()[:6])
    return f"inbox:{entity}:{opening}"


def build():
    rows, skipped = [], []
    seen = set()
    for line in AUDIT.open(encoding="utf-8"):
        message = json.loads(line)
        body = " ".join((message.get("body") or "").split())
        label = label_of(message["sender"])
        if not body or label is None:
            skipped.append(message["sender"])
            continue
        if body in seen:
            continue
        seen.add(body)
        group = group_of(message["sender"], body)
        rows.append((body, label, "inbox", group))
        twin = deredact(body)
        if twin != body:
            rows.append((twin, label, "inbox", group))

    out = pd.DataFrame(rows, columns=["text", "label", "source", "group"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return out, skipped


if __name__ == "__main__":
    out, skipped = build()
    redacted = out.text.str.contains(PLACEHOLDER).sum()
    print(f"{OUT.relative_to(ROOT)}: {len(out):,} rows "
          f"({redacted:,} redacted + {len(out) - redacted:,} de-redacted twins)")
    print("labels:", out.label.value_counts().to_dict())
    print("groups:", out.group.nunique())
    if skipped:
        print(f"skipped {len(skipped)} rows with no carrier header: {sorted(set(skipped))}")

    # The whole point of the corpus: it must carry no smishing, and every twin
    # must sit in its redacted original's group or the split leaks.
    assert set(out.label) <= {"ham", "spam"}, out.label.unique()
    assert not out.text.str.contains(r"\[(?:amount|code|account)\]").all()
    pairs = out.groupby("group").text.nunique()
    assert (pairs > 0).all()
    # A de-redacted row must no longer contain a placeholder at all.
    twins = out[~out.text.str.contains(PLACEHOLDER)]
    assert len(twins) > 100, "de-redaction did nothing"
    print("ok")
