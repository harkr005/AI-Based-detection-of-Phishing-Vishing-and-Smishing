"""What does `email_model.solicitation` cost, and what does it buy?

WHY IT EXISTS. A Gmail-spam-labelled paid-internship pitch scored ham 0.986 /
spam 0.000 on the shipped model, and that is not a one-off: all 73 messages in
the two Gmail Spam boxes score ham (median spam prob 0.000), the pre-modern
`email_distilbert_public` checkpoint reaches 2. The spam class is dead on 2026
mail because v3's modern training data is promotions+updates as HAM, and Gmail's
spam box is that same genre - newsletters and course marketing the recipient
stopped wanting. Consent is not in the text.

So the rule does not chase Gmail. It covers ONE genre - the cold paid-course /
paid-internship pitch - and this harness measures the only thing that can be
measured with one labelled positive: how often it fires on real mail that is
NOT that genre. Precision on 13,290 real ham messages, two mailboxes.

Run:  python scratch/_solicitation_eval.py
"""
import mailbox
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                       # noqa: E402  # before torch, see sms_model
import pandas as pd               # noqa: E402
from email_model import solicitation, SOLICITATION, _SOL  # noqa: E402

# The message that started this: real, Gmail-labelled spam, scored ham 0.986 /
# spam 0.000. Phone digits masked - it is someone's number.
PITCH = """Training and Internship Program - Registration Now Open
Dear Students,
Greetings from the Training Coordination Team.
We are pleased to inform you that registrations are now open for the Google
Certified Training & Internship Program, conducted by Internzlearn, a Google
Certified Training Partner. This three-month online program is designed to
provide students with structured technical training, practical project
experience, internship exposure, and career development support.
Certificates and Recognition: Internship Completion Certificate, AICTE & ISO
Certified Course Completion Certificate, Letter of Recommendation.
Registration Fee: participation in the program is subject to the applicable
registration fee and program terms.
Referral Code: ACC30 - please enter the referral code during registration to
avail the applicable concession.
Placement assistance and interview preparation are included.
National Training Coordinator, Phone/WhatsApp: +91 8XXXX XXXXX"""


def boxes():
    """(name, texts) for every ham/spam slice on disk. Parquet cache is the one
    _inbox_eval.py writes; the mbox key scan is 3.2 GB per run without it."""
    d = pd.read_parquet(data.PERSONAL / "_self_6000.parquet")
    promo = d.mbox.isin(["Category promotions", "Category updates"])
    yield_ = [("self ham, promo/updates (train genre)", d[(d.label_3 == "ham") & promo]),
              ("self ham, Inbox/Archived/social", d[(d.label_3 == "ham") & ~promo])]
    out = [(n, list(f.text), f) for n, f in yield_]

    har = data.PERSONAL / "user2"
    rows = []
    for f in sorted(har.glob("*.mbox")):
        box = mailbox.mbox(str(f))
        for k in box.keys():
            r = data._mail_row(box[k], f.name)
            if r and len(r["text"]) > 5:
                r["label_3"] = "spam" if f.stem.lower() == "spam" else "ham"
                rows.append(r)
    h = pd.DataFrame(rows)
    out.append(("other person, ALL ham (unseen mailbox)",
                list(h[h.label_3 == "ham"].text), h[h.label_3 == "ham"]))
    out.append(("Gmail's own spam boxes (both people)",
                list(d[d.label_3 == "spam"].text)
                + list(h[h.label_3 == "spam"].text), None))
    return out


if __name__ == "__main__":
    print(f"markers: {', '.join(SOLICITATION)}   (fires on fee|referral + 2 more)")
    total = 0
    for name, texts, df in boxes():
        fired = [t for t in texts if solicitation(t)]
        total += len(texts)
        print(f"\n{name:40s} n={len(texts):6d}  fires {len(fired):3d} "
              f"({100 * len(fired) / max(len(texts), 1):.2f}%)")
        for t in fired[:6]:
            print(f"    {sorted(solicitation(t))}  {t.splitlines()[0][:58]}")

    print(f"\nthe pitch that started this -> {solicitation(PITCH)}")
    assert solicitation(PITCH), "the one labelled positive must fire"
    assert not solicitation("Dear Students, the CTF registration closes on "
                            "Friday. Completion certificates will be issued."), \
        "a college notice with no commercial ask must stay quiet"
    print(f"ok - checked against {total:,} real messages")
