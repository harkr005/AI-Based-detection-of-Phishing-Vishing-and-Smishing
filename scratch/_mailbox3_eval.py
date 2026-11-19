"""THIRD MAILBOX - the held-out person the cross-user test has been waiting for.

`Dataset(Indian)/email/emails_for_training.zip` is a Google Takeout from a third
person, already reduced to one jsonl (subject / body / sender_domain / category).
It is promotions + updates only, i.e. exactly the train genre, and it is 100% HAM
(mail he kept), so every non-ham prediction is a false positive - no threshold to
pick, no labels to trust.

Standing here matters: SHIPPED and MODERN never saw any personal mail beyond
`self`; v2 saw self; v3 saw self + user2. NOBODY has seen this person, so all
four columns are cross-user for the first time.

Two things this corpus is NOT, both stated because they move the number:
  - It is CLEANED: URLs replaced with the literal "[URL]". Training ham carried
    real links. The last block re-injects a URL to size that gap.
  - It has NO DATE, so there is no temporal slice - this is "all mail", not
    "2025-26 mail" like every earlier row.

PRIVATE DATA. Rates and sender domains may go in the report; message text may not.
"""
import json
import pathlib
import zipfile

import data                      # before torch
import pandas as pd
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
ZIP = ROOT / "Dataset(Indian)" / "email" / "emails_for_training.zip"
# Shipped is a byte-copy of v6 since 2026-10-02 (v5 shipped and rolled back that day), so it is not listed twice.
MODELS = [("UNIFIED v2", "email_distilbert_v2"), ("UNIFIED v3", "email_distilbert_v3"),
          ("v4", "email_distilbert_v4"), ("v5", "email_distilbert_v5"),
          ("v6 SHIPPED", "email_distilbert_v6")]
PHISH, SPAM = em.LABELS.index("phishing"), em.LABELS.index("spam")


def load():
    with zipfile.ZipFile(ZIP) as z:
        rows = [json.loads(l) for l in z.open(z.namelist()[0]).read().decode("utf-8").splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    # Same field order as data._mail_row, or the model sees a different document.
    d["text"] = (d.subject.fillna("") + data.NL + d.body.fillna("")).str.strip()
    return d


def rates(d, tag_col="text"):
    """-> {model: (phish%, spam%)} over one frame."""
    out = {}
    for tag, name in MODELS:
        em.OUT = ROOT / "models" / name
        em.load.cache_clear()
        tok, model = em.load()
        pred = em.predict(tok, model, d[tag_col])[0].argmax(1)
        out[tag] = (100 * (pred == PHISH).mean(), 100 * (pred == SPAM).mean(), pred)
    return out


if __name__ == "__main__":
    d = load()
    # 791 rows collapse to ~400 texts - one Classroom template repeats. A raw rate
    # is really "how often does the model dislike his top 3 senders", so both.
    u = d.drop_duplicates("text")
    print(f"{len(d)} messages, {len(u)} unique, {d.sender_domain.nunique()} sender domains")
    print(d.category.value_counts().to_string())
    print("top senders:", ", ".join(f"{k} {v}" for k, v in d.sender_domain.value_counts().head(6).items()))
    cleaned = 100 * d.body.str.contains("[URL]", regex=False).mean()
    print(f"carry a stripped [URL]: {cleaned:.0f}%\n", flush=True)

    print(f"{'slice':32s} {'n':>5}" + "".join(f"{t:>16s}" for t, _ in MODELS))
    print(f"{'':32s} {'':>5}" + "".join(f"{'phish% / spam%':>16s}" for _ in MODELS))
    full, uniq = rates(d), rates(u)
    for label, frame, r in [("all messages", d, full), ("unique texts", u, uniq)]:
        print(f"{label:32s} {len(frame):5d}" + "".join(
            f"{r[t][0]:8.1f} /{r[t][1]:6.1f}" for t, _ in MODELS), flush=True)

    # A rate is not a finding; the sender list is (same move as _crossuser_fp.py).
    for tag, _ in MODELS:
        fp = d[(full[tag][2] == PHISH) | (full[tag][2] == SPAM)]
        if len(fp):
            top = fp.sender_domain.value_counts().head(5)
            tot = d.sender_domain.value_counts()
            print(f"\n{tag} flags {len(fp)}/{len(d)}: " + ", ".join(
                f"{k} {v}/{tot[k]}" for k, v in top.items()))

    # The cleaning gap: does putting a link back change the verdict? The corpus
    # cannot be un-cleaned, so this sizes the distortion, it does not remove it.
    p = u.assign(text=u.text.str.replace("[URL]", "https://notifications.google.com/a/b?c=1", regex=False))
    back = rates(p)
    print(f"\n{'[URL] -> a real link':32s} {len(p):5d}" + "".join(
        f"{back[t][0]:8.1f} /{back[t][1]:6.1f}" for t, _ in MODELS))
