"""The Inbox box - 21,253 messages that no model in this project has ever seen.

Every personal-mail number so far (train AND test) came from
HAM_BOXES = {Category promotions, Category updates}. `Inbox.mbox` was never
loaded by anything. Gmail exports a message once per label and
`load_personal_mail` dedups per person alphabetically, so the promotions/updates
copies win and what stays tagged "Inbox" is exactly the mail that carries NO
category label: primary personal mail, 1:1 correspondence, receipts, college
threads. A different genre, not just more of the same rows.

WHOSE INBOX: self's. The two takeout zips in Dataset(Indian)/email
are self's own export - byte-identical to Datasets/personal/*.mbox - and
user21.zip ships promotions/updates/spam only. There is no third-party Inbox
anywhere on disk, so THIS IS NOT A CROSS-USER TEST. It is held out by BOX, and
v2/v3 saw this person's other boxes. Read every number below as a floor.

What it does buy, and mailbox 3 could not:
  - real links (mailbox 3 replaced every URL with the literal "[URL]")
  - real dates (mailbox 3 shipped none, so it had no 2025-26 slice)
100% ham - mail he kept - so every non-ham prediction is a false positive.

PRIVATE DATA. Rates and sender domains may go in the report; message text may not.
"""
import hashlib
import pathlib
import re

import data                      # before torch
import pandas as pd
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
# models/email_distilbert/model.pt is a byte-identical copy of v3 - v3 was promoted
# on 2026-08-22 23:32 and the outgoing checkpoint was kept as _public (same mtime,
# and the name fits: it is the only email checkpoint not trained on private mail).
# So "SHIPPED" and "v3" are ONE column, and the pre-modern baseline is _public.
# _prev is older still (2026-08-21 20:44), not the baseline. Keep the md5 print or
# two columns can share weights silently again.
# 2026-09-23: v4 (v3's recipe on the fixed mail parser) was promoted, so
# models/email_distilbert is now a copy of v4 (md5 5ec60ee7), not v3.
MODELS = [("BASELINE public", "email_distilbert_public"), ("MODERN", "email_modern"),
          ("UNIFIED v2", "email_distilbert_v2"), ("v3", "email_distilbert_v3"),
          ("SHIPPED = v4", "email_distilbert_v4")]
PHISH, SPAM = em.LABELS.index("phishing"), em.LABELS.index("spam")
URL = re.compile(r"https?://\S+")


def rates(slices):
    """-> {slice: {model: (phish%, spam%, pred)}}, one checkpoint load per model."""
    out = {}
    for tag, name in MODELS:
        em.OUT = ROOT / "models" / name
        em.load.cache_clear()
        tok, model = em.load()
        for label, df in slices:
            pred = em.predict(tok, model, df.text)[0].argmax(1)
            out.setdefault(label, {})[tag] = (
                100 * (pred == PHISH).mean(), 100 * (pred == SPAM).mean(), pred)
    return out


def table(out, slices):
    print(f"\n{'slice':40s} {'n':>6}" + "".join(f"{t:>16s}" for t, _ in MODELS))
    print(f"{'':40s} {'':>6}" + "".join(f"{'phish% / spam%':>16s}" for _ in MODELS))
    for label, df in slices:
        r = out[label]
        print(f"{label:40s} {len(df):6d}" + "".join(
            f"{r[t][0]:8.1f} /{r[t][1]:6.1f}" for t, _ in MODELS), flush=True)


if __name__ == "__main__":
    for tag, name in MODELS:      # two identical hashes = two identical columns
        h = hashlib.md5((ROOT / "models" / name / "model.pt").read_bytes()).hexdigest()
        print(f"{tag:14s} {name:24s} {h[:12]}")
    # ponytail: parquet cache, the mbox key scan is 3.2 GB per run. Delete the
    # file to re-read the mailbox; it lives under the private dir on purpose.
    cache = data.PERSONAL / "_self_6000.parquet"
    if cache.exists():
        d = pd.read_parquet(cache)
    else:
        d = data.load_personal_mail(person="self", limit=6000)
        d.to_parquet(cache)
    print(d.groupby(["mbox", "label_3"]).size().to_string(), flush=True)

    d["sender_domain"] = d.sender.str.extract(r"@([\w.-]+)", expand=False).str.lower()
    ham = d[(d.label_3 == "ham") & d.date.dt.year.between(2021, 2026)]
    box, cat = ham[ham.mbox == "Inbox"], ham[ham.mbox.isin(
        {"Category promotions", "Category updates"})]
    has_url = 100 * box.text.str.contains(URL).mean()
    print(f"\nInbox ham 2021-26: {len(box)}, {box.sender.nunique()} senders, "
          f"{has_url:.0f}% carry a real link", flush=True)

    slices = [("Inbox ham ALL (unseen box)", box),
              ("Inbox ham 2025-26", box[box.date.dt.year >= 2025]),
              ("Inbox ham <=2024", box[box.date.dt.year <= 2024]),
              ("promo/updates 2025-26 (the seen box)", cat[cat.date.dt.year >= 2025])]
    slices = [(l, f) for l, f in slices if len(f)]
    out = rates(slices)
    table(out, slices)

    # A rate is not a finding; the sender list is (same move as _crossuser_fp.py).
    tot = box.sender_domain.value_counts()
    for tag, _ in MODELS:
        p = out["Inbox ham ALL (unseen box)"][tag][2]
        fp = box[(p == PHISH) | (p == SPAM)]
        if len(fp):
            print(f"\n{tag} flags {len(fp)}/{len(box)}: " + ", ".join(
                f"{k} {v}/{tot[k]}" for k, v in fp.sender_domain.value_counts().head(6).items()))

    # Mailbox 3 was URL-stripped and _mailbox3_eval could only guess what that cost
    # by injecting a fake link. Here the real links exist, so strip them the same
    # way and read the gap directly - this sizes that correction, in this genre.
    strip = [("Inbox ham ALL, URLs -> \"[URL]\"",
              box.assign(text=box.text.str.replace(URL, "[URL]", regex=True)))]
    table(rates(strip), strip)
