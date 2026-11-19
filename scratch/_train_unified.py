"""One 3-class email model that also works in 2026.

Keeps the shipped recipe exactly - CEAS Aug-2008 days 1-7 for ham/spam, the §5
temporal phishing split, §9.9 augmentation - and adds the two modern halves the
project was missing:

  + phishing_pot English <=2024          modern phishing
  + personal promotions/updates <=2024   modern ham, NOW TWO MAILBOXES

Everything from 2025 onward is held out on both modern sources, so the modern
slice is a genuine next-campaign test rather than a resample.

Scored against the shipped DistilBERT, v2 (one mailbox) and the modern-only model
on all three test sets, because the point is not "does it beat one number" - it is
whether modern data can be added WITHOUT losing the 2008 ones.

v3 adds a second person's ham (2026-08-22). The <=2024 / 2025+ rule is applied to
him unchanged, so he contributes 564 mails to train and 1,851 to the held-out
slice, and the modern numbers are reported PER PERSON: adding someone can lower
the pooled false-positive rate while doing nothing for the person it was meant to
fix. A THIRD mailbox is being collected and is the next held-out-by-person test -
v2 stays on disk as the never-saw-him baseline.

WEIGHTS DERIVE FROM PRIVATE MAIL. Do not publish, push, or demo them externally.
"""
import pathlib
import re
import sys

import data                      # before torch
import numpy as np
import pandas as pd
import adversarial
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
HAM_BOXES = {"Category promotions", "Category updates"}

# --out NAME trains (or with --score, reloads) a different checkpoint. v4 is v3's
# recipe unchanged, on text from the 2026-09-23 parser fix: v3 trained on bodies
# where <style>/<script> contents survived as text (43.8% of personal-mail rows).
NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "email_distilbert_v3"
em.OUT = ROOT / "models" / NAME
# Training writes straight into OUT. Without this a bare run silently replaced
# v3 - the model every later number is compared against.
if not {"--score", "--dump"} & set(sys.argv) and (em.OUT / "model.pt").exists():
    sys.exit(f"{em.OUT} already holds a model - pick a new --out, or pass --score")
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())
english = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(
    re.findall(r"[a-z]+", str(t).lower()[:2000])) >= 0.04

# --- the shipped recipe, unchanged -----------------------------------------
e = data.load_email()
aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
ph_tr, ph_te = em.phishing_split(e)
old_tr = pd.concat([aug[aug.date.dt.day <= 7], ph_tr])
te_time = pd.concat([aug[aug.date.dt.day == 8], ph_te])
te_src = e[e.corpus.isin(["Enron", "Ling", "SpamAssasin"])]

# --- the two modern halves --------------------------------------------------
pot = data.load_phishing_pot()
pot = pot[pot.text.map(english) & pot.date.dt.year.between(2022, 2026)].copy()
pot["label_3"] = "phishing"
allmail = data.load_personal_mail(limit=6000)
ham = allmail[allmail.mbox.isin(HAM_BOXES) & (allmail.label_3 == "ham")
              & allmail.date.dt.year.between(2021, 2026)].copy()
modern = pd.concat([pot, ham], ignore_index=True)
# --holdout NAME (v7a): that person is in NO training row, ham or spam, so his
# mail is a never-seen-person test. pot rows have no person, so they stay.
HOLD = sys.argv[sys.argv.index("--holdout") + 1] if "--holdout" in sys.argv else None
mod_tr = modern[(modern.date.dt.year <= 2024) & (modern.person != HOLD)]
te_mod = modern[modern.date.dt.year >= 2025]

# --- --daily (v5): person-to-person ham, the genre v4 never saw -------------
# v4 flags 11% of Enron-Spam ham as phishing at 87% mean confidence: its only ham
# was mailing lists and Gmail promotions/updates, so "personal tone + money
# talk" meant phishing. Colleague mail from BOTH eras goes in, so "old-looking =
# safe" is not a shortcut: Enron 2000-02 (144 mailboxes, the six Enron-Spam ones
# held out as unseen people) and Primary-tab mail (Inbox minus every Category,
# which is what Message-ID dedup leaves in "Inbox"), <=2024 train / 2025+ test.
# te_mod is left untouched so every v4 number stays comparable.
#
# v6 (--enron 1500 --social 1500): v5's 5,000 Enron rows moved the false positives
# onto account/security notifications - third mailbox 3.0% -> 9.5% (aws.training,
# email.openai.com, accounts.google.com). So a smaller colleague dose, plus the
# Social tab <=2024 (login/notification mail from LinkedIn, Instagram...), a ham
# genre no checkpoint has trained on.
arg = lambda flag, default: int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
te_enron = te_daily = pd.DataFrame(columns=["text", "label_3"])
# The third person (user: user3's), never trained on, no dates: all promo/updates ham.
te_mb3 = te_enron
if "--daily" in sys.argv:
    import scratch._mailbox3_eval as mb3
    te_mb3 = mb3.load()
if "--daily" in sys.argv:
    enr = data.load_enron_full()
    te_enron = enr[enr.held_out]
    enr_tr = enr[~enr.held_out].sample(min(arg("--enron", 5000), (~enr.held_out).sum()), random_state=0)
    pm = data.load_personal_mail(person="self")
    keep = (pm.label_3 == "ham") & pm.date.dt.year.between(2021, 2026)
    prim = pm[keep & pm.mbox.isin({"Inbox", "Archived"})].copy()
    soc = pm[keep & (pm.mbox == "Category social") & (pm.date.dt.year <= 2024)]
    soc = soc.sample(min(arg("--social", 0), len(soc)), random_state=0).copy()
    del pm, enr                   # the full 3.2 GB parse is what ran the box out of memory
    te_daily = prim[prim.date.dt.year >= 2025]
    tr = pd.concat([old_tr, mod_tr, enr_tr, prim[prim.date.dt.year <= 2024], soc], ignore_index=True)
    print("daily train   enron", len(enr_tr), " primary", (prim.date.dt.year <= 2024).sum(),
          " social", len(soc))
else:
    tr = pd.concat([old_tr, mod_tr], ignore_index=True)

# --- --spam (v7): a spam class that works on 2026 mail ---------------------
# Every checkpoint so far learned spam from CEAS 2008 only, and scores 73/73 of
# the Gmail Spam boxes ham. This adds each person's Gmail Spam folder as spam
# (GOOGLE's judgement, not the person's - say so in the report) and the third
# mailbox (user3's: promo/updates ham, "[URL]"-cleaned, undated) as ham, so it
# stops being a test set. Spam is ~37 mails a person (Gmail purges at 30 days),
# all Aug 2026, so there is no time split: the held-out PERSON is the test.
# ponytail: plain repetition (--spam-rep) to be heard next to thousands of CEAS
# rows; class weights only balance the class, not 2008 vs 2026 inside it.
te_spam = pd.DataFrame(columns=["text", "label_3"])
if "--spam" in sys.argv:
    import scratch._mailbox3_eval as mb3
    pspam = allmail[allmail.label_3 == "spam"]
    te_spam = pspam[pspam.person == HOLD]
    sp_tr = pspam[pspam.person != HOLD]
    mb3_ham = mb3.load().assign(label_3="ham")
    tr = pd.concat([tr, mb3_ham] + [sp_tr] * arg("--spam-rep", 10), ignore_index=True)
    te_mb3 = te_spam.iloc[:0]     # trained on now, so no longer a test
    print("spam train   personal spam", len(sp_tr), dict(sp_tr.person.value_counts()),
          f"x{arg('--spam-rep', 10)}   third mailbox ham", len(mb3.load()),
          "  held-out spam", len(te_spam), flush=True)
del allmail
tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)

print("train        ", len(tr), dict(tr.label_3.value_counts()))
print("temporal     ", len(te_time), dict(te_time.label_3.value_counts()))
print("cross-source ", len(te_src), dict(te_src.label_3.value_counts()))
print("modern 25-26 ", len(te_mod), dict(te_mod.label_3.value_counts()))
print("  train ham by person", dict(mod_tr[mod_tr.label_3 == "ham"].person.value_counts()))
print("  test  ham by person", dict(te_mod[te_mod.label_3 == "ham"].person.value_counts()),
      flush=True)

# --dump DIR writes the exact train/test sets of this recipe and stops, so the
# paper's classical baselines (scratch/_paper_baselines.py) see the rows the
# shipped model saw. DIR must sit under Datasets/: the rows are private mail.
if "--dump" in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index("--dump") + 1]).resolve()
    # Private mail bodies: only ever under the gitignored Datasets/.
    assert out.is_relative_to(ROOT / "Datasets"), f"--dump must be under {ROOT / 'Datasets'}"
    out.mkdir(parents=True, exist_ok=True)
    keep = lambda d: d[[c for c in ("text", "label_3", "person", "corpus") if c in d]].reset_index(drop=True)
    for name, d in [("train", tr), ("temporal", te_time), ("cross_source", te_src), ("modern", te_mod),
                    ("enron_unseen", te_enron), ("primary", te_daily)]:
        keep(d).astype({"text": str}).to_parquet(out / f"{name}.parquet")
    sys.exit(f"dumped to {out}")

# --score reloads the saved v3 and reruns the comparison table only. The baseline
# rows must be scored on the SAME te_mod as v3, and te_mod changed when the second
# mailbox arrived - an older run's numbers are not comparable.
tok, model = em.load() if "--score" in sys.argv else em.train(tr)

if "--classes" in sys.argv:      # per-class ham/spam/phishing for v3, then stop
    from sklearn.metrics import classification_report
    for name, df in [("temporal", te_time), ("cross-source", te_src),
                     ("modern 25-26", te_mod)]:
        pred = em.predict(tok, model, df.text)[0].argmax(1)
        y = df.label_3.map(em.LABELS.index).values
        present = sorted(set(y))
        print()
        print(f"--- {name}  (n={len(df):,})")
        print(pd.crosstab(pd.Series([em.LABELS[i] for i in y], name="true"),
                          pd.Series([em.LABELS[i] for i in pred], name="pred")))
        print(classification_report(y, pred, labels=present, digits=3,
                                    target_names=[em.LABELS[i] for i in present],
                                    zero_division=0))
    sys.exit()


def scores(tok, model, tag):
    from sklearn.metrics import f1_score
    row = {}
    for name, df in [("temporal", te_time), ("cross-source", te_src)]:
        p, _ = em.predict(tok, model, df.text)
        y = df.label_3.map(em.LABELS.index).values
        present = sorted(set(y))
        row[name] = f1_score(y, p.argmax(1), labels=present, average="macro",
                             zero_division=0)
    p, _ = em.predict(tok, model, te_mod.text)
    pred = p.argmax(1) == em.LABELS.index("phishing")
    ph = (te_mod.label_3 == "phishing").values
    row["modern_recall"] = 100 * pred[ph].mean()
    row["modern_fp"] = 100 * pred[~ph].mean()
    per = "  ".join(f"{who} {100 * pred[(te_mod.person == who).values].mean():.1f}%"
                    for who in sorted(te_mod.person.dropna().unique()))
    es = te_src[(te_src.corpus == "Enron") & (te_src.label_3 == "ham")]
    for name, df in [("enron-spam ham", es), ("enron unseen people", te_enron),
                     ("primary 25-26", te_daily), ("third mailbox", te_mb3)]:
        if len(df):    # every row is ham: the share called phishing is the FP rate
            per += f"  {name} {100 * (em.predict(tok, model, df.text)[0].argmax(1) == 2).mean():.1f}%"
    if "--spam" in sys.argv:
        # Spam is a LABEL, not risk (fusion reads the phishing column), so its
        # cost is a wrong SPAM badge: the false-spam rate on modern ham.
        pred_mod = p.argmax(1) == em.LABELS.index("spam")
        hamrows = (te_mod.label_3 == "ham").values
        per += "\n    SPAM  false-spam on 2025-26 ham: " + "  ".join(
            f"{who} {100 * pred_mod[hamrows & (te_mod.person == who).values].mean():.1f}%"
            for who in sorted(te_mod[hamrows].person.dropna().unique()))
        if len(te_spam):
            is_spam = em.predict(tok, model, te_spam.text)[0].argmax(1) == em.LABELS.index("spam")
            per += f"   held-out {HOLD} Gmail-spam caught {is_spam.sum()}/{len(te_spam)}"
    print(f"{tag:22s} temporal {row['temporal']:.3f}  cross-source "
          f"{row['cross-source']:.3f}  modern {row['modern_recall']:5.1f}% caught "
          f"/ {row['modern_fp']:5.1f}% FP   ham FP by person: {per}", flush=True)
    return row


print()
scores(tok, model, NAME)
BASE = ([("email_distilbert", "SHIPPED"), ("email_distilbert_v5", "v5")] if "--daily" in sys.argv else
        [("email_distilbert_v3", "v3 (two mailboxes)")] * (NAME != "email_distilbert_v3") + [
         ("email_distilbert", "SHIPPED"),
         ("email_distilbert_v2", "v2 (one mailbox)"),
         ("email_modern", "MODERN-ONLY")])
for name, tag in [b for b in BASE if b[0] != NAME]:
    em.OUT = ROOT / "models" / name
    em.load.cache_clear()
    t2, m2 = em.load()
    scores(t2, m2, tag)
