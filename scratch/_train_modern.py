"""Modern email phishing detector: 2022-2026 on BOTH sides.

The first real temporal split in this project with modern data either side of the
line. CEAS's split is next-day; this one is next-campaign.

  positives   phishing_pot English   <=2024 train / 2025-26 held out
  negatives   personal mail, PROMOTIONS + UPDATES only, same split

Promotions and Updates only, deliberately: that is machine-generated commercial
mail - the genre phishing imitates - and it keeps personal correspondence out of
a training set entirely. Inbox and Social are not read by this script.

Justified by the artifact ladder (2026-08-22, six trivial non-semantic features):
    ham vs spam within one corpus ....... 0.766   genuine class signal
    personal ham vs English pot ......... 0.703   below it, so trainable
    2002 ham vs personal ham (both ham).. 0.911   pure collection shortcut
Any pairing above 0.766 would train a collection detector wearing a phishing
label. This pairing is the only one measured below it.

THE WEIGHTS THIS WRITES ARE DERIVED FROM PRIVATE MAIL. Do not publish them, do
not push them, and do not quote example inputs from the negative class.
"""
import pathlib
import re

import data                      # before torch
import numpy as np
import pandas as pd
import adversarial
import email_model as em

HAM_BOXES = {"Category promotions", "Category updates"}
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())


def english(t):
    w = re.findall(r"[a-z]+", str(t).lower()[:2000])
    return sum(x in EN for x in w) / max(len(w), 1) >= 0.04


pot = data.load_phishing_pot()
pot = pot[pot.text.map(english) & pot.date.dt.year.between(2022, 2026)].copy()
pot["label_3"] = "phishing"

ham = data.load_personal_mail(limit=6000)
ham = ham[ham.mbox.isin(HAM_BOXES) & (ham.label_3 == "ham")
          & ham.date.dt.year.between(2021, 2026)].copy()

both = pd.concat([pot, ham], ignore_index=True)
yr = both.date.dt.year
tr, te = both[yr <= 2024].copy(), both[yr >= 2025].copy()
tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)   # §9.9

print("train      ", len(tr), dict(tr.label_3.value_counts()))
print("held out   ", len(te), dict(te.label_3.value_counts()))
print("held-out years", dict(te.date.dt.year.value_counts().sort_index()), flush=True)

em.OUT = pathlib.Path(data.__file__).resolve().parent / "models" / "email_modern"
tok, model = em.train(tr)


def catch(tok, model, df):
    """-> (phishing recall, ham false-positive rate) on one slice."""
    p, _ = em.predict(tok, model, df.text)
    pred = p.argmax(1) == em.LABELS.index("phishing")
    ph, hm = df.label_3 == "phishing", df.label_3 == "ham"
    return 100 * pred[ph.values].mean(), 100 * pred[hm.values].mean()


r, f = catch(tok, model, te)
print(f"\nMODERN MODEL on held-out 2025-26:  phishing caught {r:.1f}%   "
      f"ham false-positive {f:.1f}%")

em.OUT = pathlib.Path(data.__file__).resolve().parent / "models" / "email_distilbert"
em.load.cache_clear()
tok2, model2 = em.load()
r2, f2 = catch(tok2, model2, te)
print(f"SHIPPED model  on held-out 2025-26:  phishing caught {r2:.1f}%   "
      f"ham false-positive {f2:.1f}%")

for yr_ in (2025, 2026):
    g = te[te.date.dt.year == yr_]
    a, b = catch(tok, model, g)
    c, d_ = catch(tok2, model2, g)
    print(f"  {yr_}  n={len(g):<5} modern {a:5.1f}% caught / {b:4.1f}% FP    "
          f"shipped {c:5.1f}% caught / {d_:4.1f}% FP")
