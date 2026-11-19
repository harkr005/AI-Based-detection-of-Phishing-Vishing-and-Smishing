"""Does the modern email model's 2.6% false-positive rate survive a DIFFERENT
person's mailbox?

Every modern number in this project so far was scored against ONE inbox, held out
by TIME only. A second Google Takeout (promotions + updates + spam) makes the ham
side holdable by PERSON, which is the only way to tell a real property from one
mailbox memorised. Nothing here is trained; three existing checkpoints are scored.

PRIVATE DATA. Numbers may go in the report, message text may not.
"""
import pathlib

import data                      # before torch
import pandas as pd
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
HAM_BOXES = {"Category promotions", "Category updates"}   # same boxes as training
MODELS = [("SHIPPED", "email_distilbert"), ("MODERN", "email_modern"),
          ("UNIFIED v2", "email_distilbert_v2")]

d = data.load_personal_mail(limit=6000)
ham = d[d.mbox.isin(HAM_BOXES) & (d.label_3 == "ham") & d.date.dt.year.between(2021, 2026)]
new = sorted(set(d.person) - {"self"})
print(d.groupby(["person", "mbox"]).size(), "\n", flush=True)

slices = [("self ham 2025-26 (the published 2.6%)", ham[(ham.person == "self")
                                                        & (ham.date.dt.year >= 2025)])]
for p in new:
    g = ham[ham.person == p]
    slices += [(f"{p} ham ALL (unseen person)", g),
               (f"{p} ham 2025-26", g[g.date.dt.year >= 2025]),
               (f"{p} ham <=2024", g[g.date.dt.year <= 2024])]
    sp = d[(d.person == p) & (d.label_3 == "spam")]
    if len(sp):
        slices.append((f"{p} SPAM box (Gmail's call, n={len(sp)})", sp))

out = {}
for tag, name in MODELS:
    em.OUT = ROOT / "models" / name
    em.load.cache_clear()
    tok, model = em.load()
    for label, df in slices:
        if not len(df):
            continue
        p, _ = em.predict(tok, model, df.text)
        pred = p.argmax(1)
        out.setdefault(label, {})[tag] = (
            100 * (pred == em.LABELS.index("phishing")).mean(),
            100 * (pred == em.LABELS.index("spam")).mean(), len(df))

print(f"{'slice':44s} {'n':>6}" + "".join(f"{t:>14s}" for t, _ in MODELS))
print(f"{'':44s} {'':>6}" + "".join(f"{'phish%/spam%':>14s}" for _ in MODELS))
for label, row in out.items():
    n = list(row.values())[0][2]
    print(f"{label:44s} {n:6d}" + "".join(
        f"{row[t][0]:7.1f}/{row[t][1]:5.1f}" for t, _ in MODELS))
