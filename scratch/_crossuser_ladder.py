"""Is "which mailbox is this" a stronger shortcut than "is this phishing"?

Extends the 2026-08-22 collection-artifact ladder with the row it could not have:
two DIFFERENT PEOPLE'S ham, same era, same boxes. Six non-semantic features only.
Reference points from that run: 0.766 = genuine ham-vs-spam class signal within
one corpus, 0.911 = pure collection shortcut (2002 ham vs 2026 ham).

If person-vs-person lands near 0.766+, the cross-user false-positive number in
_crossuser_eval.py needs that caveat attached.
"""
import re

import data                      # before torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HAM_BOXES = {"Category promotions", "Category updates"}


def feats(t):
    t = str(t)
    w = t.split()
    return [len(t), len(w),
            sum(c.isdigit() for c in t) / max(len(t), 1),
            sum(c.isupper() for c in t) / max(len(t), 1),
            sum(ord(c) > 127 for c in t) / max(len(t), 1),
            float("http" in t)]


def auc(a, b, tag):
    X = np.array([feats(t) for t in pd.concat([a, b])])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    print(f"  {tag:44s} n={len(y):<6d} AUC {cross_val_score(m, X, y, cv=5, scoring='roc_auc').mean():.3f}",
          flush=True)


d = data.load_personal_mail(limit=6000)
ham = d[d.mbox.isin(HAM_BOXES) & (d.label_3 == "ham") & (d.date.dt.year >= 2023)]
a, b = ham[ham.person == "self"], ham[ham.person != "self"]
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())          # same English filter as _train_modern
english = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(
    re.findall(r"[a-z]+", str(t).lower()[:2000])) >= 0.04
pot = data.load_phishing_pot()
pot = pot[pot.text.map(english) & (pot.date.dt.year >= 2022)]

print("\nreference: 0.766 = real class signal, 0.911 = pure collection shortcut")
auc(a.text, b.text, "self ham vs OTHER PERSON ham (both ham, 2023+)")
auc(b.text, pot.text, "other-person ham vs phishing_pot")
auc(a.text, pot.text, "self ham vs phishing_pot (published 0.703)")
