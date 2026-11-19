"""How much of Table III row 1 (phishing AUC 0.998) is "which corpus is this"?

That split's phishing rows are Nazario + Nigerian_Fraud and its ham is CEAS_08, so
class and source are the same variable. Same six non-semantic features and 5-fold
AUC as _crossuser_ladder.py, so the numbers sit on the same ladder:
0.766 = real class signal within one corpus, 0.911 = pure collection shortcut.
Scores no model and retrains nothing.

    python scratch/_confound_ladder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data                      # before torch
import numpy as np
import pandas as pd
import email_model as em
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feats(t):                    # identical to _crossuser_ladder.feats
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
    print(f"  {tag:50s} n={len(y):<6d} AUC {cross_val_score(m, X, y, cv=5, scoring='roc_auc').mean():.3f}",
          flush=True)


e = data.load_email()
ceas = e[(e.corpus == "CEAS_08") & e.date.notna()]
day8 = ceas[(ceas.date.dt.strftime("%Y-%m") == "2008-08") & (ceas.date.dt.day == 8)]
_, ph_te = em.phishing_split(e)
ham8, spam8 = day8[day8.label_3 == "ham"], day8[day8.label_3 != "ham"]

print("\nreference: 0.766 = real class signal, 0.911 = pure collection shortcut")
auc(ham8.text, spam8.text, "CEAS day-8 ham vs CEAS day-8 spam (same source)")
auc(ham8.text, ph_te.text, "CEAS day-8 ham vs phishing test half (Table III)")
for c, g in ph_te.groupby("corpus"):
    auc(ham8.text, g.text, f"  ...vs {c} only")

# The real test: swap the ham source. If the shipped model's phishing AUC holds
# against ham from corpora it never trained on, "not CEAS" is not what it learned.
from sklearn.metrics import roc_auc_score

tok, model = em.load()
pp = em.predict(tok, model, ph_te.text)[0][:, 2]
print("\nshipped model, p(phishing), phishing test half vs ham from:")
for c in ["CEAS_08", "Enron", "Ling", "SpamAssasin"]:
    h = day8[day8.label_3 == "ham"] if c == "CEAS_08" else e[(e.corpus == c) & (e.label_3 == "ham")]
    h = h.sample(min(len(h), 4000), random_state=0)
    ph = em.predict(tok, model, h.text)[0][:, 2]
    y = np.r_[np.zeros(len(ph)), np.ones(len(pp))]
    print(f"  {c:12s} ham n={len(h):<5d} model AUC {roc_auc_score(y, np.r_[ph, pp]):.3f}"
          f"   ham called phishing {(ph >= 0.5).mean():6.2%}", flush=True)
    auc(h.text, ph_te.text, f"    trivial features, {c} ham vs phishing")

# The other side: swap the PHISHING source. email_distilbert_public trained on
# Nazario + Nigerian_Fraud only and never saw phishing_pot; v4 saw pot <= 2024.
import re
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())          # same English filter as _benchmark
enf = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(re.findall(r"[a-z]+", str(t).lower()[:2000]))
pot = data.load_phishing_pot()
pot = pot[(pot.text.map(enf) >= 0.04) & pot.date.notna()]
pot = pot[pot.date.dt.year >= 2025]
print(f"\nphishing recall (argmax), own-source test half vs unseen-source phishing_pot 2025-26 (n={len(pot)}):")
for name in ["email_distilbert_public", "email_distilbert"]:
    em.OUT = em.OUT.parent / name
    em.load.cache_clear()
    tok, model = em.load()
    r = lambda t: (em.predict(tok, model, t)[0].argmax(1) == 2).mean()
    print(f"  {name:24s} Nazario+Nigerian test {r(ph_te.text):6.1%}   phishing_pot 2025-26 {r(pot.text):6.1%}",
          flush=True)

# --daily: the v5 ham sources on the same ladder, BEFORE training on them. If old
# Enron ham vs modern phishing is near 1.0 while Primary ham vs phishing is not,
# an Enron-only fix would be learning era; both eras of colleague mail must go in.
if "--daily" in sys.argv:
    enr = data.load_enron_full()
    pm = data.load_personal_mail(person="self")
    prim = pm[pm.mbox.isin({"Inbox", "Archived"}) & (pm.label_3 == "ham") & (pm.date.dt.year >= 2021)]
    print(f"\nday-to-day ham (enron n={len(enr)}, primary n={len(prim)}):")
    auc(enr.text, pot.text, "Enron colleague ham vs phishing_pot 2025-26")
    auc(prim.text, pot.text, "Primary ham vs phishing_pot 2025-26")
    auc(enr.text, prim.text, "Enron ham vs Primary ham (both ham: pure era)")
