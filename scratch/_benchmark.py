"""Classical baselines vs DistilBERT on the EXACT same email splits.

Same training rows (including the §9.9 adversarial twins), same test sets, same
metric. Adds one test the transformer was also measured on: phishing_pot 2022-26,
which nothing in the training set resembles.
"""
import re
import time

import data                      # before torch
import numpy as np
import pandas as pd
import adversarial
import email_model as em
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

e = data.load_email()
aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
ph_tr, ph_te = em.phishing_split(e)

tr = pd.concat([aug[aug.date.dt.day <= 7], ph_tr])
tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)   # same as DistilBERT
te_time = pd.concat([aug[aug.date.dt.day == 8], ph_te])
te_src = e[e.corpus.isin(["Enron", "Ling", "SpamAssasin"])]

pot = data.load_phishing_pot()
EN = set("the and you your account for that with this have from will not are was please click here".split())
enf = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(re.findall(r"[a-z]+", t.lower()[:2000]))
pot_en = pot[pot.text.map(enf) >= 0.04]

y = lambda d: d.label_3.map(em.LABELS.index).values
ytr = y(tr)
print(f"train {len(tr):,}  temporal {len(te_time):,}  cross-source {len(te_src):,}  "
      f"pot(EN) {len(pot_en):,}\n")

MODELS = {
    "Naive Bayes (TF-IDF)": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Linear SVM": LinearSVC(),
    "Random Forest (300)": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                  min_samples_leaf=2, random_state=0),
}

hdr = f"{'model':24s} {'train s':>8s} {'temporal':>9s} {'x-source':>9s} {'spam R':>7s} {'pot 2022-26':>12s}"
print(hdr)
print("-" * len(hdr))
for name, clf in MODELS.items():
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                          max_features=200_000, strip_accents="unicode")
    pipe = make_pipeline(vec, clf)
    t0 = time.time()
    pipe.fit(tr.text.fillna(""), ytr)
    secs = time.time() - t0

    p_time = pipe.predict(te_time.text.fillna(""))
    p_src = pipe.predict(te_src.text.fillna(""))
    p_pot = pipe.predict(pot_en.text.fillna(""))
    f_time = f1_score(y(te_time), p_time, average="macro", zero_division=0)
    f_src = f1_score(y(te_src), p_src, labels=[0, 1], average="macro", zero_division=0)
    spam_r = recall_score(y(te_src), p_src, labels=[1], average="macro", zero_division=0)
    caught = 100 * (p_pot == 2).mean()
    print(f"{name:24s} {secs:8.1f} {f_time:9.3f} {f_src:9.3f} {spam_r:7.3f} {caught:11.1f}%")

print(f"\n{'DistilBERT (shipped)':24s} {'~900':>8s} {0.877:9.3f} {0.810:9.3f} {0.605:7.3f} {47.1:11.1f}%")
