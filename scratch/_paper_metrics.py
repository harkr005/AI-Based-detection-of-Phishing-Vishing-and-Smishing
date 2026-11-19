"""Fresh numbers for the paper, measured on the SHIPPED weights. Retrains nothing.

Every split is rebuilt exactly as the script that trained the shipped model built
it (sms_model.__main__, scratch/_train_unified.py for email v4, url_model.train),
so each test set is data the shipped model never saw. No k-fold: the project
never ran one, and cross-validating a model trained with groups/time splits
would leak the template-twins and future mail those splits exist to keep out.

"malicious" = anything not ham (SMS: spam+smishing, email: spam+phishing), and
"scam" = the scam class alone (smishing / phishing). Both are reported because
they answer different questions; the 3-class macro row is the headline metric.

Global attributions: URL uses xgboost pred_contribs, which is exact TreeSHAP;
SMS is linear, so its class coefficients are the exact attribution. Prints
features and n-grams only, never message text (CLAUDE.md: private corpora).

    python scratch/_paper_metrics.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data                       # before torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit


def row(name, y, pred, labels, scam, prob=None):
    """3-class macro + two binary views. y/pred are label indices."""
    present = sorted(set(y))
    P, R, F, _ = precision_recall_fscore_support(y, pred, labels=present,
                                                 average="macro", zero_division=0)
    out = {"test set": name, "n": len(y), "acc": accuracy_score(y, pred),
           "macro P": P, "macro R": R, "macro F1": F}
    for tag, pos in [("mal", lambda v: v != 0), ("scam", lambda v: v == scam)]:
        yt, pt = pos(np.asarray(y)), pos(np.asarray(pred))
        if yt.all() or not yt.any():          # one-class slice: only recall means anything
            out[f"{tag} R"] = pt[yt].mean() if yt.any() else np.nan
            continue
        p, r, f, _ = precision_recall_fscore_support(yt, pt, average="binary", zero_division=0)
        out[f"{tag} acc"], out[f"{tag} P"], out[f"{tag} R"], out[f"{tag} F1"] = \
            accuracy_score(yt, pt), p, r, f
        if prob is not None and tag == "scam":
            out["scam AUC"] = roc_auc_score(yt, prob[:, scam])
    return out


def show(title, rows):
    print(f"\n=== {title}")
    print(pd.DataFrame(rows).set_index("test set").round(3).to_string())


# --- SMS: shipped char-LR, sms_model.__main__'s split ----------------------
import sms_model
s = data.load_sms()
held = s[s.source == "Dataset_5971"]
pool = s[s.source != "Dataset_5971"]
_, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
               .split(pool, groups=pool.group))
te = pool.iloc[i_te]
L = sms_model.LABELS
rows = []
for name, df in [("in-source (held-out groups)", te), ("cross-source (Dataset_5971)", held)]:
    p = sms_model.score(df.text.fillna(""))
    rows.append(row(name, df.label_3.map(L.index).values, p.argmax(1), L, L.index("smishing"), p))
show(f"SMS  (classes {L})", rows)

clf, vec = sms_model.load().steps[1][1], sms_model.load().steps[0][1]
k = L.index("smishing")
names = np.array(vec.get_feature_names_out())
w = clf.coef_[k] - clf.coef_[[i for i in range(len(L)) if i != k]].max(0)
print("top smishing char n-grams (coef margin vs best rival class):",
      ", ".join(f"{names[i]!r} {w[i]:.2f}" for i in np.argsort(-w)[:15]))

# --- URL: shipped XGBoost, OpenPhish+Tranco holdout -------------------------
import url_model
import xgboost as xgb
feed = data.load_url_feed()
X = url_model.featurize(feed.url)
pu = url_model.load().predict_proba(X)[:, 1]
y = feed.label.values
p_, r_, f_, _ = precision_recall_fscore_support(y, pu > .5, average="binary")
print(f"\n=== URL  holdout n={len(y):,} ({y.sum():,} phishing)  AUC {roc_auc_score(y, pu):.3f}  "
      f"acc {accuracy_score(y, pu > .5):.3f}  P {p_:.3f}  R {r_:.3f}  F1 {f_:.3f}")
contrib = url_model.load().get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
imp = pd.Series(np.abs(contrib).mean(0), index=url_model.FEATURES).sort_values(ascending=False)
sign = pd.Series(contrib[y == 1].mean(0), index=url_model.FEATURES)
print("mean |TreeSHAP| on holdout (sign = mean push on phishing rows):")
for f in imp.index[:8]:
    print(f"  {f:14} {imp[f]:.3f}  ({sign[f]:+.3f})  {url_model.PHRASE[f]}")

# --- Email: shipped model (models/email_distilbert), _train_unified.py's test sets --
import email_model as em
e = data.load_email()
aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
_, ph_te = em.phishing_split(e)
te_time = pd.concat([aug[aug.date.dt.day == 8], ph_te])
te_src = e[e.corpus.isin(["Enron", "Ling", "SpamAssasin"])]
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())
english = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(
    re.findall(r"[a-z]+", str(t).lower()[:2000])) >= 0.04
pot = data.load_phishing_pot()
pot = pot[pot.text.map(english) & pot.date.dt.year.between(2022, 2026)].copy()
pot["label_3"] = "phishing"
ham = data.load_personal_mail(limit=6000)
ham = ham[ham.mbox.isin({"Category promotions", "Category updates"}) & (ham.label_3 == "ham")
          & ham.date.dt.year.between(2021, 2026)].copy()
modern = pd.concat([pot, ham], ignore_index=True)
te_mod = modern[modern.date.dt.year >= 2025]

L = em.LABELS
rows, persuade = [], []
for name, df in [("temporal (CEAS day 8 + later phishing)", te_time),
                 ("cross-source (Enron/Ling/SA, ham+spam)", te_src),
                 ("modern 2025-26 (phishing_pot + inbox ham)", te_mod)]:
    p, q = em.score(df.text.tolist())
    y = df.label_3.map(L.index).values
    rows.append(row(name, y, p.argmax(1), L, L.index("phishing"), p))
    persuade.append(q[y == L.index("phishing")])
show(f"EMAIL shipped  (classes {L})", rows)
q = np.vstack([x for x in persuade if len(x)])
print("persuasion head on true phishing (share firing >= 0.5):",
      ", ".join(f"{n} {v:.1%}" for n, v in zip(em.PRINCIPLES, (q >= .5).mean(0))))
