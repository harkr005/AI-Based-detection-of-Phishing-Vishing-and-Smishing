"""Are the explanations faithful? A deletion test on the shipped weights.

An explanation is faithful when the evidence it names is what the model actually
used. So take every held-out message the model flags, delete exactly what the
explanation named, and see how far the score falls - against deleting the same
number of random words (or features). Named evidence that moves the score no more
than random evidence is decoration, however readable the sentence around it.

  SMS  sms_model.explain's words, on the two test sets sms_model.__main__ holds out
       (never trained on). Deleted from the normalised text, every occurrence.
  URL  url_model's top-3 positive TreeSHAP features on the OpenPhish holdout.
       "Deleting" a feature sets it to the median of PhiUSIIL's benign URLs.
       Features are correlated (n_digits / digit_ratio / is_ip), so the URL
       numbers understate what a single named feature carries.

Email has no word-level explanation to test (the persuasion head names tactics,
not evidence), so it is not here. Retrains nothing, prints no message text.

    python scratch/_explain_faithfulness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data                       # before torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

import adversarial
import sms_model
import url_model

SEEDS = 5
STRIP = "\"'.,!?:;"


def report(name, base, named, rand, flip_named, flip_rand):
    print(f"  {name:32} n={len(base):5}  score {base.mean():.3f} -> "
          f"named {named.mean():.3f} (drop {(base - named).mean():.3f}, flips {flip_named.mean():5.1%})  "
          f"random {rand.mean():.3f} (drop {(base - rand).mean():.3f}, flips {flip_rand.mean():5.1%})")


# --- SMS -------------------------------------------------------------------
s = data.load_sms()
held = s[s.source == "Dataset_5971"]
pool = s[s.source != "Dataset_5971"]
_, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
               .split(pool, groups=pool.group))
K = sms_model.LABELS.index("smishing")
rng = np.random.default_rng(0)

print("SMS - delete the words explain() named vs the same number of random words")
for name, df in [("in-source (held-out groups)", pool.iloc[i_te]),
                 ("cross-source (Dataset_5971)", held)]:
    base_t, named_t, rand_t, ks = [], [], [], []
    for text in df.text.fillna(""):
        lab, _, _, terms = sms_model.explain(text)
        if lab != "smishing" or not terms:
            continue
        words = adversarial.normalize(text).split()
        hit = [i for i, w in enumerate(words) if w.strip(STRIP) in set(terms)]
        if not hit or len(hit) == len(words):
            continue
        keep = lambda drop: " ".join(w for i, w in enumerate(words) if i not in drop)
        base_t.append(" ".join(words))
        named_t.append(keep(set(hit)))
        rand_t.append([keep(set(rng.choice(len(words), len(hit), replace=False)))
                       for _ in range(SEEDS)])
        ks.append(len(hit))
    base = sms_model.score(base_t)
    named = sms_model.score(named_t)
    rand = sms_model.score([t for r in rand_t for t in r]).reshape(len(rand_t), SEEDS, -1)
    report(name, base[:, K], named[:, K], rand[:, :, K].mean(1),
           named.argmax(1) != K, (rand.argmax(2) != K).mean(1))
    print(f"  {'':32} words deleted per message: mean {np.mean(ks):.1f}, "
          f"message length mean {np.mean([len(t.split()) for t in base_t]):.1f}")

# --- URL -------------------------------------------------------------------
print("\nURL - reset the top-3 TreeSHAP features to the benign median vs 3 random features")
feed = data.load_url_feed()
ph = feed[feed.label == 1].url.tolist()
X = url_model.featurize(ph)
booster = url_model.load()
p0 = booster.predict_proba(X)[:, 1]
flag = (p0 >= .5) & ~np.array([url_model.verified(u) for u in ph])
X, p0 = X[flag].reset_index(drop=True), p0[flag]

train = data.load_urls()
benign = url_model.featurize(train[train.label == 0].url.sample(20_000, random_state=0))
median = benign.median()

contrib = booster.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
F = len(url_model.FEATURES)
named_X, rand_X = X.copy(), []
for i in range(len(X)):
    top = [j for j in np.argsort(-contrib[i])[:3] if contrib[i, j] > 0]
    named_X.iloc[i, top] = median.iloc[top].values
for _ in range(SEEDS):
    R = X.copy()
    for i in range(len(X)):
        cols = rng.choice(F, 3, replace=False)
        R.iloc[i, cols] = median.iloc[cols].values
    rand_X.append(booster.predict_proba(R)[:, 1])
pn = booster.predict_proba(named_X)[:, 1]
pr = np.vstack(rand_X)
report("OpenPhish holdout, flagged", p0, pn, pr.mean(0), pn < .5, (pr < .5).mean(0))
top_named = pd.Series([url_model.FEATURES[j] for i in range(len(X))
                       for j in np.argsort(-contrib[i])[:1]]).value_counts(normalize=True)
print("  most-named top feature:", ", ".join(f"{k} {v:.0%}" for k, v in top_named.head(5).items()))
