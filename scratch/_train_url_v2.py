"""URL model v2: real mail links as the benign side (2026-10-11).

WHY. url_xgb learned benign = PhiUSIIL's legit URLs, which are bare homepages
(www.brand.com). Real mail never links to those: it links through tracking
hosts (t2.service.icicisecurities.com, links.mail.crunchyroll.com). Measured on
Mohak's own Gmail scans: 19 of 38 first links scored >= 0.70, myaccount.google.com
0.87. The model is not wrong about phishing; it has never seen a normal mail link.

RECIPE. url_model's FEATURES unchanged (so the Kotlin port needs new weights
only, no code), PhiUSIIL unchanged, plus every host a link in self's non-Spam
mail points at (href hosts + visible-text URL hosts), labelled benign, weight
--w (default 10: ~thousands of hosts beside ~134k PhiUSIIL benign rows).
ponytail: fixed weight, not tuned - tuning it on the tests below would be
fitting the test.

TESTS, fixed before any result was seen:
  user2-only   hosts in user2's non-Spam mail and NOT in self's: a person
                 the model never saw -> FALSE-ALARM rate.
  pot            phishing_pot link hosts (modern phishing MAIL), minus any host
                 that also occurs in benign mail or Tranco top-50k (a phishing
                 mail's footer link to microsoft.com is not a phishing host).
                 -> CATCH rate. Never trained on.
  openphish      the existing holdout: OpenPhish vs Tranco top-50k, AUC + recall.
  sms            hosts of URLs in smishing vs ham SMS (the phone runs this model).
  gmail scans    first link of every Gmail scan in Postgres. NOT held out: self's
                 mail is the training data. Shown as before/after only.
Hosts in both benign mail and pot are dropped from training (shared shorteners,
ESPs, docs.google.com - ambiguous by nature). Pass bar: user2-only false
alarms fall a lot; pot / openphish / sms catch drop <= ~2 points.

PRIVATE: benign hosts come from private mailboxes. Weights derived from them are
private like sms_charlr.bin. The log prints counts and rates, never hosts.

    python -m scratch._train_url_v2            # -> models/url_xgb_v2.json
"""
import sys

import data                      # before torch / xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import thread
import url_model as um

OUT = um.MODEL.with_name("url_xgb_v2.json")
W = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 10.0


def hosts(df):
    """Every link host in these mail rows: hrefs (link_hosts) + visible URLs."""
    out = set()
    for lh, txt in zip(df.link_hosts, df.text):
        out.update(h for h in lh if h and h != "?")
        out.update(um._host(u) for u in thread.find_urls(txt))
    return {h[4:] if h.startswith("www.") else h for h in out if h and "." in h}


mail = data.load_personal_mail(limit=6000)
ham = mail[mail.label_3 == "ham"]
self_h = hosts(ham[ham.person == "self"])
harsh_h = hosts(ham[ham.person == "user2"])
del mail, ham
pot = data.load_phishing_pot()
pot_h = hosts(pot)
del pot
feed = data.load_url_feed()
tranco = {um._host(u).removeprefix("www.") for u in feed.url[feed.label == 0]}

train_benign = sorted(self_h - pot_h)
test_harsh = sorted(harsh_h - self_h - pot_h)
test_pot = sorted(pot_h - self_h - harsh_h - tranco)
print(f"benign mail hosts: self {len(self_h):,} (train {len(train_benign):,} after "
      f"dropping {len(self_h & pot_h):,} also in phishing)  user2-only test {len(test_harsh):,}"
      f"   pot test hosts {len(test_pot):,}   weight {W:g}", flush=True)

phi = data.load_urls()
X = pd.concat([um.featurize(phi.url), um.featurize(train_benign)], ignore_index=True)
y = np.r_[phi.label.values, np.zeros(len(train_benign), int)]
w = np.r_[np.ones(len(phi)), np.full(len(train_benign), W)]
m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                      eval_metric="logloss", n_jobs=-1)   # url_model.train's settings
m.fit(X, y, sample_weight=w)
m.save_model(OUT)
print(f"saved {OUT.name}", flush=True)

old = xgb.XGBClassifier()
old.load_model(um.MODEL)

# SMS: one row per message with a URL, scored on its worst link, like api._links.
sms = data.load_sms()
sms_urls = [thread.find_urls(t) for t in sms.text]
has = np.array([bool(u) for u in sms_urls])
gmail = []
try:
    import store
    with store.connect() as c:
        gmail = [r[0] for r in c.execute(
            "SELECT detail->'links'->0->>'url' FROM scans WHERE source = 'gmail' "
            "AND jsonb_array_length(coalesce(detail->'links', '[]')) > 0")]
except Exception as e:             # Postgres down: the row is optional
    print(f"(gmail scans skipped: {e})")


def p(model, urls):
    return model.predict_proba(um.featurize(urls))[:, 1] if len(urls) else np.array([])


def worst(model, lists):
    flat = [u for us in lists for u in us]
    s, out, i = p(model, flat), [], 0
    for us in lists:
        out.append(s[i:i + len(us)].max())
        i += len(us)
    return np.array(out)


print()
print(f"{'model':8s} | user2-only FA >=.5  >=.7 | pot catch >=.5  >=.7 | "
      f"openphish AUC  recall | sms smish caught / other FA >=.5 | gmail scans >=.7")
for name, mod in [("v1 now", old), ("v2", m)]:
    h, pt = p(mod, test_harsh), p(mod, test_pot)
    f = p(mod, list(feed.url))
    s = worst(mod, [u for u, k in zip(sms_urls, has) if k])
    smish = (sms.label_3[has] == "smishing").values
    g = p(mod, gmail)
    print(f"{name:8s} | {100 * (h >= .5).mean():15.1f}% {100 * (h >= .7).mean():5.1f}% | "
          f"{100 * (pt >= .5).mean():10.1f}% {100 * (pt >= .7).mean():5.1f}% | "
          f"{roc_auc_score(feed.label, f):13.3f} {100 * ((f >= .5) & (feed.label == 1)).sum() / (feed.label == 1).sum():6.1f}% | "
          f"{100 * (s[smish] >= .5).mean():10.1f}% / {100 * (s[~smish] >= .5).mean():.1f}% "
          f"| {int((g >= .7).sum())}/{len(g)}", flush=True)
