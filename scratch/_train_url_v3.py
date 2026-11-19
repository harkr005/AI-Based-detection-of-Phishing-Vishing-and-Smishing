"""URL model v3: judge the REGISTRABLE DOMAIN, not the host's shape (2026-10-11).

WHY. Two findings, ours and the literature's, say the same thing:
  - PhiUSIIL's benign class is 100% bare https://www.domain.tld (independent
    audit, github.com/mutwassim/Robustness-or-Artifact-). Any label in front of
    the domain therefore MEANS phishing to a model trained on it. Measured here:
    19/38 of Mohak's genuine Gmail first links scored >= .70 (myaccount.google.com .87).
  - Subdomain depth is the strongest marker of free HOSTING, not of phishing
    (Nowak, Mazurczyk & Syta, IACR ePrint 2026/1501, eta .54 vs .31).
v2 (more benign shapes) and the Tranco "known domain" rule both failed the pass
bar; see scratch/_train_url_v2.py and _url_known_eval.py.

RECIPE. Split every host with the Public Suffix List (ICANN + PRIVATE sections,
Datasets/URL/public_suffix_list.dat, version in its header):
    t2.service.icicisecurities.com -> reg icicisecurities.com, sub t2.service
    paypal-login.github.io         -> reg paypal-login.github.io (github.io is a
                                      PRIVATE suffix: anyone can get one, so the
                                      customer's label IS the registrable domain)
url_model's 15 lexical features are computed on REG only. The part in front
contributes CONTENT, never shape: lure words in it, and an embedded IP
(vjockfvt.20-218-142-124.cloud-xip.com). No depth, length or digit count of
the sub - that is the PhiUSIIL artifact. Same PhiUSIIL training rows, same
XGBoost settings as url_model.train. Nothing from private mail trains it.

TESTS (unchanged from v2 / known-domain, fixed before results):
  internal       PhiUSIIL split DISJOINT by registrable domain (GroupShuffleSplit)
  mail ham       every link host in self + user2 non-Spam mail -> false alarms
  pot            phishing_pot hosts minus benign-mail hosts and Tranco top-50k
                 entries -> catch (pessimistic: footer links to real brands stay)
  openphish      OpenPhish vs Tranco top-50k. sms: worst link per SMS.
  gmail scans    first link of each Gmail scan in Postgres.
Pass bar: mail false alarms fall a lot; pot / openphish / sms catch drop <= ~2.

    python -m scratch._train_url_v3        -> models/url_xgb_v3.json
"""
import re
import sys

import data                      # before torch / xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import thread
import url_model as um

PSL_FILE = um.PSL_FILE
# --v4: v3 failed (OpenPhish recall 80.3 -> 61.2): the sub's SHAPE carries real
# phishing signal (secure-update-x83k.attacker.com) as well as the artifact.
# v4 keeps it as separate sub_* shape features AND gives the model benign hosts
# that have a sub - self's mail link hosts, weight 10 as in v2, not tuned. Mail
# false alarms are then measured on user2-only hosts (never trained on).
V4 = "--v4" in sys.argv
OUT = um.MODEL.with_name("url_xgb_v4.json" if V4 else "url_xgb_v3.json")


# One splitter and one v4 featurizer, in url_model, so the email path scores
# exactly what this script trains (moved there 2026-10-11 when v4 shipped).
split = um.split


FEATURES = um.MAIL_FEATURES if V4 else um.FEATURES + ["sub_lure", "sub_ip"]


def features(url):
    if V4:
        return um.mail_features(url)
    sub, reg = split(um._host(url))
    f = um.features("http://" + reg)
    f["sub_lure"] = sum(w in sub for w in um.LURE)
    f["sub_ip"] = int(bool(um.EMBEDDED_IP.search(sub)))
    return f


def featurize(urls):
    return pd.DataFrame([features(u) for u in urls], columns=FEATURES)


# --- PSL self-check: the cases this whole model rests on -------------------
assert split("t2.service.icicisecurities.com") == ("t2.service", "icicisecurities.com")
assert split("www.amazon.in") == ("", "amazon.in")
assert split("lnk.custcomm.icici.bank.in") == ("lnk.custcomm", "icici.bank.in")
assert split("greekturtle-prod.s3.amazonaws.com")[1] == "greekturtle-prod.s3.amazonaws.com"
assert split("paypal-login.github.io") == ("", "paypal-login.github.io")
assert split("hdfcbank.com.evil.tk") == ("hdfcbank.com", "evil.tk")
assert split("github.io") == ("", "github.io") and split("192.168.1.4") == ("", "192.168.1.4")
assert features("https://vjockfvt.20-218-142-124.cloud-xip.com")["sub_ip"] == 1
assert features("https://t2.service.icicisecurities.com/x")["sub_ip"] == 0
assert features("http://kyc-verify.paytm.evil.tk")["sub_lure"] >= 2

# --- external tests ----------------------------------------------------------
def hosts(df):
    out = set()
    for lh, txt in zip(df.link_hosts, df.text):
        out.update(h for h in lh if h and h != "?")
        out.update(um._host(u) for u in thread.find_urls(txt))
    return {h[4:] if h.startswith("www.") else h for h in out if h and "." in h}


mail = data.load_personal_mail(limit=6000)
ham = mail[mail.label_3 == "ham"]
ham_h, self_h = hosts(ham), hosts(ham[ham.person == "self"])
del mail, ham
pot = data.load_phishing_pot()
pot_h = hosts(pot)
del pot
feed = data.load_url_feed()
tranco = {um._host(u).removeprefix("www.") for u in feed.url[feed.label == 0]}
test_ham = sorted((ham_h - self_h if V4 else ham_h) - pot_h)   # v4: user2-only
train_benign = sorted(self_h - pot_h) if V4 else []
test_pot = sorted(pot_h - ham_h - tranco)
print(f"benign mail train hosts {len(train_benign):,}   mail ham test hosts {len(test_ham):,}   pot test hosts {len(test_pot):,}", flush=True)


phi = data.load_urls()
urls = list(phi.url) + train_benign
X = featurize(urls)
y = np.r_[phi.label.values, np.zeros(len(train_benign), int)]
w = np.r_[np.ones(len(phi)), np.full(len(train_benign), 10.0)]
groups = [split(um._host(u))[1] for u in urls]
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.1,
              eval_metric="logloss", n_jobs=-1)        # url_model.train's settings
tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=0).split(X, y, groups))
held = xgb.XGBClassifier(**PARAMS).fit(X.iloc[tr], y[tr], sample_weight=w[tr])
ph = held.predict_proba(X.iloc[te])[:, 1]
print(f"internal, registrable-domain-disjoint: AUC {roc_auc_score(y[te], ph):.3f}  "
      f"recall {100 * (ph[y[te] == 1] >= .5).mean():.1f}%  FA {100 * (ph[y[te] == 0] >= .5).mean():.1f}%",
      flush=True)
m = xgb.XGBClassifier(**PARAMS).fit(X, y, sample_weight=w)
m.save_model(OUT)
print(f"saved {OUT.name}   PSL {PSL_FILE.read_text(encoding='utf-8').splitlines()[7][3:]}", flush=True)


sms = data.load_sms()
sms_urls = [thread.find_urls(t) for t in sms.text]
has = np.array([bool(u) for u in sms_urls])
smish = (sms.label_3[has] == "smishing").values
gmail = []
try:
    import store
    with store.connect() as c:
        gmail = [r[0] for r in c.execute(
            "SELECT detail->'links'->0->>'url' FROM scans WHERE source = 'gmail' "
            "AND jsonb_array_length(coalesce(detail->'links', '[]')) > 0")]
except Exception as e:
    print(f"(gmail scans skipped: {e})")

v1 = um.load()
MODELS = [("v1 now", lambda u: v1.predict_proba(um.featurize(u))[:, 1]),
          ("v4" if V4 else "v3 PSL", lambda u: m.predict_proba(featurize(u))[:, 1])]


def worst(score, lists):
    flat = [u for us in lists for u in us]
    s, out, i = score(flat), [], 0
    for us in lists:
        out.append(s[i:i + len(us)].max())
        i += len(us)
    return np.array(out)


print()
print("model    | mail ham FA >=.5  >=.7 | pot catch >=.5  >=.7 | openphish AUC  recall "
      "| sms smish caught / other FA >=.5 | gmail scans >=.7 | mail-vs-pot AUC")
for name, score in MODELS:
    h, pt, f = score(test_ham), score(test_pot), score(list(feed.url))
    s = worst(score, [u for u, k in zip(sms_urls, has) if k])
    g = score(gmail) if gmail else np.array([])
    print(f"{name:8s} | {100 * (h >= .5).mean():11.1f}% {100 * (h >= .7).mean():5.1f}% | "
          f"{100 * (pt >= .5).mean():10.1f}% {100 * (pt >= .7).mean():5.1f}% | "
          f"{roc_auc_score(feed.label, f):13.3f} {100 * (f[feed.label.values == 1] >= .5).mean():6.1f}% | "
          f"{100 * (s[smish] >= .5).mean():10.1f}% / {100 * (s[~smish] >= .5).mean():.1f}% "
          f"| {int((g >= .7).sum())}/{len(g)} | "
          # The realistic separation: real mail links vs phishing-mail links,
          # every threshold at once. Tranco (openphish column) is homepage-only,
          # the same artifact PhiUSIIL has, so it flatters v1.
          f"{roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(pt))], np.r_[h, pt]):.3f}", flush=True)
