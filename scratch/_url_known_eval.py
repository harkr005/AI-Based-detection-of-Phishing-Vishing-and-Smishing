"""Method 1 for the link false alarms: judge a WELL-KNOWN domain, not its
tracking subdomain (2026-10-11). Nothing is trained.

v2 (scratch/_train_url_v2.py) failed: host SHAPE cannot separate a tracking
host (t2.service.icicisecurities.com) from a phish of the same shape. So this
asks WHOSE domain it is instead. Rule, applied before the unchanged url_xgb:
  walk the host's suffixes, longest first; the first one in Tranco top-50k is
  the owner. If the host IS that entry, nothing changes. If it is a subdomain
  of it, score the owner alone - unless the owner is SHARED HOSTING.
Shared hosting (weebly.com, s3.amazonaws.com...) is not hand-typed: a top-50k
domain counts as abused when PhiUSIIL - the model's own training data, none of
the tests below - has a phishing URL on a subdomain of it.

TESTS, same as v2's, fixed before any result:
  mail ham       every link host in self's + user2's non-Spam mail. Nothing
                 trains here, so ALL of it is a fair false-alarm test.
  pot            phishing_pot hosts minus benign-mail hosts and exact Tranco
                 entries -> CATCH. Pessimistic: a phishing mail's footer link to
                 a real brand's subdomain (email.microsoft.com) stays in and now
                 counts as "missed".
  openphish      OpenPhish vs Tranco top-50k. sms: worst link per SMS.
  gmail scans    first link of each Gmail scan in Postgres.
Pass bar (v2's): mail false alarms fall a lot; catch drop <= ~2 points.

    python -m scratch._url_known_eval      (PRIVATE hosts: log prints rates only)
"""
import data                      # before torch / xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import thread
import url_model as um

TOP = 50_000
tranco = pd.read_csv(data.DATA / "URL" / "tranco_N2KLW.csv", names=["rank", "d"], nrows=TOP)
KNOWN = set(tranco.d.str.lower())


def owner(host):
    """-> the longest suffix of host (not host itself) in KNOWN, else None."""
    parts = host.split(".")
    return next((".".join(parts[i:]) for i in range(1, len(parts) - 1)
                 if ".".join(parts[i:]) in KNOWN), None)


def strip(h):
    return h[4:] if h.startswith("www.") else h


phi = data.load_urls()
ABUSED = {o for o in (owner(strip(um._host(u))) for u in phi.url[phi.label == 1]) if o}
print(f"top-{TOP:,} domains {len(KNOWN):,}; abused as shared hosting (PhiUSIIL phish "
      f"on a subdomain) {len(ABUSED):,}", flush=True)


def collapse(url):
    host = strip(um._host(url))
    o = None if host in KNOWN else owner(host)
    return f"https://{o}" if o and o not in ABUSED else url


def hosts(df):
    out = set()
    for lh, txt in zip(df.link_hosts, df.text):
        out.update(h for h in lh if h and h != "?")
        out.update(um._host(u) for u in thread.find_urls(txt))
    return {strip(h) for h in out if h and "." in h}


mail = data.load_personal_mail(limit=6000)
ham_h = hosts(mail[mail.label_3 == "ham"])
del mail
pot = data.load_phishing_pot()
pot_h = hosts(pot)
del pot
feed = data.load_url_feed(n_benign=TOP)
test_ham = sorted(ham_h - pot_h)
test_pot = sorted(pot_h - ham_h - KNOWN)
print(f"mail ham hosts {len(test_ham):,}   pot test hosts {len(test_pot):,} "
      f"(collapsed by the rule: ham {sum(collapse(h) != h for h in test_ham):,}, "
      f"pot {sum(collapse(h) != h for h in test_pot):,})", flush=True)

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

model = um.load()


def p(urls, rule):
    urls = [collapse(u) for u in urls] if rule else list(urls)
    return model.predict_proba(um.featurize(urls))[:, 1] if urls else np.array([])


def worst(lists, rule):
    flat = [u for us in lists for u in us]
    s, out, i = p(flat, rule), [], 0
    for us in lists:
        out.append(s[i:i + len(us)].max())
        i += len(us)
    return np.array(out)


print()
print("rule      | mail ham FA >=.5  >=.7 | pot catch >=.5  >=.7 | openphish AUC  recall "
      "| sms smish caught / other FA >=.5 | gmail scans >=.7")
for rule in (False, True):
    h, pt, f = p(test_ham, rule), p(test_pot, rule), p(feed.url, rule)
    s = worst([u for u, k in zip(sms_urls, has) if k], rule)
    g = p(gmail, rule)
    print(f"{'known' if rule else 'none':9s} | {100 * (h >= .5).mean():11.1f}% {100 * (h >= .7).mean():5.1f}% | "
          f"{100 * (pt >= .5).mean():10.1f}% {100 * (pt >= .7).mean():5.1f}% | "
          f"{roc_auc_score(feed.label, f):13.3f} {100 * (f[feed.label.values == 1] >= .5).mean():6.1f}% | "
          f"{100 * (s[smish] >= .5).mean():10.1f}% / {100 * (s[~smish] >= .5).mean():.1f}% "
          f"| {int((g >= .7).sum())}/{len(g)}", flush=True)
