"""Two layers on top of the MAIL link model (url_xgb_v4), measured per mail
(2026-10-11). Nothing is trained.

A. BRAND MISMATCH (combosquatting, Kintis et al. CCS 2017; the lexical half of
   Phishpedia's brand-vs-domain check). A host token (split on . and -, from
   everything in front of the public suffix) equals a brand, and the link's
   registrable domain is not one of that brand's domains -> the link scores at
   least BRAND_FLOOR. Brands are not hand-typed:
     - ScamShield registry entities, legit domains = their registry domains;
     - Tranco top-1000 first labels, legit domains = every Tranco top-1M
       registrable domain whose first label is, or hyphen-contains, the token.
   Generic words are dropped by Tranco alone, fixed before any test: a token
   used inside > GENERIC hyphenated top-1M labels (login 56, office 74, bank
   153, online 1529) is a word, not a brand (apple 28, amazon 27, paypal 7).
B. SENDER'S OWN LINK. Mail whose From domain passed DMARC, linking to that same
   registrable domain: score the registrable domain alone (drop the part in
   front). The domain itself is still judged, so an attacker who set up DMARC
   for hdfc-secure.tk is still scored as hdfc-secure.tk.

Per mail = worst VISIBLE link, as api._links scores it. Rows:
  user2 ham    genuine mail of a person the mail model never trained on -> FA
  self ham       reference only: v4 trained on self's link hosts
  pot phishing   phishing_pot mail with a link -> catch
Pass bar: catch up or equal, false alarms down or equal.

    python -m scratch._url_layers_eval      (rates only: private mail)
"""
import collections
import email.utils
import re

import data                      # before torch / xgboost
import numpy as np
import pandas as pd

import thread
import url_model as um

GENERIC, TOP_BRANDS, BRAND_FLOOR = 40, 1000, 0.9

tr = pd.read_csv(data.DATA / "URL" / "tranco_N2KLW.csv", names=["rank", "d"])
regs = [um.split(d.lower())[1] for d in tr.d]
first = [r.split(".")[0] for r in regs]
compound = collections.Counter(t for l in first if "-" in l for t in set(l.split("-")))
KNOWN = set(regs)       # B': owner must also be an established (top-1M) domain
by_token = collections.defaultdict(set)
for r, l in zip(regs, first):
    for t in {l, *l.split("-")}:
        by_token[t].add(r)
BRANDS = {l: by_token[l] for l in first[:TOP_BRANDS]
          if len(l) >= 3 and compound[l] <= GENERIC}
reg_kb = pd.read_csv(data.KB / "verified_urls.csv")
for _, g in reg_kb.groupby("entity_id"):
    own = {um.split(um._host(u))[1] for u in g.url}
    for r in own:
        l = r.split(".")[0]
        if len(l) >= 3 and compound[l] <= GENERIC:
            BRANDS.setdefault(l, set()).update(own)
print(f"brands {len(BRANDS):,} (Tranco top-{TOP_BRANDS} + registry, generic cutoff {GENERIC})", flush=True)


def brand_mismatch(url):
    """-> the brand a host claims but does not belong to, else None."""
    sub, reg = um.split(um._host(url))
    toks = set(re.split(r"[.\-]", sub)) | set(reg.split(".")[0].split("-"))
    for t in toks:
        if t in BRANDS and reg not in BRANDS[t]:
            return t
    return None


def sender_reg(sender):
    addr = email.utils.parseaddr(str(sender))[1]
    return um.split(addr.rsplit("@", 1)[-1])[1] if "@" in addr else None


def worst(rows, brand, own):
    """-> worst link score per mail with a visible link."""
    out = []
    for text, sender, dmarc in rows:
        urls = thread.find_urls(text)
        if not urls:
            continue
        owner = sender_reg(sender) if own and str(dmarc).lower() == "pass" else None
        if own == "known" and owner not in KNOWN:
            owner = None
        scored = [f"https://{um.split(um._host(u))[1]}"
                  if owner and um.split(um._host(u))[1] == owner else u for u in urls]
        s = um.score(scored, mail=True)
        if brand:
            s = np.array([max(v, BRAND_FLOOR) if brand_mismatch(u) else v
                          for v, u in zip(s, urls)])
        out.append(s.max())
    return np.array(out)


mail = data.load_personal_mail(limit=6000)
ham = mail[mail.label_3 == "ham"]
sets = {
    "user2 ham": ham[ham.person == "user2"],
    "self ham (ref)": ham[ham.person == "self"],
}
del mail
pot = data.load_phishing_pot()
sets["pot phishing"] = pot
for name, d in sets.items():
    print(f"{name:15s} mails {len(d):,}  DMARC pass {100 * d.dmarc.astype(str).str.lower().eq('pass').mean():.0f}%", flush=True)

print()
print(f"{'layers':22s} | " + " | ".join(f"{n:>15s} >=.5  >=.7" for n in sets))
for label, brand, own in [("v4 alone", False, False), ("+ A brand mismatch", True, False),
                          ("+ B sender's own link", False, True), ("+ A + B", True, True),
                          # chosen AFTER the rows above were seen (pot -2.2 from B):
                          ("+ B' known sender only", False, "known"), ("+ A + B'", True, "known")]:
    cells = []
    for name, d in sets.items():
        w = worst(zip(d.text, d.sender, d.dmarc), brand, own)
        cells.append(f"{100 * (w >= .5).mean():15.1f}% {100 * (w >= .7).mean():5.1f}%")
    print(f"{label:22s} | " + " | ".join(cells), flush=True)

# How often does each layer fire, and on what (brands are public names).
hits = collections.Counter(b for t in pot.text for u in thread.find_urls(t) if (b := brand_mismatch(u)))
fa = collections.Counter(b for t in sets["user2 ham"].text for u in thread.find_urls(t) if (b := brand_mismatch(u)))
print("\nbrand mismatch, pot links: ", sum(hits.values()), hits.most_common(12))
print("brand mismatch, user2 ham links:", sum(fa.values()), fa.most_common(12))
