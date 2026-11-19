"""The repeat-sender TRUST rule: what it clears, and what phishing it lets through.

behaviour.trusted() = an address seen >= TRUST_N times before (or written to),
auth did not fail, and every link stays on the sender's registrable domain.
fusion.trust_damp() then multiplies the content model's phishing column.

Three measurements, each streamed in arrival order so "seen before" means
before (the same rule as behaviour.stream):

  A. COST - phishing_pot, one honeypot mailbox, 2022-2026. Repeat phishing
     senders DO exist there. Share of phishing the rule trusts, and v6's catch
     on that slice before/after the damp. This is the number that bounds it.
  B. BENEFIT, held out - the third mailbox (_mailbox3_eval.py): 100% ham, no
     person here trained on it. It has sender DOMAIN only, no dates, URLs
     stripped, so "established" is by domain in file order and the link check
     cannot run - an UPPER bound on what the rule clears.
  C. COVERAGE - the two personal mailboxes (dated, real links): how much ham
     comes from established senders, and how much of THAT keeps every link on
     the sender's domain. C x B is the realistic benefit. v6 trained on these
     boxes, so their FP rates are not quoted - coverage only.

Rates and sender domains only, never message text (CLAUDE.md).
Run:  python scratch/_trust_eval.py [per_mailbox_limit]
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                       # noqa: E402  # before torch
import numpy as np                # noqa: E402
import behaviour as bh            # noqa: E402
import email_model as em          # noqa: E402

import _mailbox3_eval as mb3      # noqa: E402

PHISH = em.LABELS.index("phishing")
NS, DAMPS = (1, 2, 3, 5), (1.0, 0.5, 0.2, 0.1)


def damp(probs, d):
    p = probs.copy()
    p[:, PHISH] *= d
    return p / p.sum(1, keepdims=True)


def prior_counts(keys):
    """How many times each row's key appeared BEFORE it."""
    seen, out = Counter(), []
    for k in keys:
        out.append(seen[k])
        seen[k] += 1
    return np.array(out)


def on_domain(sender, hosts):
    home = bh.base_domain(bh.addr_parts(sender)[2])
    return all(bh.base_domain(h) == home for h in hosts)


def scored(texts):
    tok, model = em.load()
    return em.predict(tok, model, list(texts))[0]


def main(limit):
    # A. cost on real phishing
    pot = data.load_phishing_pot().sort_values("date", kind="stable", na_position="first")
    addr = pot.sender.map(lambda s: bh.addr_parts(s)[1])
    prior = prior_counts(addr)
    fail = lambda c: pot[c].fillna("").str.lower().isin(bh.AUTH_FAIL).to_numpy()
    ok = (~fail("spf") & ~fail("dkim") & ~fail("dmarc")
          & np.array([on_domain(s, h) for s, h in zip(pot.sender, pot.link_hosts)])
          & addr.ne("").to_numpy())
    print(f"A. phishing_pot n={len(pot):,}  (repeat sender = seen >= N times before)")
    for n in NS:
        t = ok & (prior >= n)
        print(f"   N={n}: repeat {np.mean(prior >= n):5.1%}  -> trusted (auth ok, links on-domain) {t.mean():5.1%}  n={t.sum()}")
    t = ok & (prior >= bh.TRUST_N)
    if t.sum():
        p = scored(pot.text[t])
        print(f"   v6 phishing caught on the trusted slice (N={bh.TRUST_N}), by damp: "
              + "  ".join(f"{d}: {np.mean(damp(p, d).argmax(1) == PHISH):.1%}" for d in DAMPS))
        lost = [np.sum((p.argmax(1) == PHISH) & (damp(p, d).argmax(1) != PHISH)) for d in DAMPS]
        print(f"   phishing LOST overall (of {len(pot):,}): "
              + "  ".join(f"{d}: {l} ({l / len(pot):.2%})" for d, l in zip(DAMPS, lost)))

    # B. benefit on the held-out third mailbox
    d3 = mb3.load()
    p3 = scored(d3.text)
    prior3 = prior_counts(d3.sender_domain)
    print(f"\nB. third mailbox n={len(d3)}, v6 phishing FP = {np.mean(p3.argmax(1) == PHISH):.1%}")
    for n in NS:
        t3 = prior3 >= n
        row = "  ".join(f"{d}: {np.mean(np.where(t3, damp(p3, d).argmax(1), p3.argmax(1)) == PHISH):.1%}"
                        for d in DAMPS)
        print(f"   N={n}: established {t3.mean():5.1%}  FP by damp  {row}")
    t3 = prior3 >= bh.TRUST_N
    left = d3[(np.where(t3, damp(p3, .2).argmax(1), p3.argmax(1)) == PHISH)].sender_domain
    print("   still flagged at N=TRUST_N, damp .2:", ", ".join(f"{k} {v}" for k, v in left.value_counts().head(5).items()))

    # C. coverage on dated personal mail with real links
    me = data.load_personal_mail(limit=limit)
    me = me[me.label == 0].sort_values("date", kind="stable", na_position="first")
    for who, g in me.groupby("person"):
        pc = prior_counts(g.sender.map(lambda s: bh.addr_parts(s)[1]))
        est = pc >= bh.TRUST_N
        ond = np.array([on_domain(s, h) for s, h in zip(g.sender, g.link_hosts)])
        print(f"\nC. {who}: ham n={len(g):,}  established {est.mean():.1%}  "
              f"of those, every link on-domain {ond[est].mean():.1%}  -> trusted {np.mean(est & ond):.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
