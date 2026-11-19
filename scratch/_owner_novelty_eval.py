"""Should "new sender + new domain" alone escalate mail from an established,
authenticated domain? (2026-10-28, after real Gmail use.)

On the first 121 live Gmail scans, 10 were escalated to SUSPICIOUS by the
behaviour gate alone (fusion.ESCALATE): every one DMARC-passed, content ~0.000,
and the only signals were first_contact + new_domain (+ url_first_contact).
9 of the 10 sat in Gmail's Promotions/Updates; 1 was in Gmail's Spam.

Proposed exemption: the novelty features (first_contact, new_domain,
url_first_contact) abstain when the From domain passed DMARC, is in Tranco
top-1M (url_model.sender_owner) and is NOT a free-mail provider - anyone can
open a new gmail.com address and it passes DMARC.

Pass rule, fixed before running:
  1. exempt share of real phishing (phishing_pot) <= 3%
  2. clears >= 5 of the 9 non-Spam live escalations
Also reported, not gated: exempt share of personal ham and of Gmail-Spam.

Rates and counts only, never message text or sender addresses (CLAUDE.md).
Run:  python scratch/_owner_novelty_eval.py [per_mailbox_limit]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                       # noqa: E402  # before torch
import behaviour as bh            # noqa: E402
import store                      # noqa: E402
import url_model                  # noqa: E402


def exempt(sender, dmarc):
    return (url_model.sender_owner(sender, dmarc) is not None
            and bh.addr_parts(sender)[2] not in bh.FREEMAIL)


def share(d):
    return sum(exempt(s, m) for s, m in zip(d.sender.fillna(""), d.dmarc.fillna(""))), len(d)


def main(limit):
    pot = data.load_phishing_pot()
    me = data.load_personal_mail(limit=limit)
    rows = [("phishing_pot (phishing)", share(pot)),
            ("personal ham", share(me[me.label == 0])),
            ("personal Gmail-Spam", share(me[me.label == 1]))]

    with store.connect() as c:
        live = c.execute(
            "SELECT sender, detail->>'dmarc', (detail->>'escalated')::boolean, "
            "detail->'gmail_labels' ? 'SPAM' FROM scans WHERE source = 'gmail'").fetchall()
    esc = [r for r in live if r[2]]
    cleared = [r for r in esc if exempt(r[0] or "", r[1] or "")]
    print(f"{'set':<28}{'exempt':>14}")
    for name, (k, n) in rows:
        print(f"{name:<28}{f'{k}/{n}':>14}  {k / max(n, 1):6.1%}")
    print(f"\nlive Gmail scans {len(live)}, escalated {len(esc)} "
          f"({sum(r[3] for r in esc)} in Gmail Spam)")
    print(f"  escalations the exemption clears: {len(cleared)} "
          f"(non-Spam {sum(not r[3] for r in cleared)}, Spam {sum(r[3] for r in cleared)})")

    k, n = rows[0][1]
    ok1, ok2 = k / n <= 0.03, sum(not r[3] for r in cleared) >= 5
    print(f"\n  {'PASS' if ok1 else 'FAIL'}  1 exempt phishing <= 3%")
    print(f"  {'PASS' if ok2 else 'FAIL'}  2 clears >= 5 of the non-Spam live escalations")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
