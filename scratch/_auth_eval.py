"""How often does SPF/DKIM/DMARC FAIL on real phishing vs real ham?

Measured before `behaviour.auth_fail` got a weight (data.load_phishing_pot's
docstring: "measure it before weighting it" - SPF PASSES on 53% of pot).
A pass is not evidence of legitimacy, so the only question is the fail side:
what fraction of phishing it catches, and what it costs on mail you kept.

Phishing = phishing_pot (stamped by the honeypot's receiver). Ham = personal
mailboxes, label 0 (stamped by Gmail). Different stampers is a caveat, but
"fail" means the same thing in RFC 8601 whoever writes it. Rates only - never
print message text (CLAUDE.md, private mailboxes).

Run:  python scratch/_auth_eval.py [per_mailbox_limit]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                       # noqa: E402  # before torch, see sms_model
import pandas as pd               # noqa: E402

FAIL = {"fail", "softfail", "permerror"}


def rules(d):
    low = lambda c: d[c].fillna("").str.lower()
    spf, dkim, dmarc = low("spf"), low("dkim"), low("dmarc")
    return {
        "no auth header at all": (spf == "") & (dkim == "") & (dmarc == ""),
        "spf fail/softfail": spf.isin(FAIL),
        "dkim fail": dkim.isin(FAIL),
        "dmarc fail": dmarc.isin(FAIL),
        "dmarc none/absent": dmarc.isin({"", "none"}),
        "spf AND dkim not pass": ~spf.eq("pass") & ~dkim.eq("pass"),
        "ANY fail (spf|dkim|dmarc)": spf.isin(FAIL) | dkim.isin(FAIL) | dmarc.isin(FAIL),
    }


def main(limit):
    pot = data.load_phishing_pot()
    me = data.load_personal_mail(limit=limit)
    sets = [("phishing_pot", pot)]
    for who, g in me.groupby("person"):
        sets.append((f"ham {who}", g[g.label == 0]))
        if (g.label == 1).sum():
            sets.append((f"gmail-spam {who}", g[g.label == 1]))
    print(f"{'rule':<28}" + "".join(f"{n:>18}" for n, _ in sets))
    print(f"{'n':<28}" + "".join(f"{len(d):>18,}" for _, d in sets))
    table = {n: rules(d) for n, d in sets}
    for r in table[sets[0][0]]:
        print(f"{r:<28}" + "".join(f"{table[n][r].mean():>17.1%} " for n, _ in sets))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
