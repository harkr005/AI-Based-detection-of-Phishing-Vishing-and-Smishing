"""Should HTML-only links reach url_model on the Gmail path?

_pot_body strips tags, so a link that lives only in href="" never reaches
api._links - on HTML mail the URL channel is silent. data.link_hosts sees them.
But fusion renormalises over live channels: once a URL score exists it carries
~0.375 of the risk, so content-ham 0.00 + one odd-looking tracking host 0.95
already crosses SUSPICIOUS. This measures the trade end to end through
fusion.fuse (behaviour left out, as on a cold store), three ways:

  text   - today: only URLs visible in the scored text
  all    - text URLs + every href host
  off    - text URLs + href hosts NOT on the sender's own registrable domain

Ham = personal mailboxes (v6 trained on them, so the CONTENT FP is deflated;
the url increment is what to read). Phishing = phishing_pot sample.

RESULT 2026-10-09 (1500/mailbox, 2000 pot): ham not-SAFE 7.2 -> 40.4% (all)
/ 39.2% (off), phishing not-SAFE 96.3 -> 97.7 / 97.4%. NOT shipped - see
api._links. It also found url_model crashing on "http://[broken" links,
which is fixed (url_model._host).
Rates only, never message text (CLAUDE.md).
Run:  python scratch/_href_eval.py [per_mailbox_limit] [pot_sample]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                       # noqa: E402  # before torch
import numpy as np                # noqa: E402
import behaviour as bh            # noqa: E402
import email_model as em          # noqa: E402
import fusion                     # noqa: E402
import thread                     # noqa: E402
import url_model                  # noqa: E402


def url_max(urls):
    return float(url_model.score(urls).max()) if urls else None


def variants(row):
    text = thread.find_urls(row.text)
    home = bh.base_domain(bh.addr_parts(row.sender)[2])
    hosts = [f"http://{h}/" for h in row.link_hosts]
    off = [f"http://{h}/" for h in row.link_hosts if bh.base_domain(h) != home]
    return {"text": url_max(text), "all": url_max(text + hosts), "off": url_max(text + off)}


def verdicts(d):
    tok, model = em.load()
    probs = em.predict(tok, model, list(d.text))[0]
    out = {k: [] for k in ("text", "all", "off")}
    for p, row in zip(probs, d.itertuples()):
        for k, u in variants(row).items():
            out[k].append(fusion.fuse(p, url=u, malicious_idx=2)["verdict"])
    return {k: np.array(v) for k, v in out.items()}


def main(limit, n_pot):
    me = data.load_personal_mail(limit=limit)
    ham = me[me.label == 0]
    pot = data.load_phishing_pot().sample(n_pot, random_state=0)
    print(f"ham n={len(ham):,} (href hosts on {np.mean(ham.link_hosts.map(len) > 0):.0%}), "
          f"phishing n={len(pot):,} (href hosts on {np.mean(pot.link_hosts.map(len) > 0):.0%})")
    vh, vp = verdicts(ham), verdicts(pot)
    print(f"{'variant':8} {'ham not SAFE':>14} {'ham PHISHING':>14} {'phish not SAFE':>16} {'phish PHISHING':>16}")
    for k in ("text", "all", "off"):
        print(f"{k:8} {np.mean(vh[k] != 'SAFE'):14.1%} {np.mean(vh[k] == 'PHISHING'):14.1%} "
              f"{np.mean(vp[k] != 'SAFE'):16.1%} {np.mean(vp[k] == 'PHISHING'):16.1%}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 1500, int(a[1]) if len(a) > 1 else 2000)
