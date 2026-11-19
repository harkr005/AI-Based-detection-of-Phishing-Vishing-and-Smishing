"""Do the attachment checks (attachments.py) catch phishing without flagging real mail?

Sets: every phishing_pot mail with a named attachment or a QR link in an
embedded image (positives), and every personal-mailbox HAM mail with one
(Spam/Trash excluded, deduped by Message-ID). Each goes through fusion.fuse as
on a cold store (no behaviour, no Safe Browsing - offline), the link slot being:
  base   - links visible in the text (today)
  types  - base, or LISTED_RISK if a dangerous file type / zip is attached
  links  - base + links inside PDF / HTML / ICS attachments, one url_model call
  qr     - base + links in QR codes (images, embedded images, PDF images);
           only the qr-check branch reads QR codes, on master this equals base
  all    - types, links and qr
Ham caveat as in _href_eval.py: v6 trained on these mailboxes, so ham's content
false alarms are deflated; the INCREMENT over base is what to read.

FIXED 2026-10-28 BEFORE ANY SCORE WAS SEEN (types/links; qr added the same day,
same rule, also before its scores): a layer ships iff
  (1) ham-with-attachments not-SAFE rises <= 1.0 pt over base, and
  (2) it lifts at least one phishing-with-attachments mail off SAFE.
Rates and counts only, never message text or filenames.
    python -m scratch._attach_eval
"""
import collections
import mailbox
import re

import data   # before torch (Windows libiomp)
import numpy as np

import attachments
import email_model as em
import fusion
import safe_browsing
import thread
import url_model

# filename= / name= / RFC 2231 filename*= - get_filename() reads all three
NAMED = re.compile(rb"name\*?=", re.I)
IMG = re.compile(rb"content-type:\s*image/", re.I)
MID = re.compile(rb"^message-id:\s*(<[^>]*>)", re.I | re.M)


def pot():
    for f in sorted(data.POT.glob("*.eml")):
        yield f.read_bytes()


def ham():
    seen = set()
    for f in sorted(data.PERSONAL.glob("*.mbox")) + sorted(data.PERSONAL.glob("*/*.mbox")):
        if f.stem.lower() in {"spam", "trash", "junk"}:
            continue
        box = mailbox.mbox(str(f))
        for k in box.iterkeys():
            raw = box.get_bytes(k)
            if not (NAMED.search(raw) or IMG.search(raw)):   # cheap pre-filter; scan() decides
                continue
            mid = MID.search(raw)                     # Gmail folds the id onto the next line
            mid = mid[1] if mid else raw[:500]
            if mid in seen:
                continue
            seen.add(mid)
            yield raw


def rows(gen):
    out = []
    for raw in gen:
        a = attachments.scan(raw)
        if not (a["count"] or a["urls"]):
            continue
        r = data.parse_raw(raw)
        if len(r["text"]) > 5:
            out.append((r["text"], a))
    return out


def umax(urls):
    return float(url_model.score(urls, mail=True).max()) if urls else None


def mx(*v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def verdicts(rs):
    tok, model = em.load()
    probs = em.predict(tok, model, [t for t, _ in rs])[0]
    out, lifted_by = collections.defaultdict(list), collections.Counter()
    for p, (t, a) in zip(probs, rs):
        base = umax(list(dict.fromkeys(thread.find_urls(t))))
        typ = safe_browsing.LISTED_RISK if a["bad"] else None
        att = umax([u for u, k in a["urls"] if not k.endswith("QR code")])
        qr = umax([u for u, k in a["urls"] if k.endswith("QR code")])
        v = {"base": base, "types": mx(base, typ), "links": mx(base, att), "qr": mx(base, qr),
             "all": mx(base, typ, att, qr)}
        for k, u in v.items():
            out[k].append(fusion.fuse(p, url=u, malicious_idx=2)["verdict"])
        if out["base"][-1] == "SAFE" and out["all"][-1] != "SAFE":
            lifted_by.update({kind for _, kind in a["urls"]} | ({"(type)"} if a["bad"] else set()))
    return {k: np.array(v) for k, v in out.items()}, lifted_by


def main():
    P, H = rows(pot()), rows(ham())
    for name, rs in (("phishing", P), ("ham", H)):
        print(f"{name}: {len(rs):,} mails with attachments | dangerous type {sum(bool(a['bad']) for _, a in rs)} | "
              f"links inside {sum(any(not k.endswith('QR code') for _, k in a['urls']) for _, a in rs)} | "
              f"QR links {sum(any(k.endswith('QR code') for _, k in a['urls']) for _, a in rs)}", flush=True)
    vp, lp = verdicts(P)
    vh, lh = verdicts(H)
    print(f"\n{'variant':8} {'ham not SAFE':>13} {'ham PHISHING':>13} {'phish not SAFE':>15} {'phish PHISHING':>15}")
    for k in ("base", "types", "links", "qr", "all"):
        print(f"{k:8} {np.mean(vh[k] != 'SAFE'):13.1%} {np.mean(vh[k] == 'PHISHING'):13.1%} "
              f"{np.mean(vp[k] != 'SAFE'):15.1%} {np.mean(vp[k] == 'PHISHING'):15.1%}")
    print("\nlifted off SAFE, by attachment kind:  phishing", dict(lp), "| ham", dict(lh))
    for k in ("types", "links", "qr"):
        rise = np.mean(vh[k] != "SAFE") - np.mean(vh["base"] != "SAFE")
        caught = int(((vp["base"] == "SAFE") & (vp[k] != "SAFE")).sum())
        print(f"-> {k}: ham +{rise:.1%} not-SAFE, {caught} phishing lifted -> "
              f"{'PASS' if rise <= .010 and caught >= 1 else 'FAIL'} by the rule in the docstring")


if __name__ == "__main__":
    main()

# RESULT 2026-10-28 (scratch/_attach_eval.log). phishing_pot 615 mails with attachments
# (19 dangerous type, 150 with links inside); personal ham 1,606 (0 dangerous type, 403
# with links inside). Cold store, no Safe Browsing.
#            ham not SAFE  ham PHISHING  phish not SAFE  phish PHISHING
#   base          0.4%         0.2%          93.0%          53.3%
#   types         0.4%         0.2%          93.3%          53.8%
#   links         0.4%         0.2%          93.5%          56.1%
#   both          0.4%         0.2%          93.8%          56.6%
# Both layers PASS: zero added ham false alarms; 5 phishing mails lifted off SAFE
# (3 by a link in a PDF, 2 by file type) and +3.3 pts reach PHISHING. Shipped in api.py.
# The gain is small because content already catches 93% of this set; the layer is
# for the mail whose body is bland and whose payload is the file.

# RESULT 2026-10-28, QR layer (scratch/_attach_eval_qr.log): FAIL. phishing_pot holds no
# QR codes at all - 901 images, 761 readable, 0 barcodes of any format (checked
# separately, mostly logos under 400 px) - so the layer lifts 0 phishing mails, while
# personal ham has QR links in 74 mails and 2 of them went SAFE -> SUSPICIOUS (+0.1 pt).
# Quishing is real, but this corpus cannot show the layer helps. types/links unchanged
# (ham 1,608 with the embedded-image mails added: still +0.0 pt).
