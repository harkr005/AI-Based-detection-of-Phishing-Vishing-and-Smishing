"""Dataset loaders. One function per channel, each returns a DataFrame.

Label convention everywhere: 1 = malicious, 0 = benign.
PhiUSIIL ships the opposite (1 = legit), so it gets flipped here.
"""
import email
import email.header
import hashlib
import json
import mailbox
import os
import email.utils
import html
import re
from email import policy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "Datasets"
INDIA = Path(__file__).parent / "Dataset(Indian)" / "sms"
# ScamShield lookup tables. NOT training data (zero labelled messages) - the
# three whitelists in here are consumed as features by url_model and behaviour.
KB = INDIA / "smishing-intelligence-dataset" / "data"

# ponytail: phishing_email.csv is the pre-merged version of the 6 below -> skip it
# or you train and test on the same mails twice.
EMAIL_FILES = ["CEAS_08.csv", "Enron.csv", "Ling.csv", "Nazario.csv",
               "Nigerian_Fraud.csv", "SpamAssasin.csv"]


# Which corpora carry real phishing rather than ordinary spam. CEAS_08's shipped
# label means SPAM (CEAS = Conference on Email and Anti-Spam) - a model trained on
# it catches only 31% of Nazario. So the 3-class label comes from provenance, not
# from the label column, exactly as the SMS channel does.
PHISH_CORPORA = {"Nazario", "Nigerian_Fraud"}


def load_email():
    """subject + body + label (+ sender/date where the corpus has them).

    label   1 = unwanted (spam or phishing), 0 = ham   - kept for compatibility
    label_3 ham | spam | phishing                      - use this one
    """
    parts = []
    for f in EMAIL_FILES:
        d = pd.read_csv(DATA / "phishing" / f, low_memory=False)
        d["corpus"] = f.replace(".csv", "")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["subject"] = d["subject"].fillna("")
    d["body"] = d["body"].fillna("")
    d["text"] = (d["subject"] + "\n" + d["body"]).str.strip()
    d["receiver"] = d.get("receiver", pd.Series(index=d.index, dtype=object))
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce", utc=True, format="mixed")

    # SpamAssasin.csv ships 412 columns and 10,917 rows carrying no label at all
    # (10,793 of them completely empty padding rows). That makes its label column
    # dtype str, so its 4,079 HAM rows arrive as the string '0' - and `label == 0`
    # is False for every one of them. Before this line the loader emitted
    # {'spam': 16698} for SpamAssasin: all 4,079 ham relabelled spam, plus 10,793
    # empty documents labelled spam. Coerce first, then drop what has no label;
    # 4,079 ham + 1,702 spam is the real published SpamAssassin corpus.
    d["label"] = pd.to_numeric(d["label"], errors="coerce")
    d = d[d.label.isin([0, 1])].reset_index(drop=True)
    d["label"] = d["label"].astype(int)

    d["label_3"] = np.where(d.label == 0, "ham",
                            np.where(d.corpus.isin(PHISH_CORPORA), "phishing", "spam"))
    return d[["text", "label", "label_3", "sender", "receiver", "date", "urls", "corpus"]]


POT = DATA / "phishing" / "phishing_pot" / "email"
SA_RAW = DATA / "phishing" / "spamassassin_raw"
PERSONAL = DATA / "personal"
NL = chr(10)
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")
# Never mail text: stripping tags alone keeps what is INSIDE them, so every
# <style> block reached the model as CSS ("#outlook a { padding:0;}") and every
# <script> as JS. The <head> goes too - its <title> repeats the subject.
_NOT_TEXT = re.compile(r"<(style|script|head)\b.*?</\1\s*>|<!--.*?-->", re.I | re.S)
# HTML sent inside a part labelled text/plain (a real job-portal mail did).
_LOOKS_HTML = re.compile(r"<(!doctype|html|body|div|table)\b", re.I)
# Invisible preheader padding - marketing mail follows its preview line with
# hundreds of &zwnj;/&shy; so the inbox shows nothing after it.
_INVISIBLE = re.compile("[\u00ad\u034f\u200b-\u200f\u2060\ufeff]")
# readable_body only: where a line ends, and a link with its real target.
_BREAK = re.compile(r"<br\b[^>]*>|</(p|div|tr|li|h[1-6]|table|blockquote)\s*>", re.I)
_LINK = re.compile(r"""<a\b[^>]*?href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a\s*>""", re.I | re.S)
_SPF = re.compile(r"\bspf=(\w+)", re.I)
_DKIM = re.compile(r"\bdkim=(\w+)", re.I)
_DMARC = re.compile(r"\bdmarc=(\w+)", re.I)


def _hdr(raw):
    """RFC2047-decoded header text. compat32 does not decode =?UTF-8?B?..?= for
    us, and 2,000+ of these subjects are encoded - left raw they train the model
    on base64."""
    try:
        return str(email.header.make_header(email.header.decode_header(raw or "")))
    except Exception:
        return str(raw or "")


def _pot_body(m):
    """Plain part if there is a non-empty one, else the HTML part as text.
    -> (one line of text, has_url). This is what every email model reads."""
    txt, is_html = _body_part(m)
    has_url = bool(_URL.search(txt))          # BEFORE tag stripping: an HTML-only
    if is_html:                               # phish keeps its links in href=""
        txt = html.unescape(_TAG.sub(" ", _NOT_TEXT.sub(" ", txt)))
    return " ".join(_INVISIBLE.sub("", txt).split()), has_url


_HREF_URL = re.compile(r"""https?://[^\s"'<>]+""", re.I)


def link_hosts(m):
    """Every host a link in the scored part points at, lowercased, deduped.

    From the part BEFORE tag stripping: _pot_body's text has lost every href, so
    a check on it would call an HTML phish with a link to evil.tk "link-free".
    behaviour.trusted() reads this - its whole safety rests on seeing every link.
    """
    def host(u):
        try:
            return urlsplit(html.unescape(u)).hostname
        except ValueError:          # "http://[bad" - phishing produces these
            return "?"              # unknown host: never on the sender's domain
    txt, _ = _body_part(m)
    return sorted({h.lower() for h in map(host, _HREF_URL.findall(txt)) if h})


def readable_body(m):
    """The same part as _pot_body, laid out for a PERSON: line breaks kept, and
    every link followed by the host it really goes to. Display only - the web
    app's View - because the models were trained on _pot_body's single line."""
    txt, is_html = _body_part(m)
    if is_html:
        txt = _LINK.sub(_show_link, _NOT_TEXT.sub(" ", txt))
        txt = html.unescape(_TAG.sub(" ", _BREAK.sub("\n", txt)))
    lines = (" ".join(line.split()) for line in _INVISIBLE.sub("", txt).splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _show_link(g):
    # "Verify now" pointing at a lookalike domain is the one thing a reader of
    # a scam checker must see, so the host follows the words. Host only: the
    # full tracking URL buries the sentence it sits in.
    # Skipped only when the words ARE that host: "host in text" let
    # "https://www.hdfcbank.com/?r=evil.tk" hide a link to evil.tk.
    text = html.unescape(_TAG.sub(" ", g[2])).strip()
    host = urlsplit(html.unescape(g[1])).netloc
    shown = urlsplit(text if "//" in text else "//" + text).netloc
    return f"{g[2]} ({host})" if text and host and shown.lower() != host.lower() else g[2]


def parse_readable(raw):
    """RFC 822 bytes -> what the web app's View shows (api.py /email/parse)."""
    m = email.message_from_bytes(raw, policy=policy.compat32)
    # " ".join(split()) unfolds long headers, which arrive broken across lines.
    # page is the largest text/html part, defused (safe_html), for the View's
    # sandboxed "as sent" frame. Never scored and never stored.
    parts = [(x, x.get_payload(decode=True) or b"") for x in m.walk()
             if x.get_content_type() == "text/html"]
    part, body = max(parts, key=lambda pr: len(pr[1]), default=(None, b""))
    try:
        page = body.decode(part.get_content_charset() or "utf-8", "replace") if part else ""
    except LookupError:                   # a charset name Python does not know
        page = body.decode("utf-8", "replace")
    return {"subject": " ".join(_hdr(m.get("Subject", "")).split()),
            "sender": " ".join(_hdr(m.get("From", "")).split()), "text": readable_body(m),
            "html": safe_html(page) or None}


class _Defuse(HTMLParser):
    """Re-emits mail HTML minus everything that can reach the network or move
    the frame without the CSP's say-so. The browser's sandbox="" + CSP is the
    other half; neither is enough alone (privacy review 2026-10-11):
      - a link's own target="_self" beats <base target> and navigates the
        frame to the sender's page, outside our CSP -> no href/target at all;
      - <link rel=preconnect|dns-prefetch> is not covered by CSP, and Firefox
        may follow <meta http-equiv=refresh> -> no <link>, <meta>, <base>;
      - the mail's own <meta name=referrer> would override ours -> no <meta>;
      - inside <svg>/<math> a browser parses <style> as markup while this
        parser passes it through raw, and SVG <set> can write an href back
        (privacy review 2026-10-12) -> no <svg>/<math> at all.
    Link destinations are not lost: the verdict panel lists every one."""
    DROP = {"script", "meta", "link", "base", "iframe", "frame", "object", "embed",
            "svg", "math"}
    VOID = {"meta", "link", "base", "embed", "frame"}   # no end tag, no content
    STRIP = {"href", "xlink:href", "target", "action", "formaction", "ping",
             "http-equiv", "srcdoc"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.raw = [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.DROP:
            self.skip += tag not in self.VOID
            return
        if self.skip:
            return
        self.raw = tag == "style"             # CSS is emitted as-is, never escaped
        keep = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs
                       if k not in self.STRIP and not k.startswith("on"))
        self.out.append(f"<{tag}{keep}>")

    def handle_endtag(self, tag):
        if tag in self.DROP:
            self.skip = max(0, self.skip - (tag not in self.VOID))
        elif not self.skip:
            self.raw = False
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data if self.raw else html.escape(data, quote=False))


def safe_html(page):
    """Mail HTML -> the same mail with no links, scripts, metas or embeds."""
    p = _Defuse()
    p.feed(page)
    p.close()
    return "".join(p.out)


def _body_part(m):
    """-> (decoded text of the part worth reading, whether it is HTML).

    Walks the parts by hand under compat32 rather than using EmailMessage.
    get_body(): policy.default parses address headers strictly and a malformed
    Reply-To - which phishing mail produces constantly - raises out of the
    parser. compat32 hands back raw strings and cannot hit that path.
    """
    parts = [x for x in m.walk() if x.get_content_maintype() == "text"]
    # Empty parts are dropped BEFORE choosing plain over HTML: a mail with an
    # empty text/plain beside a full text/html otherwise returned "" - the
    # whole message scored on its subject alone.
    decoded = [(x, x.get_payload(decode=True) or b"") for x in parts]
    decoded = [(x, raw) for x, raw in decoded if raw.strip()]
    plain = [pr for pr in decoded if pr[0].get_content_subtype() == "plain"] or decoded
    if not plain:
        return "", False
    # The LARGEST candidate, not the first. Real mail carries decoy parts: a
    # multipart/alternative in the personal Spam box ships a tiny text/plain
    # ahead of the 871-byte one.
    part, raw = max(plain, key=lambda pr: len(pr[1]))
    try:
        txt = raw.decode(part.get_content_charset() or "utf-8", "replace")
    except LookupError:            # real sample: charset='"u"tf-8'
        txt = raw.decode("utf-8", "replace")
    return txt, part.get_content_subtype() != "plain" or bool(_LOOKS_HTML.search(txt))


def _parse_mail(path):
    """One .eml -> one row. Shared by both raw-mail corpora.

    compat32 throughout: policy.default parses address headers strictly and a
    malformed Reply-To - which phishing mail produces constantly - raises out of
    the parser.
    """
    with path.open("rb") as fh:
        try:
            m = email.message_from_binary_file(fh, policy=policy.compat32)
        except Exception:
            return None
    return _mail_row(m, path.name)


def parse_raw(raw):
    """RFC 822 bytes -> the same row a .eml file gives. Gmail's
    messages.get(format=raw) lands here (api.py /email/raw), so live mail is
    read by the exact parser every email number was measured through. Also
    carries `subject` on its own (unfolded), for the web app's list."""
    m = email.message_from_bytes(raw, policy=policy.compat32)
    return {**_mail_row(m, ""), "subject": " ".join(_hdr(m.get("Subject", "")).split())}


def _mail_row(m, name):
    """A parsed message -> one row. Same fields whether it came from a .eml file
    or an mbox, so every raw-mail corpus lines up column for column."""
    body, has_url = _pot_body(m)
    auth = " ".join([str(m.get("Authentication-Results", "")),
                     str(m.get("Received-SPF", ""))])
    try:
        date = email.utils.parsedate_to_datetime(m.get("Date"))
    except Exception:
        date = None
    return {
        "text": f"{_hdr(m.get('Subject', ''))}{NL}{body}".strip(),
        "sender": _hdr(m.get("From", "")),
        "receiver": str(m.get("To", "")),
        "date": date,
        "urls": int(has_url),
        "spf": (_SPF.search(auth) or [None, None])[1],
        "dkim": (_DKIM.search(auth) or [None, None])[1],
        "dmarc": (_DMARC.search(auth) or [None, None])[1],
        "link_hosts": link_hosts(m),
        "reply_to": str(m.get("Reply-To", "")),
        "return_path": str(m.get("Return-Path", "")),
        "message_id": str(m.get("Message-ID", "")).strip(),
        "file": name,
    }


def load_phishing_pot():
    """Real phishing .eml WITH HEADERS - the one thing no CSV in this repo has.

    8,614 messages, CC BY-NC 4.0 (academic use only, no commercial). Measured
    over all of them: dated 2022-2026 (2023 2,174 / 2024 2,141 / 2025 2,012 /
    2026 1,984, and exactly 2 forged years), 77.7% carry a URL, 25.9% set
    Reply-To. Only the honeypot's OWN address is anonymised to phishing@pot; the
    spoofed sender survives in 96% of mails, which is the field
    behaviour.features() actually reads.

    TWO NUMBERS THAT CHANGE WHAT §7 CAN CLAIM:
      spf   pass 4,559 | none 2,419 | fail 783 | softfail 449
      dkim  none 4,262 | pass 3,353 | fail 815
    SPF PASSES ON 53% OF THIS PHISHING. The spec lists "SPF/DKIM/DMARC result"
    as an email behaviour feature; on modern phishing it is close to useless
    alone, because the attacker owns the sending domain and publishes his own
    SPF record. It separates spoofed-domain phishing only, which is a shrinking
    slice. Measure it before weighting it.

    Only ~43.6% is English (stopword heuristic) - the rest is mostly Portuguese,
    Dutch and German honeypot traffic. Filter before feeding an English model:
    that leaves ~3,750 modern phishing mails, still more than double Nazario.

    100% positive, exactly like Nazario and Nigerian_Fraud - it yields recall,
    never an AUC, and a benign half has to come from somewhere else.

    Deliberately NOT folded into load_email(): adding 8.6k modern phishing there
    would silently move every email number already measured. Callers merge it on
    purpose or not at all.
    """
    rows = [r for r in (_parse_mail(f) for f in sorted(POT.glob("*.eml"))) if r]
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d.date, errors="coerce", utc=True)
    d["label"], d["label_3"], d["corpus"] = 1, "phishing", "phishing_pot"
    return d


ENRON_FULL = DATA / "enron_full" / "maildir"
# Enron-Spam's six ham mailboxes (Metsis et al. 2006). The cross-source "Enron"
# test set is drawn from them, so they never train anything.
ENRON_HELD_OUT = {"farmer-d", "kaminski-v", "kitchen-l", "williams-w3", "beck-s", "lokay-m"}
_QUOTE = re.compile(r"-{3,}\s*Original Message\s*-{3,}|-{3,}\s*Forwarded by", re.I)


def load_enron_full(limit=150, held_limit=1000):
    """CMU Enron (2015-05-07 release): real colleague-to-colleague mail, the genre
    no other ham source here has. v4 flags 11% of Enron-Spam ham as phishing
    because its ham was mailing lists and Gmail promotions only.

    Internal senders only (@enron.com), so the newsletters and spam sitting in
    the inboxes stay out. Quoted chains are cut at "Original Message" /
    "Forwarded by": phishing never carries them, so the marker alone would be a
    free ham label. `limit` files per mailbox (`held_limit` for the six), evenly spaced.

    `held_out` marks the six Enron-Spam mailboxes. Any other row whose body also
    appears there (a mail sent TO kaminski sits in both mailboxes) is dropped, so
    held-out rows can be scored as unseen people.
    """
    rows = []
    # Every file is named "1.", "2.", ... and Win32 strips a trailing dot, so a
    # plain path finds nothing; the \\?\ prefix turns that normalisation off.
    root = Path("\\\\?\\" + str(ENRON_FULL.resolve())) if os.name == "nt" else ENRON_FULL
    for user in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(Path(dp, f) for dp, _, fs in os.walk(user) for f in fs)
        n = held_limit if user.name in ENRON_HELD_OUT else limit   # test needs rows
        if n and len(files) > n:
            step = len(files) / n
            files = [files[int(i * step)] for i in range(n)]
        for f in files:
            r = _parse_mail(f)
            if r and "@enron.com" in r["sender"].lower():
                r["text"] = _QUOTE.split(r["text"])[0].strip()
                r["user"] = user.name
                rows.append(r)
    d = pd.DataFrame(rows)
    d = d[d.text.str.split().str.len() >= 5]
    # Same mail, many folders (inbox, all_documents, discussion_threads...).
    d["dup"] = d.text.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str[:300]
    d["held_out"] = d.user.isin(ENRON_HELD_OUT)
    d = d[d.held_out | ~d.dup.isin(set(d.dup[d.held_out]))]
    d = d.drop_duplicates("dup")
    d["date"] = pd.to_datetime(d.date, errors="coerce", utc=True)
    d["label"], d["label_3"], d["corpus"] = 0, "ham", "enron_full"
    return d.drop(columns="dup").reset_index(drop=True)


def load_spamassassin_raw():
    """The SpamAssassin public corpus as SHIPPED: raw mail with headers.

    Same corpus as SpamAssasin.csv, from spamassassin.apache.org/old/publiccorpus.
    The CSV kept only subject+body and ships 10,917 unlabelled rows (see the note
    in load_email); here every message keeps its Received path, Return-Path,
    Reply-To and real From. This is the HAM-with-headers half that phishing_pot
    cannot supply.

    The nine release tarballs overlap heavily: 10,745 files hold 6,112 distinct
    messages. Deduped by Message-ID (content hash for the 3 without one) ->
    4,211 ham + 1,901 spam, which is the published corpus. Deduping by BYTES
    instead leaves 9,165, because releases were re-anonymised between 2002 and
    2005 - that would leak the same mail into train and test, the Dataset_10191
    vs Dataset_5971 trap again.

    NO SPF/DKIM: this is 2002-2005 mail and neither protocol existed yet. Sender,
    Received path, Reply-To and real timestamps are what it offers.

    223 mails (3.6%) carry a Date outside 2002-2005 - 1998, 0102, 2028 - and
    **222 of the 223 are spam** (11.7% of spam, 0.02% of ham). Forged Date is a
    feature in its own right, but sort by it before feeding behaviour.stream()
    and 222 spam land in the wrong decade, so the profile the stream builds is
    not the one that mail arrived in. Filter or clamp first.
    """
    rows, seen = [], set()
    for folder in sorted(x for x in SA_RAW.iterdir() if x.is_dir()):
        cls = "spam" if "spam" in folder.name else "ham"
        for f in sorted(folder.rglob("*")):
            if not f.is_file() or f.name == "cmds":
                continue
            r = _parse_mail(f)
            if r is None:
                continue
            key = r["message_id"] or hashlib.md5(f.read_bytes()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            r["label_3"], r["release"] = cls, folder.name
            rows.append(r)
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d.date, errors="coerce", utc=True)
    d["label"] = (d.label_3 == "spam").astype(int)
    d["corpus"] = "spamassassin_raw"
    return d


def load_personal_mail(limit=None, person=None):
    """YOUR OWN MAILBOX, exported with Google Takeout. Private data.

    NEVER COMMIT THIS DIRECTORY, never paste its contents into a report, an
    artifact or an issue, and do not train anything you intend to publish
    weights for without re-reading that sentence. It is here because it is the
    only source of 2022-2026 ham that is era-matched AND genre-matched (real
    HTML, real brand styling, real Indian transactional mail) - the thing
    lore.kernel.org and a 2002 corpus both fail to be.

    Export: takeout.google.com -> Mail -> pick labels -> .mbox per label.
    Drop the .mbox files in Datasets/personal/. One file per label; the filename
    is the label, so `Spam.mbox` and `Trash.mbox` become spam and everything
    else becomes ham.

    Layout. One person: drop the .mbox files straight in Datasets/personal/ and
    they are tagged person="self". Several: one folder per person,
    Datasets/personal/<name>/*.mbox, and `person` carries the folder name. That
    column is the whole point of collecting more than one mailbox - it lets the
    ham side be held out BY PERSON, which is the only way to find out whether a
    2.6% false-positive rate is a real property or one inbox memorised.

    `limit` caps messages PER MAILBOX, taken evenly spaced across each box rather
    than from the front: mbox order is oldest-first, so the first N would be the
    2020 mail and this corpus exists precisely for its 2022-2026 half.

    Profiled 2026-08-22 (42,054 messages before Message-ID dedup; Gmail exports a
    message once per label, so Inbox and the Category boxes overlap heavily):
      Inbox 21,253 | updates 11,559 | promotions 5,255 | social 3,824 |
      Archived 126 | Spam 37
    ~100% carry Authentication-Results AND DKIM-Signature - the first legitimate
    mail here with real SPF/DKIM - and ~100% is HTML, mean 415 words against
    phishing_pot's 410, which is the genre match no public ham corpus offers.

    SPAM IS 37 MESSAGES. Gmail purges spam after 30 days, so this export is not a
    source of modern spam and never will be without a standing re-export.

    Gmail's Spam label is GOOGLE'S judgement, not yours. Training on it teaches
    the model to imitate Gmail's filter, which is a legitimate thing to do and a
    dishonest thing to leave unsaid - say it in the report. The ham side has no
    such problem: mail you kept is mail you wanted.
    """
    rows = []
    # Flat *.mbox is one person; <name>/*.mbox is several, tagged by folder,
    # which is what makes leave-one-person-out possible.
    for f in sorted(PERSONAL.glob("*.mbox")) + sorted(PERSONAL.glob("*/*.mbox")):
        who = f.parent.name if f.parent != PERSONAL else "self"
        if person and who != person:      # skip the key scan entirely - one
            continue                      # mailbox here is 3.2 GB
        label = "spam" if f.stem.lower() in {"spam", "trash", "junk"} else "ham"
        box = mailbox.mbox(str(f))
        # keys() builds the index by scanning for "From " lines; only the keys we
        # actually take get parsed. Iterating the mailbox instead parses all
        # 42,054 messages across 3.2 GB every call, which is minutes per run.
        keys = list(box.keys())
        if limit and len(keys) > limit:
            step = len(keys) / limit          # evenly spaced, not the first N -
            keys = [keys[int(i * step)] for i in range(limit)]   # mbox is oldest-first
        for k in keys:
            r = _mail_row(box[k], f.name)
            if r and len(r["text"]) > 5:
                r["label_3"], r["mbox"] = label, f.stem
                r["person"] = who
                rows.append(r)
    if not rows:
        raise FileNotFoundError(
            f"no .mbox in {PERSONAL} - export from takeout.google.com first")
    # Message-ID dedup runs per person, not globally: two people on the same
    # mailing list receive the SAME Message-ID, and dropping it globally would
    # silently delete one of them - exactly the row a cross-user test needs.
    d = pd.DataFrame(rows)
    # Gmail applies a Category label to spam TOO, so a message sits in both
    # Spam.mbox and Category updates.mbox with one Message-ID. Alphabetical
    # keep="first" therefore relabelled EVERY spam message ham: 36 of 36 here,
    # 36 of 37 in the other mailbox. Sort spam to the front so the spam label
    # wins the dedup.
    d = d.sort_values("label_3", key=lambda c: c.ne("spam"), kind="stable")
    d = d.drop_duplicates(subset=["person", "message_id"], keep="first")
    d["date"] = pd.to_datetime(d.date, errors="coerce", utc=True)
    d["label"] = (d.label_3 == "spam").astype(int)
    d["corpus"] = "personal"
    return d.reset_index(drop=True)


def load_sms():
    """3-class: ham / spam / smishing. label = 1 for smishing only."""
    parts = []
    for f in ["Dataset_10191.csv", "Dataset_5971.csv"]:
        d = pd.read_csv(DATA / "smishing" / f)
        d.columns = [c.lower() for c in d.columns]
        d["source"] = f.replace(".csv", "")
        parts.append(d[["label", "text", "url", "email", "phone", "source"]])

    uci = pd.read_csv(DATA / "smishing" / "SMSSpamCollection", sep="\t",
                      names=["label", "text"], encoding="latin-1")
    uci["source"] = "uci"
    parts.append(uci)  # ham/spam only, no smishing class

    # Indian corpus: 8 scam/legit topic pairs (e-challan, KYC, digital arrest...).
    # The other three are US/UK; a model trained on them alone scores 0.509 acc
    # here and flags 58% of legit Indian transactional SMS. Its benign rows are
    # real service messages, not bulk marketing, so they map to ham, not spam.
    # platform_found is NOT read: Call/Telegram are 100% fraud in it, a free label.
    ind = pd.read_csv(INDIA / "india_fraud_detection_FINAL.csv")
    ind["text"] = ind.pop("message_text")
    ind["label"] = np.where(ind.label == 1, "smishing", "ham")
    ind["source"] = "india"
    parts.append(ind[["label", "text", "source"]])

    # The spam class had no Indian rows at all: every corpus above supplies
    # Indian ham and Indian smishing, and Indian bulk marketing exists in none
    # of them. So "offer + urgency + CTA + link" was a clean smishing feature -
    # explicit CTAs split 92 smishing / 4 spam / 1 ham - and a real Subway
    # coupon scored smishing 0.631. Synthetic, authored here, declared as such:
    # see _gen_promo.py for how and why. It carries its own `group` column.
    parts.append(pd.read_csv(INDIA / "promo_sms.csv"))

    # One real Indian inbox, 943 messages off a handset, added 2026-09-23. Every
    # corpus above is a dataset; this is traffic, and it is the only place the
    # model's real errors live. Public SMS corpora carry almost no legitimate
    # automated bank mail, so "your account has been blocked by an amount of
    # Rs.X against the UPI mandate" - an ordinary ICICI notice - read as
    # smishing at 0.86, and 7.0% of this inbox was called smishing when NONE of
    # it is. Labels come from the carrier's own DLT category suffix, not from
    # the model: -P is spam, -S/-T/-G are ham, and the access provider stamps
    # both outer parts of the header in transit, so the sender chose neither.
    # Bodies are redacted (the phone wrote them that way) and each ships a
    # de-redacted twin in the same group - see scratch/_gen_inbox_sms.py, which
    # is also where the ceilings are written down.
    inbox = INDIA / "inbox_sms.csv"
    if inbox.exists():
        parts.append(pd.read_csv(inbox))

    d = pd.concat(parts, ignore_index=True)
    d["label_3"] = d["label"].str.lower().str.strip()
    # Dataset_10191 is a rebalanced copy of 5971 -> same texts appear twice.
    d = d.drop_duplicates(subset="text").reset_index(drop=True)
    d["label"] = (d["label_3"] == "smishing").astype(int)
    # 3200 Indian rows come from ~1400 templates, so a random split puts near-twins
    # on both sides and reads 0.99 f1 off a TF-IDF baseline. Group on the opening
    # six words and split by group instead. Other corpora are one row per group.
    auto = np.where(d.source == "india",
                    d.text.str.lower().str.replace(r"[^a-z ]", "", regex=True)
                    .str.split().str[:6].str.join(" "), d.text)
    # ...except where a corpus ships its own grouping, which promo_sms.csv does.
    # Opening words would not work there: those rows start with the brand, so one
    # template scatters across 44 groups and its twins land on both sides.
    d["group"] = d.get("group", pd.Series(index=d.index, dtype=object)).fillna(
        pd.Series(auto, index=d.index))
    return d[["text", "label", "label_3", "url", "email", "phone", "source", "group"]]


def load_urls():
    """PhiUSIIL URLs. Only the raw URL string - the page-content columns
    (LineOfCode, NoOfJS, ...) need a live fetch, so they cannot be used at
    scoring time. URLSimilarityIndex is near-perfect and leaks; also dropped.
    """
    d = pd.read_csv(DATA / "URL" / "PhiUSIIL_Phishing_URL_Dataset.csv",
                    usecols=["URL", "label"])
    d["label"] = 1 - d["label"]  # their 1 = legit -> our 1 = phishing
    return d.rename(columns={"URL": "url"})


def load_url_feed(n_benign=50_000):
    """Live holdout: OpenPhish (fresh phishing) vs Tranco (top benign domains).
    Newer than PhiUSIIL, so it works as the temporal test set.
    """
    bad = pd.read_csv(DATA / "URL" / "openphish.txt", names=["url"])
    bad["label"] = 1
    good = pd.read_csv(DATA / "URL" / "tranco_N2KLW.csv",
                       names=["rank", "url"], nrows=n_benign)
    good["url"] = "https://" + good["url"]
    good["label"] = 0
    return pd.concat([bad, good[["url", "label"]]], ignore_index=True)


def load_transcripts():
    """Authored call transcripts (§5). 25 scam / 25 legitimate, written for this
    project because no public vishing transcript corpus exists.

    This is an EVALUATION set, not training data - 50 samples cannot train
    anything, and they were written by the same person who wrote the rules, so
    treating them as training data would measure the author, not the channel.
    The legitimate half deliberately contains the hard cases: a gas delivery man
    who genuinely needs your OTP, a real bank call that mentions PIN and CVV in
    order to say it will never ask for them, a real constable, a real customs
    notice. A set without those measures nothing.

    The dev/test split is stored in the file, not derived here: a split that
    lives in code is a split that quietly changes when someone edits the code.
    Tune on dev, score test once.
    """
    d = pd.read_json(DATA / "vishing" / "transcripts.jsonl", lines=True)
    d["label"] = (d["label"] == "scam").astype(int)
    return d


SGD = DATA / "vishing" / "sgd"

# The legitimate transactional domains - the ones whose vocabulary overlaps a
# scam call. A model that has only ever seen restaurant bookings as "legitimate"
# has still never seen a real bank call.
SGD_FINANCIAL = ("Banks_1", "Banks_2", "Payment_1",
                 "Services_1", "Services_2", "Services_3", "Services_4")


def load_sgd(financial_only=False):
    """Schema-Guided Dialogue -> one row per dialogue, flattened to plain text.

    22,825 legitimate task-oriented conversations (Google, CC BY-SA 4.0), of
    which ~4,200 are banking, payment or service calls. These are the negative
    class the voice channel has never had: someone legitimately discussing an
    account, a balance, a transfer, an appointment.

    **Speaker labels are dropped on purpose.** Whisper emits an undiarized
    stream of words, so a training negative that carries USER:/SYSTEM: markers
    would teach the model to key on turn structure - a cue that simply does not
    exist at serving time, and one that would separate these dialogues from
    monologue robocalls for free. Flattening keeps the model on the words.
    """
    import json
    rows = []
    for split in ("train", "dev", "test"):
        for f in sorted((SGD / split).glob("dialogues_*.json")):
            for d in json.load(open(f, encoding="utf-8")):
                rows.append({"id": f"{split}/{d['dialogue_id']}",
                             "services": ",".join(d["services"]),
                             "text": " ".join(t["utterance"] for t in d["turns"])})
    d = pd.DataFrame(rows)
    if financial_only:
        d = d[d.services.str.split(",").apply(
            lambda ss: any(x in SGD_FINANCIAL for x in ss))].reset_index(drop=True)
    d["label"] = 0                     # every SGD dialogue is legitimate
    return d


ASV21 = DATA / "vishing" / "ASVspoof2021_LA_eval"
ASV21_KEYS = DATA / "vishing" / "keys" / "LA" / "CM" / "trial_metadata.txt"


def load_spoof21(phase="eval", codec=None):
    """ASVspoof 2021 LA eval -> utt, path, codec, attack, label (1 = spoof).

    The 2019 corpus is clean studio audio; this one is the SAME kind of attacks
    pushed through real transmission channels, and the `codec` column says which:
    none, pstn, gsm, ulaw, alaw, opus, g722. That makes it the labelled
    ground-truth test of `spoof_model.telephony()` - alaw and ulaw are exactly
    the companding the simulation applies, and pstn/gsm are the real thing
    rather than anyone's model of it.

    An earlier note in this project claimed this set could not be scored because
    the tar ships only a trial list. That was wrong: the keys are a free 21 MB
    download from asvspoof.org and are now in `keys/`.

    phase: "eval" (148,176 trials), "progress", "hidden", or None for all.
    """
    cols = ["speaker", "utt", "codec", "tx", "attack", "key", "trim", "phase"]
    d = pd.read_csv(ASV21_KEYS, sep=" ", names=cols, usecols=range(8))
    if phase:
        d = d[d.phase == phase]
    if codec:
        d = d[d.codec == codec]
    d = d.reset_index(drop=True)
    d["path"] = str(ASV21 / "flac") + "/" + d.utt + ".flac"
    d["label"] = (d.key == "spoof").astype(int)
    return d[["utt", "path", "codec", "attack", "label"]]


ROBOCALL = DATA / "vishing" / "robocall"


def load_robocalls(lang="en"):
    """FTC Project Point of No Entry robocalls -> id, path, transcript, label.

    1,432 real recordings (1,378 English) that the FTC collected from the cease
    and desist letters it sent to gateway carriers, i.e. calls that actually
    reached American phones and actually defrauded people. Audio is public
    domain, already WAV pcm_s16le at 16 kHz, so it needs no conversion.

    THERE IS NO BENIGN HALF. Every row is label 1, so this yields a CATCH RATE,
    never an AUC - the same shape as Nazario and Nigerian_Fraud on the email
    side. A number computed here says what fraction of real scam calls the
    channel fires on; it says nothing about false alarms, and quoting it as
    accuracy would be a lie by omission.

    Two more limits worth stating before any number is read off this:
      - These are ROBOCALLS, one-sided recorded messages, not live two-party
        vishing. Counted, not assumed: of 1,353, exactly 3 transcripts hold a
        real human dialogue, and those 3 are mixed mono with no speaker labels.
        The 21 files carrying a `_right.wav` recipient channel are robot vs
        robot - the right channel is the honeypot's own challenge bot ("Robot
        Ready. Please say your name..."), never a person. So §6's rolling
        escalation and split_stereo() are both untested by this corpus; only
        the content layer is.
      - They are US scams (auto warranty, IRS, Social Security, Amazon order)
        and this project's authored eval set is Indian (bank OTP, KYC, digital
        arrest). Off-distribution on purpose: that is the point of testing here.

    The `transcript` column is the dataset's own Whisper-medium output, which
    makes it a free reference to diff this project's faster-whisper-small
    against - a second opinion on ASR that costs nothing to obtain.
    """
    d = pd.read_csv(ROBOCALL / "metadata.csv")
    if lang:
        d = d[d.language == lang]
    # 25 recordings appear twice: the FTC sent letters to two different gateway
    # carriers for the same campaign audio, so one wav carries two case_details.
    # Deduplicate or those 25 are counted twice in any rate. Same trap as
    # Dataset_10191 on the SMS side - a corpus that ships its own copies.
    d = d.drop_duplicates("file_name").reset_index(drop=True)
    d["path"] = str(ROBOCALL) + "/" + d.file_name
    d["id"] = d.file_name.str.split("/").str[-1].str.replace(".wav", "", regex=False)
    d["label"] = 1                     # every row is a confirmed robocall
    return d[["id", "path", "transcript", "language", "case_details", "label"]]


ASV = DATA / "vishing" / "LA"
SPOOF_PROTO = {"train": "ASVspoof2019.LA.cm.train.trn.txt",
               "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
               "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}


def load_spoof(split="train"):
    """ASVspoof 2019 LA -> utt id, flac path, attack id, label (1 = spoof).

    The attack column matters: train and dev share attacks A01-A06, eval uses
    A07-A19, which are algorithms the model has never heard. That is the split
    the corpus was built to enforce, and it is the same rule as everywhere else
    in this project - never test on what you trained on.
    """
    cols = ["speaker", "utt", "_", "attack", "key"]
    d = pd.read_csv(ASV / "ASVspoof2019_LA_cm_protocols" / SPOOF_PROTO[split],
                    sep=" ", names=cols)
    d["path"] = str(ASV / f"ASVspoof2019_LA_{split}" / "flac") + "/" + d.utt + ".flac"
    d["label"] = (d.key == "spoof").astype(int)
    return d[["utt", "path", "speaker", "attack", "label"]]


def load_korccvi():
    """KorCCVi's 706 real Korean voice-phishing calls (FSS), machine-translated
    to English by `scratch/_korccvi.py` -> id, transcript. All label 1: a CATCH
    RATE only - the corpus's legit half comes from another source (its own
    authors' warning), so it is not loaded. Not redistributable; rates only."""
    t = json.load(open(DATA / "vishing" / "korccvi" / "translated.json"))
    return pd.DataFrame({"id": list(t), "transcript": list(t.values())})


ASV5 = DATA / "vishing" / "asvspoof5"


def load_spoof5(split="train"):
    """ASVspoof 5 (2024, Zenodo 14498691) -> utt, path, speaker, codec, attack,
    label (1 = spoof). Only rows whose flac is on disk.

    Crowdsourced MLS-English speech from ~2,000 speakers in ordinary rooms, not a
    studio - the gap the 2019 corpus leaves. Attacks never cross splits: train
    A01-A08, dev A09-A16, eval A17-A32 (adversarial ones included), and only eval
    carries codecs (C01-C11: opus, amr, speex, encodec, mp3, m4a, ... plus
    narrowband variants). Just one of the ten eval tars is downloaded, so eval is
    a ~10% slice; filtering on disk keeps a missing file from looking like a
    label.
    """
    tsv = {"train": "ASVspoof5.train.tsv", "dev": "ASVspoof5.dev.track_1.tsv",
           "eval": "ASVspoof5.eval.track_1.tsv"}[split]
    folder = ASV5 / {"train": "flac_T", "dev": "flac_D", "eval": "flac_E_eval"}[split]
    cols = ["speaker", "utt", "gender", "codec", "q", "seed", "tag", "attack", "key", "tmp"]
    d = pd.read_csv(ASV5 / tsv, sep=" ", names=cols)
    have = {p.stem for p in folder.glob("*.flac")} if folder.exists() else set()
    d = d[d.utt.isin(have)].reset_index(drop=True)
    d["path"] = str(folder) + "/" + d.utt + ".flac"
    d["label"] = (d.key == "spoof").astype(int)
    return d[["utt", "path", "speaker", "codec", "attack", "label"]]


if __name__ == "__main__":
    e, s, u, f = load_email(), load_sms(), load_urls(), load_url_feed()
    t = load_transcripts()
    print(f"transcripts {len(t)}  scam={t.label.mean():.0%}  "
          f"types={t.type.nunique()}  words={t.text.str.split().str.len().mean():.0f} avg")
    assert len(t) == 50 and t.label.sum() == 25
    assert t.id.is_unique and t.text.str.split().str.len().min() > 25
    # .to_numpy() first: pandas 3 stopped broadcasting a nested list through
    # DataFrame.eq elementwise, so the old form silently compared False on data
    # that was in fact correct.
    assert t.groupby("split").label.agg(["size", "sum"]).to_numpy().tolist() == \
        [[26, 13], [24, 12]], "split must stay balanced"
    if POT.exists():
        pot = load_phishing_pot()
        assert len(pot) > 8_000 and pot.label.eq(1).all()
        assert pot.spf.notna().mean() > 0.9, "auth headers ARE the point of this corpus"
        assert pot.date.dt.year.between(2022, 2026).mean() > 0.95
        # 6 subjects stay encoded on purpose: their encoded-word is padded with
        # hundreds of junk bytes (Bayesian poisoning) so it is no longer valid.
        assert pot.text.str.startswith("=?").sum() <= 10, "RFC2047 decoding broke"
        assert pot.urls.mean() > 0.5, "href links are being stripped before counting"
        print(f"pot    {len(pot):>7,}  spf pass={pot.spf.eq('pass').mean():.1%}")

    if SA_RAW.exists():
        sa = load_spamassassin_raw()
        assert sa.label_3.value_counts().to_dict() == {"ham": 4211, "spam": 1901},             "release tarballs missing, or Message-ID dedup changed"
        # exactly 2 spam carry no From header at all - which is a signal, not a gap
        assert (sa.sender.str.len() == 0).sum() == 2, "From headers went missing"
        assert sa.spf.isna().all(), "2002 mail cannot carry SPF"
        assert (sa.text.str.len() > 5).all()
        print(f"sa_raw {len(sa):>7,}  ham={(sa.label == 0).sum():,} spam={sa.label.sum():,}")

    for name, d in [("email", e), ("sms", s), ("url", u), ("feed", f)]:
        print(f"{name:6} {len(d):>7,}  malicious={d.label.mean():.1%}")
    print("sms 3-class:", s.label_3.value_counts().to_dict())

    print("email 3-class:", e.label_3.value_counts().to_dict())
    assert e.label.isin([0, 1]).all() and len(e) > 50_000
    assert e.label_3.isin(["ham", "spam", "phishing"]).all()
    assert (e[e.label_3 == "phishing"].corpus.isin(PHISH_CORPORA)).all()
    assert s.label_3.isin(["ham", "spam", "smishing"]).all()
    assert u.label.mean() > 0.4, "phiusiil flip is wrong"
    assert f.label.sum() > 100 and f.label.mean() < 0.1
    print("ok")


def load_harpervalley():
    """HarperValleyBank -> id, transcript, label(=0). The legitimate half this
    channel never had.

    1,446 real recorded human-human banking calls (23 h, 8 kHz telephony, public
    domain, Gridspace/Stanford CS224S). Every row is label 0 - a benign corpus
    with no scam half, the mirror image of load_robocalls(), so it yields a FALSE
    ALARM RATE and never an accuracy.

    Transcripts come from `_hvb_transcribe.py`, i.e. OUR faster-whisper pass over
    the two speaker channels mixed to mono. The corpus ships human transcripts
    and they are deliberately unused: `speech_model` reached in-source AUC 1.000
    and real-call 0.576 by learning "ASR output = scam" when the negatives were
    hand-written text. Negatives have to arrive through the same ASR as the
    positives or the shortcut comes straight back.

    CAVEAT THAT MUST TRAVEL WITH THE NUMBER: the speakers are crowd workers
    performing assigned banking tasks, so this is ACTED legitimate speech, the
    same weakness as the 15 legitimate YouTube calls. It is 96x bigger and it is
    the exact register the content layer fails on (balance, cards, transfers),
    but it is not a recording of real customers.
    """
    p = DATA / "vishing" / "harpervalley" / "transcripts.json"
    d = pd.DataFrame(sorted(json.load(open(p)).items()),
                     columns=["id", "transcript"])
    d["label"] = 0
    return d


def load_scambait():
    """Scambait calls -> id, transcript, label(=1), channel, dur. The LIVE
    two-party scam half the FTC robocalls are not.

    Recorded scam calls from the six public YouTube channels NDSS 2024
    ("An analysis of scam baiting calls", arxiv 2307.01965) built its 341-call /
    90 h set from. That dataset was never released, so `scratch/_scambait.py`
    reproduces the recipe: same channels, a title-level vetting filter, and OUR
    faster-whisper over the audio - identical provenance to the 27 calls and to
    HarperValley, which is the property every corpus this channel rejected lacked.

    WHY IT EXISTS. Of 1,353 FTC robocalls the 7B judge finds an act in 22%, and
    credential + payment + remote together fire on 17. A robocall is the bait
    ("press 1 to speak to an agent") and the recording stops before the ask, so
    the acts the taxonomy was built for have almost no examples. These calls ARE
    the ask.

    TWO CAVEATS THAT MUST TRAVEL WITH ANY NUMBER FROM THIS:
      - The victim half is PERFORMED. Scam-baiters are not real victims and they
        steer the call. Usable here only because act_model scores what the CALLER
        does and the scammer half is genuine - say it, do not let it be found.
      - Every row is label 1, so like load_robocalls() this yields a CATCH RATE
        and never an accuracy. The negatives stay HarperValley.
    """
    root = DATA / "vishing" / "scambait"
    t = json.load(open(root / "transcripts.json"))
    d = pd.DataFrame(json.load(open(root / "index.json")))
    d = d[d.id.isin(t)].copy()
    d["transcript"] = d.id.map(t)
    d["label"] = 1
    return d[["id", "transcript", "label", "channel", "dur", "title"]]
