"""FastAPI sidecar - the web app's only door into the Python models.

Next.js cannot import torch, so every route handler in web/ proxies to here.
Nothing in this file re-implements detection: it wires the existing modules
together exactly the way fusion.main() and voice_model.__main__ already do, so
a number shown on the dashboard is the same number the eval scripts print.

Run:  python api.py            (127.0.0.1:8000, --reload not wanted: models are
                                loaded once by lru_cache and a reload drops them)
"""
import data  # noqa: F401  # must precede torch (Windows libiomp), see sms_model
import base64
import datetime as dt
import os
import tempfile
from email.utils import getaddresses

import attachments
import behaviour
import email_model
import fusion
import safe_browsing
import sms_model
import store
import thread
import url_model
import voice_model
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Phish/Smish/Vish detection")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

# Per-user email sender profiles (§7), kept in Postgres by store.py and cached
# here for the life of the process. A dict alone reset every restart, so a real
# mailbox never reached MIN_PROFILE and the channel abstained from behaviour
# scoring forever. If the database is unreachable this falls back to the old
# in-process dict: fusion already treats None as "no profile", so an empty store
# degrades honestly instead of lying.
_STORES = {}


def _store(user):
    user = user or "anon"
    if user not in _STORES:
        try:
            _STORES[user] = store.load(user)
        except Exception as e:
            print(f"sender profiles unavailable ({e}) - scoring from an empty store")
            _STORES[user] = behaviour.new_store()
    return _STORES[user]


def _links(text, mail=False, owner=None, attached=()):
    """-> [(url, phishing prob)] sorted worst-first. One batched url_model call.
    `attached` = attachments.scan's (url, filename) pairs: links inside a PDF /
    HTML / calendar attachment, scored the same way and tagged with the file.

    Reads only links VISIBLE in the scored text. On HTML mail that misses every
    href, and that is deliberate, measured 2026-10-09 (scratch/_href_eval.py,
    7,846 personal ham / 2,000 phishing_pot through fusion.fuse):
      links scored          ham not SAFE   phishing not SAFE
      visible text only           7.2%              96.3%
      + every href host          40.4%              97.7%
      + off-domain hrefs         39.2%              97.4%
    Tracking hosts look odd to a domain-lexical model, and once a URL score
    exists fusion gives it ~0.375 of the risk. +1.4 points of catch for 33
    points of false alarms. data.link_hosts still feeds behaviour.trusted.
    """
    seen = thread.find_urls(text)
    where = {u: kind for u, kind in attached if u not in seen}
    urls = list(dict.fromkeys(seen + list(where)))  # a link repeated is one link
    if not urls:
        return []
    scores = url_model.score(urls, mail=mail, owner=owner)   # mail: url_xgb_v4 + layers, see url_model
    # Website-only known-bad layer: a listed link is raised, an unlisted one is
    # left alone ("not listed" is not "safe"). Abstains without key or network.
    listed = safe_browsing.check(urls)
    return sorted(({"url": u, "score": round(max(float(s), safe_browsing.LISTED_RISK)
                                             if u in listed else float(s), 3),
                    **({"listed": listed[u]} if u in listed else {}),
                    **({"attachment": where[u]} if u in where else {})}
                   for u, s in zip(urls, scores)),
                  key=lambda d: -d["score"])


def _explain(link, **kw):
    """url_model.explain for the worst link, unless Google listed it: that is
    the stronger, plainer reason. Worded for a stored scan read later - the
    terms allow a warning only on data under 30 minutes old."""
    if link.get("listed"):
        why = (f"Google Safe Browsing listed it as {link['listed']} when it was scanned "
               "(advisory provided by Google)")
        return link["score"], f"link is dangerous: {why}", [why]
    return url_model.explain(link["url"], **kw)


class SmsIn(BaseModel):
    text: str
    sender: str = ""
    # None = we do not know, and the rule abstains. Only a caller with the
    # phone's own message store (§10 gateway) can answer this; the web form
    # cannot, and answering True there claimed "no prior history" about every
    # sender the store had merely never been told about.
    first_contact: bool | None = None
    # Whose store to check the sender against - same per-user profiles the email
    # route builds. See EmailIn.user_id.
    user_id: str = ""
    # When the message was SENT, if that is actually known - a screenshot's
    # timestamp read by OCR, or an ingested message's own metadata. None on a
    # typed message, and then the odd-hour rule abstains instead of scoring the
    # hour the user happened to be at the keyboard.
    sent_at: dt.datetime | None = None
    # True only when a CARRIER supplied the header (§10 gateway mode). A sender
    # typed into the web form attests nothing, so it defaults to False and the
    # de-escalation gate below stays disarmed on every path that exists today.
    verified_sender: bool = False


@app.post("/sms")
def sms(x: SmsIn):
    label, prob, why, terms = sms_model.explain(x.text)
    probs = sms_model.score([x.text])[0]
    links = _links(x.text)
    url = links[0]["score"] if links else None

    beh, tags = None, []
    if x.sender:
        # The lookup still happens - a sender already in this user's store clears
        # itself - but a cold store yields None and the first-contact rules
        # abstain instead of asserting a history nobody has.
        prof = _store(x.user_id)
        known = behaviour.known_sender(prof, x.sender, channel="sms")
        first = x.first_contact if x.first_contact is not None else             (None if known is None else not known)
        f = behaviour.sms_features(x.sender, x.text, first_contact=first,
                                   has_url=bool(links),
                                   hour=x.sent_at.hour if x.sent_at else None)
        beh, tags = behaviour.sms_score(f)
        if first is None:
            tags = tags + [f"sender history not checked - only "
                           f"{behaviour.profile_count(prof, 'sms')} of "
                           f"{behaviour.MIN_PROFILE} SMS senders known here"]

        # Persist the sender, or the history check above can never become true:
        # nothing else in this app writes an SMS sender. On a phone (§10) this
        # comes from the message store instead, which is the difference between
        # "you have a relationship with this sender" and "we have scanned it
        # before" - the weaker claim this path can make.
        key = behaviour.sender_key(x.sender, "sms")
        sent = x.sent_at.astimezone() if x.sent_at else None
        behaviour.update(prof, key, "me", sent, behaviour.style_of(x.text))
        try:
            store.save(x.user_id or "anon", key, prof,
                       when=sent or dt.datetime.now().astimezone(),
                       has_url=bool(links), display=x.sender, domain="")
        except Exception as e:
            print(f"sender profile not saved ({e})")

    # §8 act gate (see fusion.gate_by_act). Content scores topic; the act layer
    # scores whether anything was actually asked for. A bank's own fraud warning
    # is maximally on-topic and asks for nothing, so it gets damped back down.
    evidence = voice_model.evidence_risk(x.text)
    probs = fusion.gate_by_act(probs, evidence, malicious_idx=2)
    if evidence == 0:
        tags = tags + ["message contains no request, threat or payment demand"]

    if url is not None:
        _, link_why, links[0]["why"] = _explain(links[0])
    reasons = [why] + ([link_why] if url is not None else []) + tags
    out = fusion.fuse(probs, url=url, behaviour=beh, reasons=reasons,
                      malicious_idx=2)  # smishing column, §8

    # §8 de-escalation (see fusion.deescalate). Only SMS gets this: a DLT header
    # is verifiable sender identity, and email has no equivalent here because no
    # SPF/DKIM result reaches us.
    #
    # ...but only when someone other than the sender vouches for the header.
    # Passing registered() on a typed string made the gate strictly harmful:
    # measured 2026-08-21 it cleared 2 of 40 legitimate transactional SMS and 14
    # of 200 smishing, and a scammer filling in this form picks any registered
    # entity he likes. Gateway mode sets verified_sender and gets the gate back.
    out = fusion.deescalate(out, evidence, url=url,   # evidence computed above
                            trusted_sender=x.verified_sender
                            and behaviour.registered(x.sender))
    return {**out, "channel": "sms", "label": label, "label_prob": round(prob, 3),
            "evidence": round(evidence, 3), "terms": terms,
            "classes": dict(zip(sms_model.LABELS, [round(float(p), 3) for p in probs])),
            "links": links}


class EmailIn(BaseModel):
    subject: str = ""
    body: str
    sender: str = ""
    # Whose mailbox this is. Profiles are per user (spec §7); the web app passes
    # its signed-in user id, and anything else shares the "anon" profile.
    user_id: str = ""
    # The Date: header, when the mail arrived through something that carries one
    # (the Gmail push of the Phase-2 plan). None for a pasted mail - see SmsIn.
    sent_at: dt.datetime | None = None
    # SPF/DKIM/DMARC results from the receiving server's Authentication-Results.
    # Only /email/raw (Gmail polling) has headers; a pasted or screenshotted
    # mail sends None and the auth check abstains.
    spf: str | None = None
    dkim: str | None = None
    dmarc: str | None = None
    # Every host a link in the raw mail points at (data.link_hosts), hrefs
    # included. None = no raw mail, links unknown, so the trust rule cannot arm.
    link_hosts: list[str] | None = None


@app.post("/email")
def email(x: EmailIn):
    return _email(x)


def _email(x: EmailIn, att=None):
    """att = attachments.scan(raw), Gmail path only: pasted mail has no files."""
    text = f"{x.subject}\n{x.body}".strip()
    probs, persuasion = email_model.score([text])
    probs, persuasion = probs[0], persuasion[0]
    raw = probs
    # The sender's own links are judged by their domain alone - only when DMARC
    # passed on an established domain (url_model.sender_owner). Pasted mail has
    # no DMARC, so it abstains.
    owner = url_model.sender_owner(x.sender, x.dmarc)
    att = att or {"count": 0, "bad": [], "urls": []}
    links = _links(text, mail=True, owner=owner, attached=att["urls"])
    url = links[0]["score"] if links else None
    # A file that runs code or installs settings is the payload itself: it fills
    # the link slot like a link Google has listed (scratch/_attach_eval.py).
    if att["bad"]:
        url = max(url or 0.0, safe_browsing.LISTED_RISK)

    beh, tags, trust = None, [], []
    if x.sender:
        # .astimezone() keeps the LOCAL hour that hour_dev profiles while making
        # `now` aware - a naive one cannot be subtracted from the timestamps a
        # hydrated profile carries, which is what burst_1h does.
        prof, now = _store(x.user_id), dt.datetime.now().astimezone()
        # `sent` is the message's own clock and `now` is ours. hour_dev and
        # burst_1h are statements about when the SENDER sends, so they get
        # `sent` and abstain when it is None; three pasted mails in a demo are
        # not a burst, and 1am at the keyboard is not 1am in the mailbox. Only
        # the bookkeeping timestamps below may fall back to `now`, where the
        # question really is when we saw it.
        sent = x.sent_at.astimezone() if x.sent_at else None
        auth = behaviour.auth_fail(x.spf, x.dkim, x.dmarc)
        # Gmail path only (link_hosts is set by /email/raw alone): on the typed
        # web path this would be a cross-user lookup driven by browser input.
        crowd = None
        if x.link_hosts is not None:
            try:
                crowd = behaviour.crowd_reported(store.crowd(x.sender, x.user_id))
            except Exception:
                pass                  # Postgres down: the crowd abstains
        f = behaviour.features(prof, x.sender, "me", sent, text, bool(links),
                               auth, crowd)
        # §7 scores DEVIATION from a profile, so it needs one. Against a cold
        # store first_contact and new_domain are both 1 for everybody, rule_score
        # returns 0.65, and fusion's escalation gate lifts the mail to SUSPICIOUS
        # on structure alone - which made the FIRST mail from any sender
        # suspicious no matter what it said. Abstaining (None, not 0.0) is what
        # the fuser already understands and what the _STORES comment above
        # promises: behaviour forfeits its weight instead of voting an artifact.
        if behaviour.profile_ready(prof, channel="email"):
            beh, tags = behaviour.rule_score(f), behaviour.explain(f)
        else:
            tags = [f"sender behaviour not scored - only "
                    f"{behaviour.profile_count(prof, 'email')} of "
                    f"{behaviour.MIN_PROFILE} senders known for this mailbox"]
            # Auth and crowd reports judge THIS message or sender, not a
            # deviation from a profile, so a hit counts on a cold store too. A
            # clean result stays None: voting 0.0 would assert "normal sender"
            # about one we know nothing of.
            free = {k: f[k] for k in behaviour.PROFILE_FREE if f.get(k)}
            if free:
                beh = behaviour.rule_score(free)
                tags = behaviour.explain(free) + tags
        # Read before update(): "established" means mail BEFORE this one.
        if not crowd and behaviour.trusted(prof, x.sender, x.link_hosts, auth):
            probs = fusion.trust_damp(probs, True)
            trust = ["established sender (earlier mail, or one you wrote to or marked "
                     "safe): every link here stays on its own domain, so the content "
                     "score was damped"]
        display, addr, domain = behaviour.addr_parts(x.sender)
        behaviour.update(prof, addr, "me", sent, behaviour.style_of(text))
        try:
            store.save(x.user_id or "anon", addr, prof, when=sent or now,
                       has_url=bool(links), display=display, domain=domain)
        except Exception as e:
            # A scan whose profile cannot be filed is still a valid scan - the
            # web route treats a dead Postgres the same way.
            print(f"sender profile not saved ({e})")

    # The content verdict must lead the reason. Building it from the persuasion
    # head alone made a PHISHING 1.000 verdict report "no risk signal on any
    # channel" whenever no principle crossed 0.5 - §4 requires the opposite.
    if links:
        _, link_why, links[0]["why"] = _explain(links[0], mail=True, owner=owner)
        if "attachment" in links[0]:
            link_why = f"{link_why} (from the attached {links[0]['attachment']})"
    # Stored scans keep a link from inside an attachment only if it looks risky,
    # five at most: an HTML statement's unsubscribe/tracking links carry the
    # user's own address. url above was taken from the full list, as measured.
    kept = [d for d in links if "attachment" in d and d["score"] >= 0.35][:5]
    links = [d for d in links if "attachment" not in d or d in kept]
    lab = email_model.LABELS[int(probs.argmax())]
    hits = [n for n, v in zip(email_model.PRINCIPLES, persuasion) if v >= .5]
    # The model's own numbers, before any trust damp - the reason must show
    # what the content model said, and the trust line says what changed it.
    content = (f"content reads as {email_model.LABELS[int(raw.argmax())]} "
               f"(phishing {raw[2]:.2f}, spam {raw[1]:.2f})")
    reasons = [content + (f", using {' and '.join(hits)}" if hits else "")] \
        + trust + [f"attached {b['type']} file is {b['why']}" for b in att["bad"]] \
        + ([link_why] if links else []) + tags
    # The spam class is blind to the cold paid-course pitch (see the note on
    # email_model.SOLICITATION - 73/73 Gmail-spam messages score ham), so a
    # marker rule carries that genre. It moves the LABEL only: fusion scores the
    # phishing column and an unsolicited course pitch is not phishing, so risk
    # and verdict are left exactly where the model put them.
    marks = email_model.solicitation(text)
    if marks and lab == "ham":
        lab = "spam"
        reasons.append("unsolicited paid-course/internship solicitation ("
                       + ", ".join(marks) + ") - the content model read it as ham")

    out = fusion.fuse(probs, url=url, behaviour=beh, reasons=reasons,
                      malicious_idx=2)  # phishing column, §8
    # dmarc rides into scans.detail: store.crowd counts a report only on mail
    # whose From domain passed DMARC, i.e. really came from that sender.
    return {**out, "channel": "email", "label": lab, "solicitation": marks,
            "dmarc": (x.dmarc or "").lower() or None,
            "classes": dict(zip(email_model.LABELS,
                                [round(float(p), 3) for p in raw])),
            "persuasion": dict(zip(email_model.PRINCIPLES,
                                   [round(float(p), 3) for p in persuasion])),
            "links": links, "attachments": {"count": att["count"], "bad": att["bad"]}}


class RawEmailIn(BaseModel):
    # base64url RFC 822, exactly what Gmail's messages.get(format=raw) returns.
    raw: str
    user_id: str = ""


@app.post("/email/parse")
def email_parse(x: RawEmailIn):
    # The web app's "View" on a Gmail scan: the full text, read live from Gmail
    # and never stored. Parse only - viewing must not re-score the mail or file
    # its sender into the behaviour profile a second time. Laid out for reading
    # (data.readable_body), not the single line the model scores.
    return data.parse_readable(base64.urlsafe_b64decode(x.raw + "=" * (-len(x.raw) % 4)))


@app.post("/email/raw")
def email_raw(x: RawEmailIn):
    # ponytail: the whole message, attachments included, crosses the wire -
    # format=raw has no way to leave them out. format=full + walking the parts
    # in JS is the upgrade if big mail ever makes polling slow.
    raw = base64.urlsafe_b64decode(x.raw + "=" * (-len(x.raw) % 4))
    m = data.parse_raw(raw)
    if not m["text"]:
        raise HTTPException(422, "no text part")
    # The Date: header is the message's own clock, which is what EmailIn.sent_at
    # has been waiting for - hour_dev and burst_1h stop abstaining on this path.
    # Attachments are read here, in memory, and never stored (attachments.py).
    out = _email(EmailIn(body=m["text"], sender=m["sender"], user_id=x.user_id,
                         sent_at=m["date"], spf=m["spf"], dkim=m["dkim"],
                         dmarc=m["dmarc"], link_hosts=m["link_hosts"]),
                 attachments.scan(raw))
    # preview is capped here, not trusted to the caller: the full body must not
    # reach scans.detail (see saveScan's note on §10's Preview mode).
    # subject rides into scans.detail (lib/gmail.js keeps everything but
    # preview/sender there), so the Gmail page can list mail by subject.
    return {**out, "sender": m["sender"], "subject": m["subject"],
            "preview": m["text"][:200]}


class SeedEvent(BaseModel):
    sender: str
    sent_at: dt.datetime | None = None


class SeedIn(BaseModel):
    user_id: str = ""
    events: list[SeedEvent] = []
    # Raw To/Cc header values from the user's own SENT mail.
    contacts: list[str] = []


@app.post("/email/seed")
def email_seed(x: SeedIn):
    """Fold mailbox HEADERS into the profiles without scoring anything - the
    fix for "behaviour not scored - 0 of 20 senders known" on a new Gmail link.

    lib/gmail.js sends From + internalDate of recent inbox mail (metadata only,
    no bodies, so style_of gets nothing) and the To/Cc of recent Sent mail. It
    must not include mail it is about to scan, or that mail would count itself
    as its own history. Also called by the poller for each new Sent mail."""
    user, prof = x.user_id or "anon", _store(x.user_id)
    utc = dt.timezone.utc
    seeded = 0
    for e in sorted(x.events, key=lambda e: e.sent_at or dt.datetime.min.replace(tzinfo=utc)):
        display, addr, domain = behaviour.addr_parts(e.sender)
        if not addr:
            continue
        when = e.sent_at.astimezone() if e.sent_at else None
        behaviour.update(prof, addr, "me", when, None)
        store.save(user, addr, prof, when=when, display=display, domain=domain)
        seeded += 1
    addrs = [a for _, a in getaddresses(x.contacts) if "@" in a]
    addrs = sorted({behaviour.add_contact(prof, a) for a in addrs} - {""})
    if addrs:
        store.save_contacts(user, addrs)
    return {"senders": seeded, "contacts": len(addrs),
            "profile_ready": behaviour.profile_ready(prof, channel="email")}


class ForgetIn(BaseModel):
    user_id: str


@app.post("/email/forget")
def email_forget(x: ForgetIn):
    """Gmail disconnect: the web app has already deleted the contact rows; drop
    the cached store so the next scan reloads without them."""
    _STORES.pop(x.user_id, None)
    return {"ok": True}


class VoiceIn(BaseModel):
    text: str
    # The number the call came from, as the user typed it, or "withheld". Empty
    # = not given, and the caller checks abstain (behaviour.voice_features).
    caller: str = ""
    user_id: str = ""


def _risky_sms(user_id, caller):
    """store.risky_sms, only when there is a user to look up."""
    return store.risky_sms(user_id, caller) if user_id and caller else None


@app.post("/voice/text")
def voice_text(x: VoiceIn):
    """Same §8 fusion as the audio path, so both return one schema. With no
    audio the spoof head reports None and its 0.20 renormalises away - which is
    not the same as scoring the caller as confirmed human."""
    v = voice_model.score_transcript(x.text)
    beh, tags = behaviour.voice_score(behaviour.voice_features(
        x.caller, x.text, _risky_sms(x.user_id, x.caller)))
    f = fusion.fuse_voice(v["peak_risk"], behaviour=beh, reasons=[v["reason"]] + tags)
    return _voice_out({**v, **f, "semantic_risk": v["peak_risk"]}, x.text)


@app.post("/voice/file")
async def voice_file(file: UploadFile = File(...), caller: str = Form(""),
                     user_id: str = Form("")):
    # voice_model.transcribe() takes a path, so the upload has to hit disk.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(await file.read())
        # carries its own transcript
        v = voice_model.analyse(path, caller, _risky_sms(user_id, caller))
    finally:
        os.unlink(path)
    return _voice_out(v, filename=file.filename)


def _voice_out(v, transcript=None, filename=None):
    """Windows are (time, text) pairs on a 2 s hop with 15 s of context, so they
    overlap heavily - pair them with the risk curve for the chart, and take the
    transcript from analyse() rather than joining them back together."""
    win = v.pop("windows", [])
    return {**v, "channel": "voice", "filename": filename,
            "transcript": transcript or v.get("transcript", ""),
            "timeline": [{"t": round(float(t), 1), "risk": c}
                         for (t, _), c in zip(win, v.get("curve", []))]}


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
