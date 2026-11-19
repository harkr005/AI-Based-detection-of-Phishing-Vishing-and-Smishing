"""Behavioural intelligence engine (spec §7) - the project's differentiator.

Phase 1 measured the ceiling this exists to break: spam recall pins at 0.609
across three independent runs, and the missed 39% is newsletters, promo mail and
romance-scam openers - text that is content-indistinguishable from ham. What
makes them unwanted is *who sent it and when*, not what it says. No text model
fixes that. This module is the "who and when".

Principle (§7): build a rolling profile per sender, score deviation from it,
emit one behaviour_anomaly_score in [0,1]. Model per §4: Isolation Forest on the
deviation features, with a rule baseline to compare against.

Streaming by construction: features for an event are computed from the profile
state BEFORE that event is folded in. Nothing sees its own future, so the score
is what a live system would have produced. Events must arrive in time order.

ponytail: the profile store is a plain dict, not Redis/Room (§7). Same keys,
same update path; swap new_store/update when it needs to outlive the process.
Nothing else in the codebase knows how profiles are stored.
"""
import re
from collections import defaultdict

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data  # noqa: F401
import numpy as np
import pandas as pd

# Display-name impersonation: the name claims an institution the domain does not
# back up. Short list on purpose - it demonstrates the rule, it is not a brand DB.
BRANDS = ("paypal", "amazon", "apple", "microsoft", "google", "netflix", "ebay",
          "irs", "hdfc", "sbi", "icici", "axis", "kotak", "bank", "chase",
          "wellsfargo", "amex", "dhl", "fedex", "usps", "facebook", "instagram")
FREEMAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "mail.com", "yandex.com", "protonmail.com", "gmx.com", "live.com",
            "rediffmail.com", "icqmail.com", "hotpop.com", "msn.com"}

# The <...> address first: it is the one the mail came from. A bare address in
# the display name ("bank@bank.com <x@evil.tk>") is text the sender typed.
ANGLE, BARE = re.compile(r"<([^>]+)>"), re.compile(r"([^\s<>,;]+@[^\s<>,;]+)")
SENT = re.compile(r"[.!?]+")

FEATURES = ["first_contact", "new_domain", "display_mismatch", "freemail_brand",
            "hour_dev", "burst_1h", "blast", "url_first_contact", "style_dev"]


# --- profile store -----------------------------------------------------------

def new_store():
    # contacts: addresses this mailbox has SENT mail to (Gmail's SENT label).
    # Someone you wrote to is not a stranger, whether or not they ever wrote back.
    return {"p": defaultdict(lambda: {"n": 0, "hours": [], "times": [],
                                      "rcpts": set(), "style": []}),
            "seen": set(), "domains": set(), "contacts": set()}


# first_contact and new_domain are defined RELATIVE TO A PROFILE. Against an
# empty store they are not observations, they are artifacts of the store being
# empty: every sender is a first contact, every domain is new, and rule_score()
# returns 0.35 + 0.30 = 0.65 on perfectly ordinary mail - above fusion.ESCALATE,
# so the first mail from anybody gets lifted to SUSPICIOUS on structure alone.
# Measured on the CEAS window, ham is only 35.6% first_contact / 8.3% new_domain,
# which is what makes those two features worth 0.65 - but that number describes a
# WARM mailbox, not a cold one.
# ponytail: 20 is the calibration knob, not a measurement. Raise it for a noisy
# mailbox, lower it for a demo.
MIN_PROFILE = 20


def sender_key(sender, channel="email"):
    """The store key for one sender.

    SMS senders are NAMESPACED because they share the per-user profile table
    with email addresses and the two are not the same population: 20 SMS
    headers must not arm the email channel's MIN_PROFILE gate for a mailbox
    nobody has ever scanned, and vice versa.
    """
    if channel == "sms":
        return "sms:" + sid(sender).lower()
    return addr_parts(sender)[1] or str(sender).strip().lower()


def profile_ready(store, min_senders=MIN_PROFILE, channel=None):
    """True once the store holds enough senders OF THAT CHANNEL for
    first_contact/new_domain to carry information. Callers pass behaviour=None
    (or disarm fusion's escalation gate) while this is False - see api.email."""
    keys = store["p"] if channel is None else         [k for k in store["p"] if k.startswith("sms:") == (channel == "sms")]
    return len(keys) >= min_senders


def profile_count(store, channel=None):
    """How many senders of that channel the store holds - the number the UI
    quotes when it says why the sender could not be judged."""
    return sum(1 for k in store["p"]
               if channel is None or k.startswith("sms:") == (channel == "sms"))


def known_sender(store, sender, channel="email", min_senders=MIN_PROFILE):
    """-> True (we have seen this sender before), False (we have not, and the
    store holds enough senders for that to MEAN something), or None (no basis).

    "Not in our database" is not the same statement as "no prior history". A
    cold store cannot tell a stranger from a sender it was simply never told
    about, and answering True there put "no prior history with this sender" on
    every message anyone ever scanned. So the lookup still happens - a sender we
    DO know clears itself - and only the negative half needs a warm store.
    """
    if not profile_ready(store, min_senders, channel):
        return None
    return sender_key(sender, channel) in store["p"]


def update(store, sender, receiver, when, style):
    """Fold one event into the sender's rolling profile. Call AFTER features()."""
    p = store["p"][sender]
    p["n"] += 1
    if when is not None:
        p["hours"] = (p["hours"] + [when.hour])[-256:]
        p["times"] = (p["times"] + [when])[-64:]  # only a burst window is ever read
    p["rcpts"].add(receiver)
    if style is not None:      # None = seeded from headers, no body was read
        p["style"] = (p["style"] + [style])[-64:]
    store["seen"].add((receiver, sender))
    store["domains"].add(sender.rsplit("@", 1)[-1])


# --- feature extraction ------------------------------------------------------

def addr_parts(sender):
    """'Bank of America <x@evil.tk>' -> ('bank of america', 'x@evil.tk', 'evil.tk')"""
    s = str(sender or "")
    m = ANGLE.search(s) or BARE.search(s)
    email = m.group(1).lower().strip() if m else ""
    display = s[:m.start()].strip(' "\'<>').lower() if m else s.lower()
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    return display, email, domain


def style_of(text):
    """Stylometry (§7): avg sentence length + punctuation habit. Two numbers is
    enough to catch a hijacked thread; a real stylometry model is a paper of its
    own. ponytail: extend here if drift alone stops separating."""
    t = str(text or "")
    sents = [s for s in SENT.split(t) if s.strip()]
    return (len(t.split()) / max(len(sents), 1),
            sum(c in ".,!?;:" for c in t) / max(len(t), 1))


def circ_dev(hour, hours):
    """Hours are circular - 23:00 is 2h from 01:00, not 22. Plain z-scoring the
    raw hour makes every 2am mail from a 9am sender look ordinary."""
    if len(hours) < 3:
        return 0.0
    a = np.array(hours) * np.pi / 12
    mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * 12 / np.pi % 24
    dev = lambda h: min(abs(h - mean), 24 - abs(h - mean))
    return float(min(dev(hour) / max(np.mean([dev(h) for h in hours]), 1.5), 4.0) / 4.0)


AUTH_FAIL = {"fail", "softfail", "permerror"}


def auth_fail(spf=None, dkim=None, dmarc=None):
    """1.0 if the receiving server's SPF, DKIM or DMARC check FAILED, 0.0 if
    results exist and none failed, None if there is no result to read.

    Only the automated path has headers (Gmail stamps its own Authentication-
    Results); a pasted or screenshotted mail has none and ABSTAINS. A pass is
    not evidence of legitimacy - SPF passes on 53% of phishing_pot because the
    attacker owns his domain - so this only ever counts against a sender.
    scratch/_auth_eval.py: any fail = 35.8% of phishing_pot, 0.5% / 0.8% of
    ham in two personal mailboxes.
    """
    got = [str(v).lower() for v in (spf, dkim, dmarc) if v]
    return float(any(v in AUTH_FAIL for v in got)) if got else None


def add_contact(store, address):
    """Fold one address this mailbox sent mail to. Its domain stops being new."""
    a = addr_parts(address)[1]
    if a:
        store["contacts"].add(a)
        store["domains"].add(a.rsplit("@", 1)[-1])
    return a


# Second-level labels under which the registrable name sits one level deeper.
# ponytail: a hand list, not the Public Suffix List - wrong for rare SLDs
# (e.g. *.pvt.k12.ma.us); tldextract is the upgrade if a miss ever shows up.
_SLD = {"co", "com", "net", "org", "gov", "ac", "edu", "nic", "res", "gen", "ind", "firm"}


def base_domain(host):
    """'click.mailer.shopmail.co.in' -> 'shopmail.co.in', 'a.b.example.com' -> 'example.com'."""
    parts = str(host or "").lower().strip(".").split(".")
    n = 3 if len(parts) >= 3 and parts[-2] in _SLD and len(parts[-1]) == 2 else 2
    return ".".join(parts[-n:])


# How many earlier mails make a sender "established". Measured in
# scratch/_trust_eval.py - see TRUST_DAMP in fusion.py for the table.
TRUST_N = 3


def trusted(store, sender, link_hosts, auth=None, n=TRUST_N):
    """An established sender whose mail stays on its own domain.

    All three must hold: we have seen >= n mails from this exact address (or
    this mailbox wrote to it), its auth did not FAIL, and EVERY link goes to the
    sender's own registrable domain. link_hosts=None (no raw mail, so links are
    unknown) is never trusted - the rule must not override a link it cannot see.
    A compromised account passes all three; that is the documented ceiling.
    """
    _, email, domain = addr_parts(sender)
    if not email or link_hosts is None or auth == 1.0:
        return False
    p = store["p"].get(email)
    if not ((p and p["n"] >= n) or email in store["contacts"]):
        return False
    home = base_domain(domain)
    return all(base_domain(h) == home for h in link_hosts)


# Crowd reports (Gmail-layer 4). Gmail's own bar is a 0.3% report rate; ours
# adds a floor of distinct reporters so one account, or two friends, cannot
# condemn a sender. Hand-set, unmeasured: there is no multi-user report log yet.
# ponytail: CROWD_MIN/CROWD_RATE are the calibration knobs.
CROWD_MIN, CROWD_RATE = 3, 0.003


def crowd_reported(counts):
    """store.crowd()'s (reporters, receivers) -> 1.0 / 0.0, or None to abstain."""
    if counts is None:
        return None
    reporters, receivers = counts
    return float(reporters >= CROWD_MIN and reporters / max(receivers, 1) >= CROWD_RATE)


def features(store, sender, receiver, when, text, has_url, auth=None, crowd=None):
    """One event -> the §7 email feature vector, from profile state only.

    Every value lands in [0,1] so the rule baseline can add them up and the
    Isolation Forest is not dominated by whichever feature has the widest units.
    `auth` is auth_fail()'s value; None leaves the key out (abstain), and it is
    not in FEATURES because the corpora the forest fits on carry no headers.
    """
    display, email, domain = addr_parts(sender)
    p = store["p"].get(email)

    first = 0.0 if (receiver, email) in store["seen"] or email in store["contacts"] else 1.0
    new_dom = 0.0 if domain in store["domains"] else 1.0

    brand = next((b for b in BRANDS if b in display), "")
    mismatch = 1.0 if brand and brand not in domain else 0.0
    freemail_brand = 1.0 if brand and domain in FREEMAIL else 0.0

    hour_dev = circ_dev(when.hour, p["hours"]) if (p and when is not None) else 0.0

    burst = 0.0
    if p and when is not None:
        recent = sum(1 for t in p["times"] if 0 <= (when - t).total_seconds() < 3600)
        burst = min(recent / 10.0, 1.0)       # 10 mails/hour from one sender = full

    blast = min(len(p["rcpts"]) / 20.0, 1.0) if p else 0.0

    sd = 0.0
    if p and len(p["style"]) >= 3:
        hist = np.array(p["style"])
        gap = np.abs(np.array(style_of(text)) - hist.mean(0))
        sd = float(np.mean(np.minimum(gap / np.maximum(hist.std(0), [3.0, 0.01]),
                                      4.0)) / 4.0)

    f = {"first_contact": first, "new_domain": new_dom,
         "display_mismatch": mismatch, "freemail_brand": freemail_brand,
         "hour_dev": hour_dev, "burst_1h": burst, "blast": blast,
         "url_first_contact": first * float(bool(has_url)), "style_dev": sd}
    if auth is not None:
        f["auth_fail"] = auth
    if crowd is not None:
        f["crowd_reported"] = crowd
    return f


def stream(df):
    """DataFrame (sender, receiver, date, text, urls) -> feature DataFrame.

    Sorted by date first: profiles are built from the past, and an out-of-order
    stream silently makes first_contact and burst meaningless.
    """
    df = df.sort_values("date", kind="stable", na_position="last")
    store, rows = new_store(), []
    for e in df.itertuples():
        when = e.date if pd.notna(e.date) else None
        rows.append(features(store, e.sender, e.receiver, when, e.text, e.urls))
        update(store, addr_parts(e.sender)[1], e.receiver, when, style_of(e.text))
    return pd.DataFrame(rows, index=df.index)


# --- scoring -----------------------------------------------------------------

# Weights set from per-feature AUC on the CEAS training window ONLY (1-7 Aug, the
# same days the content model trained on - the 8 Aug test slice was not looked at):
#   first_contact .764  new_domain .778  url_first_contact .701   <- signal
#   display_mismatch .501  freemail_brand .500                    <- absent in 2008
#   hour_dev .238  burst_1h .317  blast .180  style_dev .231      <- INVERTED
#
# The inverted four are not broken features, they are a corpus property: CEAS
# recipients are list-subscribed accounts, so bursts, blasts, 3am delivery and
# style drift describe apache/python mailing lists, i.e. the ham. In a personal
# mailbox they flip back to §7's reading. They stay in FEATURES (the forest and
# the stacker can use the direction) and carry zero weight in the hand rule,
# because a hand rule cannot know which mailbox it is in.
# auth_fail is not from CEAS (2008 mail has no auth headers): 0.60 = fusion's
# ESCALATE, so a failed SPF/DKIM/DMARC alone lifts a content-clean mail to
# SUSPICIOUS and never higher. Likelihood ratio ~50 (35.8% vs ~0.7%, above).
# ponytail: these weights ARE the calibration knob - refit per deployment.
RULE_W = {"auth_fail": 0.60, "crowd_reported": 0.60, "first_contact": 0.35, "new_domain": 0.30, "display_mismatch": 0.30,
          "freemail_brand": 0.25, "url_first_contact": 0.20,
          "hour_dev": 0.0, "burst_1h": 0.0, "blast": 0.0, "style_dev": 0.0}

TAGS = [("crowd_reported", "reported as a scam by several other users"),
        ("auth_fail", "sender failed SPF/DKIM/DMARC - it may not come from the domain it claims"),
        ("display_mismatch", "display name claims a brand the domain does not match"),
        ("freemail_brand", "institution name sent from a free mail account"),
        ("url_first_contact", "link in a first-ever message from this sender"),
        ("burst_1h", "burst of messages from this sender in the last hour"),
        ("blast", "same sender hitting many recipients"),
        ("hour_dev", "sent well outside this sender's usual hours"),
        ("style_dev", "writing style differs from this sender's history"),
        ("new_domain", "sender domain never seen before"),
        ("first_contact", "no prior history with this sender")]


# Verdicts on THIS message or sender that need no profile of the mailbox, so a
# cold store may still score them (api.email). Never first_contact/new_domain.
PROFILE_FREE = ("auth_fail", "crowd_reported")


def rule_score(f):
    """The baseline §4 asks the Isolation Forest to beat. Same features, weights
    set by hand from §7's own ordering (impersonation and first-contact-with-a-
    link are the two it calls out). ponytail: hand weights are the calibration
    knob - refit them the day a labelled event log exists."""
    s = sum(f[k] * v for k, v in RULE_W.items() if k in f)   # absent = abstained
    return np.clip(s, 0, 1).to_numpy() if isinstance(f, pd.DataFrame) else float(np.clip(s, 0, 1))


def explain(f):
    """Rule tags - §4 requires a human reason on every verdict."""
    hard = {"auth_fail", "crowd_reported", "display_mismatch", "freemail_brand", "url_first_contact",
            "first_contact", "new_domain"}
    tags = [t for k, t in TAGS if k in f and f[k] >= (1.0 if k in hard else 0.5)]
    return tags[:3] or ["nothing unusual about this sender"]


class Behaviour:
    """Isolation Forest over the §7 features (§4: unsupervised, zero-day
    resistant). It never sees an attack label, so a campaign it was never
    trained on is not a special case for it.

    Fit it on NORMAL traffic, not on everything. Measured: fitting on the whole
    CEAS stream gives AUC 0.221 - worse than chance, and correctly so, because
    56% of that stream is spam, so "first contact from a new domain" IS the
    majority pattern and the forest learns to call ordinary mail the anomaly.
    Fitting the same forest on the benign half of the same window gives 0.595.
    §7 says "score deviation from a profile"; the profile has to be of normal.
    """

    def __init__(self):
        self.iso = self.lo = self.hi = None

    def fit(self, feats, normal=None):
        """normal: boolean mask of known-benign rows (mail the user never
        reported). Defaults to the whole window, which is only right when the
        stream really is mostly benign - a real mailbox, not a spam corpus."""
        from sklearn.ensemble import IsolationForest
        base = feats if normal is None else feats[normal]
        self.iso = IsolationForest(n_estimators=200, random_state=0).fit(
            base[FEATURES].to_numpy(float))
        s = -self.iso.score_samples(base[FEATURES].to_numpy(float))
        # percentiles, not min/max: one freak mail would squash every real score
        self.lo, self.hi = np.percentile(s, [1, 99])
        return self

    def score(self, feats):
        s = -self.iso.score_samples(feats[FEATURES].to_numpy(float))
        return np.clip((s - self.lo) / max(self.hi - self.lo, 1e-9), 0, 1)


# --- SMS (§7) ----------------------------------------------------------------

# A real Indian header has THREE parts: AD-AXISBK-S is
# [operator+circle]-[registered header]-[category]. The telco prepends the prefix
# (whose infrastructure carried the message), the DLT platform appends the
# category - P promotional / S service / T transactional / G government,
# mandatory since 6 May 2025. Only the middle part belongs to the sender; both
# outer parts are added in transit and neither is chosen by the entity.
# 6 characters is the norm for the middle, but the registry holds real headers
# from BOB to SP_INLAND_DOC, so the range is 3-13 with underscores.
# A person also reports the sender the way their phone showed it and drops the
# prefix - "Subway", "Zepto", "HDFCBK". So for the ANOMALY features every part
# outside the entity stays optional: a merchant name typed off a screen is a
# header, not evidence of anything.
# Group 1 is the entity - the only part worth matching against the registry.
HEADER = re.compile(r"^(?:[A-Z]{2}-)?([A-Z0-9_]{3,13})(?:-[PSTG])?$")

# ...but TRUST is a different question, and it gets the strict form. The operator
# prefix is the one part of a header NOBODY can choose: the access provider
# stamps it on in transit, along with the category suffix, during DLT scrubbing.
# Since 6 May 2025 a genuine carrier-delivered message therefore arrives as
# AX-ICICIT-S or DM-AXISBK-T, never as a bare ICICIT.
#
# So a bare entity is not a weaker form of a real header - it is a form a real
# header no longer takes, which makes trusting it a pure loss: no legitimate
# message needs it, and the messages that DO arrive that way skipped the
# pipeline the whole gate is built on. Requiring the prefix does not make the
# header unforgeable (nothing in a sender string is), it removes the cheapest
# shape.
CARRIER_HEADER = re.compile(r"^[A-Z]{2}-([A-Z0-9_]{3,13})(?:-[PSTG])?$")
TEN_DIGIT = re.compile(r"^\+?(?:91)?[6-9]\d{9}$")
INSTITUTION = re.compile(r"\b(bank|kyc|account|otp|upi|netbanking|card|loan|"
                         r"insurance|refund|customs|courier|electricity)\b", re.I)

# A DLT header's value is that an OPERATOR assigned it to a registered entity -
# not that it is two letters and a hyphen. `HEADER` matches the SHAPE, which is
# all the anomaly features need. Anything that grants TRUST must additionally
# check the entity is one we know: "VM-2204" and the look-alike "VM-HDFCBNK-S"
# both match the shape and neither belongs to anyone.
#
# Fails closed on purpose. In gateway mode (§10) the header arrives from the
# carrier and the operator supplies the real registry; a header typed into a web
# form attests nothing, so an unknown entity is never trusted.
#
# EVERY ENTRY COMES FROM A FILE. A hand-typed starter list of 47 lived here
# until 2026-09-22 and was deleted: 30 of them appeared in no registry anywhere
# ("SWIGGY", "ZOMATO", "UBERIN", "ZERODH", "SBIBNK"...), half naming
# organisations the dataset does not even carry, and the rest inventing a second
# spelling for a bank whose real headers the file already lists - SBI's are
# SBIINB, SBIPSG, SBICRD, ATMSBI, and SBIBNK is not one of them.
#
# They were inert for as long as `registered()` was only ever called with
# verified_sender=False. The Android app arms that gate (the carrier supplies
# the header there, which is the case §8's de-escalation was written for), and
# the moment it did, thirty unverifiable strings became load-bearing trust. A
# trust list that cannot be audited is not a trust list, and this module's own
# self-check already says so: "a real header we cannot verify fails closed".
#
# Populated below, once entity() exists to normalise the rows.
DLT_REGISTRY = set()


def sid(sender):
    """Normalise a sender the way a person actually types one.

    Only one rewrite, and it is the one that matters: a phone number typed with
    spaces, brackets or hyphens is still a phone number. Without this,
    "+91 98765 43210" missed TEN_DIGIT and silently lost
    personal_number_as_institution - §7's strongest signal, 0.60 of the SMS
    score - to a space bar. Header hyphens survive because the collapsed string
    is only kept when it turns out to be a number.
    """
    s = str(sender or "").strip().upper()
    bare = re.sub(r"[\s()-]", "", s)
    return bare if TEN_DIGIT.match(bare) else s


def entity(sender):
    """The registered-entity part: 'AD-AXISBK-S', 'AX-HDFCBK', 'HDFCBK' -> the middle.

    Strips the operator prefix and the category suffix. Splitting on the first
    hyphen was enough until TRAI made the suffix mandatory in May 2025; it then
    yielded 'AXISBK-S', which matches no registry entry, so every genuine bank
    SMS lost its trust and gained a non_dlt_header anomaly.
    """
    m = HEADER.match(sid(sender))
    return m.group(1) if m else sid(sender)


# 204 real TRAI-registered headers from the ScamShield registry, for 83
# organisations - several headers each, because a bank genuinely holds several
# (HDFC alone ships 8). This is the whole registry now, not an addition to one.
#
# A missing folder therefore leaves it EMPTY, and that is the right failure:
# every de-escalation stops, no message is wrongly cleared, and nothing is
# wrongly flagged either, because the gate only ever clears. The old fallback
# was a list of guesses, which fails open.
#
# Every row goes through entity(): the CSV is someone else's file and already
# ships one prefixed row ("BZ-CBSSBI"), which would never match a lookup.
try:
    DLT_REGISTRY |= {entity(h) for h in
                     pd.read_csv(data.KB / "sms_header_registry.csv").header}
except OSError:
    pass


def registered(sender):
    """-> True only for a CARRIER-STAMPED header whose entity is in DLT_REGISTRY.

    This is the trust signal, and it asks for two independent things: that an
    access provider put its prefix on the message in transit, and that the
    entity inside is one the registry names. A format match alone is NOT trust -
    treating format as trust let any typed string earn a de-escalation.

    The prefix requirement was added 2026-09-22 on Mohak's observation that his
    own bank SMS all arrive as AX-ICICIT-S / DM-AXISBK-T. `HEADER` stays lenient
    for the anomaly features, where a merchant name typed off a phone screen
    must not read as an anomaly; only this function got strict, because this is
    the only one that GRANTS anything. The bare form is refused here without
    becoming an anomaly there - which is the split the two regexes exist for.

    Still only gateway mode (§10) reaches this: a sender typed into a web form
    attests nothing however well shaped, so api.py gates the whole call behind
    verified_sender.
    """
    m = CARRIER_HEADER.match(sid(sender))
    return bool(m) and m.group(1) in DLT_REGISTRY


SMS_W = {"personal_number_as_institution": 0.60, "non_dlt_header": 0.15,
         "first_contact": 0.10, "url_first_contact": 0.25, "burst_1h": 0.10,
         "odd_hour": 0.10}

SMS_TAGS = [("personal_number_as_institution",
             "personal number claiming to be an institution"),
            ("url_first_contact", "link in a first-ever message from this sender"),
            ("non_dlt_header", "sender ID does not follow the DLT header format"),
            ("burst_1h", "burst of messages from this sender"),
            ("odd_hour", "delivered at an odd hour"),
            ("first_contact", "no prior history with this sender")]


def sms_features(sender_id, text, first_contact=None, has_url=False, hour=None,
                 burst=0):
    """§7's SMS list. No corpus backs these: every public SMS set is bare text
    with no sender field, so this is rules with a self-check, not a measured
    model. Wire it to real Android metadata and it becomes evidence.

    The known-contact skip rule (§7) is the caller's job - do not call this for a
    saved contact. That rule is what kills the false positives.
    """
    s = sid(sender_id)
    return {
        # §7 calls this the very strong signal: under Indian DLT rules a personal
        # number claiming to be an institution can never be legitimate
        "personal_number_as_institution":
            float(bool(TEN_DIGIT.match(s) and INSTITUTION.search(str(text)))),
        "non_dlt_header": float(not HEADER.match(s) and not TEN_DIGIT.match(s)),
        # None = we have no history to judge against, so BOTH of these abstain.
        # See known_sender(): absence from an empty store is not evidence of a
        # first contact, and url_first_contact carries 0.25 on that same claim.
        # A caller that really knows (gateway mode, the phone's message store)
        # passes True/False and gets the full rule back.
        "first_contact": 0.0 if first_contact is None else float(first_contact),
        "url_first_contact": 0.0 if first_contact is None
                             else float(first_contact and has_url),
        "burst_1h": min(burst / 5.0, 1.0),
        # ABSTAINS WITHOUT A REAL SEND TIME, and `hour=None` is the default
        # because the common case has none. The scan clock is not the message
        # clock: a message pasted into a form at 2am was not DELIVERED at 2am,
        # and scoring it that way accused the user of the hour they happened to
        # be awake. A real hour reaches here only from a screenshot timestamp
        # (OCR) or from an ingested message's own metadata (§10 gateway, the
        # Gmail push). Defaulting to noon instead would be the same lie with a
        # friendlier sign - it would assert "normal hour" about a time nobody
        # knows.
        "odd_hour": 0.0 if hour is None else float(hour < 7 or hour >= 23),
    }


def sms_score(f):
    """-> (behaviour_anomaly_score in [0,1], tags)."""
    return (float(np.clip(sum(f[k] * v for k, v in SMS_W.items()), 0, 1)),
            [t for k, t in SMS_TAGS if f[k] >= 0.5][:3])


# --- voice: the caller number (§7) -------------------------------------------
# TRAI direction of Nov 2025: banks, NBFCs, insurers and other regulated
# financial entities must place service and transaction calls from the 1600
# series (commercial banks by 2026-01-01, the rest by mid-March 2026); 140 is
# promotional telemarketing. So a call that says it is from a bank and comes
# from an ordinary number breaks a rule a real bank must follow - the voice twin
# of personal_number_as_institution. A 1600 number earns no trust of its own
# (caller ID can be spoofed): it only switches this one flag off, and then
# counts like any other clean caller check. Courier, police, electricity
# and the like are not on 1600, so their claims are deliberately not covered.
_FIN = (r"(?:bank|banking|insurance|credit card|debit card|card department|loans?|"
        r"finance|financial|nbfc|mutual funds?|demat|kyc|rbi|reserve bank)")
FIN_CLAIM = re.compile(r"\b(?:calling|call(?:ing)? you|speaking|this is|i am|i'm|we are|"
                       rf"we're|my name is)\b[^.?!]{{0,60}}?\b{_FIN}\b", re.I)
WITHHELD = {"withheld", "private", "unknown", "hidden", "no caller id", "anonymous"}

VOICE_W = {"fin_claim_not_1600": 0.60, "risky_sms": 0.60, "withheld": 0.30,
           "foreign_number": 0.25}
VOICE_TAGS = [("fin_claim_not_1600", "caller says they are from a bank or financial "
               "company, but those must call from 1600-series numbers (TRAI)"),
              ("risky_sms", "this number also sent you a message flagged as risky"),
              ("withheld", "caller ID was withheld"),
              ("foreign_number", "call came from an international number")]


def national(caller):
    """-> (digits without the India prefix, is_indian), or None if no number."""
    s = str(caller or "").strip()
    d = re.sub(r"\D", "", s)
    if not d:
        return None
    if s.startswith("+") or d.startswith("00"):
        d = d.lstrip("0")
        return (d[2:], True) if d.startswith("91") else (d, False)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    return d[1:] if len(d) == 11 and d.startswith("0") else d, True


def voice_features(caller, text, risky_sms=None):
    """§7's voice list, the part an uploaded call can answer. None = no caller
    number given, and the whole channel abstains (fusion renormalises it away)
    rather than reading "no number typed" as "nothing odd about the number".

    risky_sms: highest risk among this user's scanned SMS from the same number
    (store.risky_sms), None when it could not be looked up.
    ponytail: no first-call-from-this-number rule. A web upload never warms a
    per-user call history to MIN_PROFILE, so it would abstain forever.
    """
    c = str(caller or "").strip()
    held = c.lower() in WITHHELD
    num = None if held else national(c)
    if not held and num is None:
        return None
    d, india = num or ("", True)
    official = india and d.startswith(("1600", "140"))
    return {"fin_claim_not_1600": float(bool(FIN_CLAIM.search(str(text)))
                                        and not official),
            "risky_sms": float((risky_sms or 0) >= 0.35),
            "withheld": float(held),
            "foreign_number": float(not held and not india)}


def voice_score(f):
    """-> (behaviour_anomaly_score in [0,1], tags), or (None, []) to abstain."""
    if f is None:
        return None, []
    return (float(np.clip(sum(f[k] * v for k, v in VOICE_W.items()), 0, 1)),
            [t for k, t in VOICE_TAGS if f[k] >= 0.5])


# --- cross-channel correlation (§7) ------------------------------------------

def correlate(events, hours=24, boost=0.15):
    """A smishing SMS, then a call from a related identity within 24h, is one
    campaign - escalate BOTH events (§7, the part nobody publishes).

    events: [{"id", "channel", "time", "risk"}] -> same list with risks raised
    where a different channel from the same identity sits nearby in time.
    ponytail: exact identity match. Fuzzy linking (same number in another format,
    domain <-> brand) is the upgrade once a real event log exists to tune on.
    """
    out = [dict(e) for e in events]
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if (a["id"] != b["id"] or a["channel"] == b["channel"]
                    or abs((a["time"] - b["time"]).total_seconds()) > hours * 3600):
                continue
            if max(a["risk"], b["risk"]) < 0.35:   # only a real signal escalates
                continue
            for e, other in ((a, b), (b, a)):
                e["risk"] = min(1.0, e["risk"] + boost)
                e["linked"] = (f"{other['channel']} from the same identity "
                               f"({a['id']}) within {hours}h")
    return out


if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone

    T = datetime(2026, 8, 18, 10, tzinfo=timezone.utc)

    # 0. the address is the <...> one, never one typed into the display name,
    #    and a From line with a comma yields one address, not two
    assert addr_parts("bank@bank.com <atk@evil.tk>")[1] == "atk@evil.tk"
    assert addr_parts("x@evil.tk, ceo@bank.com")[1] == "x@evil.tk"

    # 1. streaming: nothing sees its own future
    df = pd.DataFrame({
        "sender": ["Alice <a@corp.com>"] * 3 + ["HDFC Bank <win@gmail.com>"],
        "receiver": ["me@x.com"] * 4,
        "date": [T, T + timedelta(hours=24), T + timedelta(hours=48),
                 T + timedelta(hours=49)],
        "text": ["Hi. See attached. Thanks."] * 3 + ["Your KYC expires. Click now."],
        "urls": [0, 0, 0, 1]})
    f = stream(df)
    assert f.first_contact.tolist() == [1, 0, 0, 1], "first_contact must use the past"
    assert f.display_mismatch.iloc[3] == 1 and f.freemail_brand.iloc[3] == 1
    assert f.display_mismatch.iloc[0] == 0, "a name with no brand is not a mismatch"
    assert f.url_first_contact.iloc[3] == 1
    assert rule_score(f)[3] > rule_score(f)[0], "impersonation must outscore a known sender"
    assert "free mail" in " ".join(explain(f.iloc[3]))
    assert explain(f.iloc[1])[0].startswith("nothing unusual")

    # 1b. a cold store cannot accuse anybody: every sender looks new in it
    cold = new_store()
    cf = features(cold, "Groww <no-reply@groww.in>", "me", T, "Your IPO is placed.", False)
    assert cf["first_contact"] == 1 and cf["new_domain"] == 1
    assert round(rule_score(cf), 3) == 0.65, rule_score(cf)   # ...above ESCALATE
    assert not profile_ready(cold), "an empty store must never be treated as a profile"
    for i in range(MIN_PROFILE):
        update(cold, f"s{i}@x.com", "me", T, style_of("hello"))
    assert profile_ready(cold), "a store with MIN_PROFILE senders is a profile"

    # 1c. auth: unknown abstains, a pass is not a vote, a fail reaches ESCALATE alone
    assert auth_fail() is None and auth_fail(None, "", None) is None
    assert auth_fail("pass", "none", "pass") == 0.0
    assert auth_fail("pass", "pass", "FAIL") == 1.0 and auth_fail("softfail") == 1.0
    assert "auth_fail" not in features(cold, "a@b.com", "me", T, "hi", False)
    af = features(cold, "a@b.com", "me", T, "hi", False, auth=1.0)
    assert rule_score(af) == 1.0, "stranger 0.65 + auth 0.60, clipped"
    assert rule_score({"auth_fail": 1.0}) == 0.60 and "SPF" in explain({"auth_fail": 1.0})[0]

    # 1d. trust: established + auth ok + every link on the sender's own domain
    assert base_domain("click.mailer.shopmail.co.in") == "shopmail.co.in"
    assert base_domain("a.b.example.com") == "example.com" and base_domain("x.co") == "x.co"
    ts = new_store()
    for _ in range(TRUST_N):
        update(ts, "news@shopmail.co.in", "me", T, None)       # seeded: no body
    assert ts["p"]["news@shopmail.co.in"]["style"] == [], "a header-only event has no style"
    z = "Shop <news@shopmail.co.in>"
    assert trusted(ts, z, ["www.shopmail.co.in", "click.shopmail.co.in"])
    assert trusted(ts, z, []), "no links at all is on-domain"
    assert not trusted(ts, z, ["shopmail.co.in", "evil.tk"]), "one off-domain link disarms it"
    assert not trusted(ts, z, ["shopmail.co.in.evil.tk"]), "suffix lookalike"
    assert not trusted(ts, z, None), "links unknown (no raw mail) never trusts"
    assert not trusted(ts, z, [], auth=1.0), "a failed auth check never trusts"
    assert not trusted(ts, "new@shopmail.co.in", []), "trust is per address, not per domain"
    add_contact(ts, "Friend <pal@corp.com>")
    assert trusted(ts, "pal@corp.com", []), "someone this mailbox wrote to"
    assert features(ts, "pal@corp.com", "me", T, "hi", False)["first_contact"] == 0
    assert features(ts, "x@corp.com", "me", T, "hi", False)["new_domain"] == 0

    # 1e. crowd: abstain without data, gate on distinct reporters AND rate
    assert crowd_reported(None) is None
    assert crowd_reported((2, 2)) == 0.0, "two reporters are not a crowd"
    assert crowd_reported((3, 10)) == 1.0
    assert crowd_reported((3, 10_000)) == 0.0, "3 of 10,000 receivers is under 0.3%"
    cf2 = features(cold, "a@b.com", "me", T, "hi", False, crowd=1.0)
    assert "other users" in explain({"crowd_reported": 1.0})[0] and cf2["crowd_reported"] == 1.0

    # 2. hours are circular
    assert circ_dev(2, [9, 10, 9, 10]) > circ_dev(10, [9, 10, 9, 10])
    assert circ_dev(23, [0, 1, 23, 0]) < circ_dev(12, [0, 1, 23, 0])

    # 3. SMS rules
    scam, _ = sms_score(sms_features("9876543210", "Your bank KYC is pending",
                                     has_url=True, first_contact=True))
    real, _ = sms_score(sms_features("AX-HDFCBK", "Your OTP is 4321",
                                     first_contact=False))
    assert scam > 0.7 > real, (scam, real)
    assert "personal number" in sms_score(sms_features("9876543210", "bank kyc"))[1][0]

    # a number typed the way people type numbers is still a number. Before sid(),
    # a single space dropped §7's strongest signal from 0.60 to nothing.
    for typed in ("+91 98765 43210", "98765-43210", "(+91) 9876543210"):
        assert sms_features(typed, "bank kyc")[
            "personal_number_as_institution"] == 1, typed

    # an unknown SENDER is not a first contact either - both halves of the claim
    # abstain, and a caller that actually knows still gets the full rule
    unknown = sms_features("9876543210", "Your bank KYC is pending", has_url=True)
    assert unknown["first_contact"] == unknown["url_first_contact"] == 0.0
    told = sms_features("9876543210", "Your bank KYC is pending", has_url=True,
                        first_contact=True)
    assert told["url_first_contact"] == 1.0
    assert sms_score(told)[0] > sms_score(unknown)[0] > 0.5   # still the strong signal
    assert known_sender(new_store(), "AX-HDFCBK", "sms") is None  # cold: no basis

    # SMS and email senders share one table and must not arm each other's gate
    mixed = new_store()
    for i in range(MIN_PROFILE):
        mixed["p"][sender_key(f"AX-BANK{i}", "sms")]["n"] = 1
    assert profile_ready(mixed, channel="sms") and not profile_ready(mixed, channel="email")
    assert known_sender(mixed, "AX-BANK0", "sms") is True
    assert known_sender(mixed, "AX-NEVER", "sms") is False
    assert known_sender(mixed, "a@b.com") is None            # email side still cold

    # an unknown send time is not an odd one - the scan clock is not the message
    # clock, and this fired on every message typed into the web form at night
    assert sms_features("AX-HDFCBK", "Your OTP is 4321")["odd_hour"] == 0.0
    assert sms_features("AX-HDFCBK", "Your OTP is 4321", hour=3)["odd_hour"] == 1.0

    # a merchant name typed off the phone screen is a header, not an anomaly...
    for m in ("Subway", "Zepto", "HDFCBK", "AD-ZEPTO"):
        assert sms_features(m, "offer inside")["non_dlt_header"] == 0, m
    # ...and still not a phone number pretending to be one
    assert sms_features("HDFC Bank", "kyc")["non_dlt_header"] == 1

    # ...but only the registry grants trust, in either transcription
    assert registered("ax-hdfcbk") and registered("AX-HDFCBK")
    assert not registered("SUBWAY") and not registered("ZEPTO"), \
        "an unregistered merchant name must never be trusted"
    assert not registered("VM-HDFCBNK") and not registered("9876543210")

    # The carrier's own stamp is required, because the carrier is the only party
    # that can apply it: since 6 May 2025 a delivered message reads AX-ICICIT-S,
    # never a bare ICICIT. A bare entity is refused for TRUST...
    for bare in ("HDFCBK", "HDFCBK-S", "SBIINB", "ATMSBI", "AXISBK-T"):
        assert not registered(bare), f"{bare} carries no access provider prefix"
    # ...and is still not an ANOMALY, which is the whole point of two regexes:
    # a person reporting a sender off their screen drops the prefix.
    assert sms_features("HDFCBK", "your otp")["non_dlt_header"] == 0

    # the three-part form TRAI made mandatory in May 2025. The category suffix
    # and the operator prefix are both added in transit; stripping both is what
    # keeps a real bank trusted instead of reading as an unknown entity.
    for real in ("AD-AXISBK-S", "VM-HDFCBK-T", "AX-ICICIB-P", "AX-HDFCBK"):
        assert registered(real), real
        assert sms_features(real, "Your OTP is 4321")["non_dlt_header"] == 0, real
    assert entity("AD-AXISBK-S") == "AXISBK" and entity("JD-ISECLT-S") == "ISECLT"
    assert entity("BZ-CBSSBI") == "CBSSBI" in DLT_REGISTRY, "registry rows normalise too"
    assert not registered("VM-HDFCBNK-S"), "a look-alike is one in any form"
    assert not registered("JD-ISECLT-S"), "a real header we cannot verify fails closed"

    # ...and the same rule applied to ourselves. These were trusted until
    # 2026-09-22 because they had been typed into this file by hand, which is
    # not a source. SBI's real headers are in the registry; SBIBNK never was.
    for guess in ("SWIGGY", "ZOMATO", "UBERIN", "OLACAB", "ZERODH", "SBIBNK",
                  "UIDAIG", "BESCOM", "AD-SWIGGY-S"):
        assert not registered(guess), f"{guess} is a guess, not a registration"
    assert registered("VM-SBIINB-S") and registered("JD-ATMSBI-T"), \
        "the real ones survive, in the form a carrier actually delivers them"
    assert DLT_REGISTRY <= {entity(h) for h in
                            pd.read_csv(data.KB / "sms_header_registry.csv").header}, \
        "every trusted entity must come from the registry file, not from here"

    # 4. cross-channel: an SMS and a call from one identity inside 24h
    ev = [{"id": "9876543210", "channel": "sms", "time": T, "risk": 0.55},
          {"id": "9876543210", "channel": "voice", "time": T + timedelta(hours=3),
           "risk": 0.30},
          {"id": "9999999999", "channel": "voice", "time": T, "risk": 0.30}]
    out = correlate(ev)
    assert round(out[0]["risk"], 2) == 0.70 and round(out[1]["risk"], 2) == 0.45, out
    assert out[2]["risk"] == 0.30, "unrelated identity must not be escalated"
    assert "linked" not in out[2]

    # 5. voice caller number
    BANK = "Hello sir, I am calling from the SBI credit card department."

    def vs(caller, text=BANK, risky=None):
        score, tags = voice_score(voice_features(caller, text, risky))
        return (None if score is None else round(score, 2)), tags

    assert vs("") == (None, []) and vs("call me") == (None, []), "no number abstains"
    for mobile in ("9876543210", "+91 98765 43210", "098765-43210", "0091 9876543210"):
        assert vs(mobile)[0] == 0.60, mobile
    assert national("+91 1600 123 456") == ("1600123456", True)
    for official in ("1600123456", "+91 1600123456", "1401234567"):
        assert vs(official) == (0.0, []), f"{official}: TRAI's own series"
    assert vs("9876543210", "hi it's me, are we still on for dinner") == (0.0, []), \
        "a mobile number alone is not an anomaly"
    assert vs("Withheld")[0] == 0.90 and vs("withheld", "hello")[0] == 0.30
    assert vs("+1 202 555 0143", "hello")[0] == 0.25
    assert vs("+1 202 555 0143")[0] == 0.85, "a foreign number is not 1600 either"
    assert vs("9876543210", "hello", 0.8)[0] == 0.60 and vs("9876543210", "hello", 0.2)[0] == 0
    assert "1600" in vs("9876543210")[1][0]
    print("ok")
