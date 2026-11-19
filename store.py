"""Sender profiles that outlive the process (spec §7, behaviour persistence).

behaviour.py builds the profile; this is only where it is KEPT. The dict
behaviour.new_store() returns stays the working copy - hydrated from Postgres on
first use, written back after each event - so features()/update() and every
number they produce are unchanged. This is the swap the behaviour.py header
promised ("swap new_store/update when it needs to outlive the process").

Why it matters: MIN_PROFILE is 20 senders, and an in-process dict reset on every
restart meant a real mailbox never reached it, so the email channel abstained
from behaviour scoring forever. Profiles that persist are what arm it.

One row per (user, sender). Identity and counts are real columns - that is the
sender information you can look at; `state` is the rolling window the scorer
reads and nothing else.
"""
import datetime as dt
import os
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

import behaviour

SCHEMA = """
CREATE TABLE IF NOT EXISTS sender_profiles (
  user_id     TEXT        NOT NULL,
  sender      TEXT        NOT NULL,
  domain      TEXT        NOT NULL DEFAULT '',
  display     TEXT        NOT NULL DEFAULT '',
  n           INT         NOT NULL DEFAULT 0,
  urls        INT         NOT NULL DEFAULT 0,
  first_seen  TIMESTAMPTZ,
  last_seen   TIMESTAMPTZ,
  state       JSONB       NOT NULL DEFAULT '{}',
  PRIMARY KEY (user_id, sender)
);
-- This mailbox SENT mail to them (Gmail's SENT label). A contact who never
-- wrote back is a row with n = 0: it feeds behaviour's contacts set, not the
-- MIN_PROFILE count, which is about senders.
ALTER TABLE sender_profiles ADD COLUMN IF NOT EXISTS contacted BOOLEAN NOT NULL DEFAULT false;
"""
# This table is owned here, not by web/lib/db.js: Python is the only writer, and
# api.py has to work with the Next app never having been started.


def env(name):
    """A setting from the environment, falling back to web/.env.local, so every
    credential stays in the one file the web app already has. ponytail: three
    lines instead of python-dotenv."""
    if os.environ.get(name):
        return os.environ[name]
    path = Path(__file__).parent / "web" / ".env.local"
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def url():
    return env("DATABASE_URL")


def connect():
    # ponytail: a connection per call. FastAPI runs these sync handlers in a
    # threadpool, so a shared one would need its own locking, and localhost
    # connect is ~2 ms against two queries per scan. psycopg_pool the day that
    # shows up in scan latency.
    c = psycopg.connect(url(), autocommit=True)
    c.execute(SCHEMA)
    return c


def load(user):
    """-> a behaviour store holding every sender profile this user has."""
    store = behaviour.new_store()
    with connect() as c:
        rows = c.execute(
            "SELECT sender, domain, n, state, contacted FROM sender_profiles"
            " WHERE user_id = %s", [user]).fetchall()
    for sender, domain, n, state, contacted in rows:
        if contacted:
            behaviour.add_contact(store, sender)
        if n == 0:
            continue
        p = store["p"][sender]
        p["n"] = n
        p["hours"] = state.get("hours", [])
        p["times"] = [dt.datetime.fromisoformat(t) for t in state.get("times", [])]
        p["rcpts"] = set(state.get("rcpts", []))
        p["style"] = [tuple(s) for s in state.get("style", [])]
        # `seen` is (receiver, sender) pairs and rcpts holds exactly those
        # receivers, so first_contact rebuilds from the profile - no second column.
        store["seen"] |= {(r, sender) for r in p["rcpts"]}
        store["domains"].add(domain)
    return store


def save(user, sender, store, when=None, has_url=False, display="", domain=""):
    """Write back the one sender the current event touched. Call after
    behaviour.update(), which is what put the new state in the store."""
    p = store["p"][sender]
    state = {"hours": p["hours"],
             "times": [t.isoformat() for t in p["times"]],
             "rcpts": sorted(p["rcpts"]),
             "style": [list(s) for s in p["style"]]}
    with connect() as c:
        c.execute("""
            INSERT INTO sender_profiles
                   (user_id, sender, domain, display, n, urls, first_seen, last_seen, state)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (user_id, sender) DO UPDATE SET
                domain     = EXCLUDED.domain,
                display    = EXCLUDED.display,
                n          = EXCLUDED.n,
                -- n is absolute (the dict was hydrated from this row); urls is
                -- not kept in the dict, so it accumulates in SQL instead.
                urls       = sender_profiles.urls + EXCLUDED.urls,
                first_seen = LEAST(sender_profiles.first_seen, EXCLUDED.first_seen),
                last_seen  = GREATEST(sender_profiles.last_seen, EXCLUDED.last_seen),
                state      = EXCLUDED.state""",
            [user, sender, domain, display, p["n"], int(bool(has_url)),
             when, when, Jsonb(state)])


def save_contacts(user, addresses):
    """Mark addresses this mailbox wrote to. Idempotent: a re-seed or a second
    Sent mail to the same person changes nothing."""
    with connect() as c, c.cursor() as cur:
        cur.executemany("""
            INSERT INTO sender_profiles (user_id, sender, domain, contacted)
            VALUES (%s, %s, %s, true)
            ON CONFLICT (user_id, sender) DO UPDATE SET contacted = true""",
            [(user, a, a.rsplit("@", 1)[-1]) for a in addresses])


CROWD_DAYS = 90
# A report counts only if the reporter's account was this old WHEN they made it.
# Sign-up is open, so three fresh accounts subscribed to a real newsletter could
# otherwise condemn it the same afternoon. ponytail: it delays a sock-puppet
# campaign by two weeks per batch, it does not stop one; reporter reputation
# (agreement with later reports) is the upgrade if that ever matters.
CROWD_ACCOUNT_DAYS = 14


def crowd(sender, caller=""):
    """-> (reporters, receivers) for one address across OTHER users, or None.

    Reads the web app's `scans` table: a report is a scan whose feedback is
    'is_a_threat'. Only Gmail scans count - their sender is the From header we
    actually received, where the web form's sender is whatever was typed, and a
    typed sender would let one account aim reports at anybody. Model verdicts
    are never read here: only humans move the crowd score, or the model would
    teach itself. Both counts are DISTINCT users within CROWD_DAYS.

    A REPORT counts only if that mail passed DMARC (scans.detail.dmarc, written
    by api.email): a From header is otherwise forgeable, and three users
    honestly reporting a spoof of alerts@bank.in would condemn the real bank.
    The caller is left out, so the reason "reported by several OTHER users" is
    true and a user cannot count their own report. Ceiling, not fixed: a few
    sign-ups subscribed to a real newsletter can still report it, once each
    account is CROWD_ACCOUNT_DAYS old (an unknown account never counts).
    None = scans is absent or unmigrated - the feature abstains.
    ponytail: a substring scan on scans.sender, then an exact match in Python;
    a sender_address column + index when scans gets big.
    """
    addr = behaviour.addr_parts(sender)[1]
    if not addr:
        return None
    try:
        with connect() as c:
            rows = c.execute("""
                SELECT s.user_id, s.sender, s.feedback = 'is_a_threat'
                         AND s.feedback_at > now() - make_interval(days => %s)
                         AND s.detail->>'dmarc' = 'pass'
                         AND s.feedback_at - u.created_at >= make_interval(days => %s),
                       s.created_at > now() - make_interval(days => %s)
                  FROM scans s LEFT JOIN users u ON u.id::text = s.user_id
                 WHERE s.source = 'gmail' AND s.channel = 'email' AND s.user_id <> %s
                   AND strpos(lower(s.sender), %s) > 0""",
                [CROWD_DAYS, CROWD_ACCOUNT_DAYS, CROWD_DAYS, str(caller), addr]).fetchall()
    except psycopg.Error:
        return None
    rows = [r for r in rows if behaviour.addr_parts(r[1])[1] == addr]
    return (len({u for u, _, rep, _ in rows if rep}),   # NULL (no account) = no
            len({u for u, _, _, recent in rows if recent}))


RISKY_SMS_DAYS = 30


def risky_sms(user, caller):
    """-> highest risk among this user's SMS scans from the caller's number in
    the last RISKY_SMS_DAYS (0.0 if none), or None when it cannot be looked up.

    §7's cross-channel link, call side only. §7 says "within 24h", but an
    uploaded recording carries no call time and the scan clock is not the call
    clock, so the window is the user's recent scans instead. Numbers match on
    their last 10 digits, so "+91 98765 43210" and "9876543210" are one sender.
    ponytail: the SMS's own stored verdict is not raised in turn; regex over
    scans.sender per call is fine at one user's volume, an indexed number
    column if scans gets big.
    """
    num = behaviour.national(caller)
    if not num or len(num[0]) < 10:
        return None
    try:
        with connect() as c:
            (risk,), = c.execute("""
                SELECT max(risk) FROM scans
                 WHERE user_id = %s AND channel = 'sms'
                   AND created_at > now() - make_interval(days => %s)
                   AND right(regexp_replace(coalesce(sender, ''), '\\D', '', 'g'), 10) = %s""",
                [str(user), RISKY_SMS_DAYS, num[0][-10:]]).fetchall()
    except psycopg.Error:
        return None
    return float(risk or 0.0)


def profiles(user, limit=100):
    """The sender information itself, newest contact first - for a report, an
    admin page, or checking what the mailbox actually knows."""
    with connect() as c:
        return c.execute("""
            SELECT sender, domain, display, n, urls, first_seen, last_seen
              FROM sender_profiles WHERE user_id = %s
             ORDER BY last_seen DESC NULLS LAST LIMIT %s""",
            [user, limit]).fetchall()


if __name__ == "__main__":
    U = "_selftest"
    T = dt.datetime(2026, 8, 27, 9, tzinfo=dt.timezone.utc)
    with connect() as c:
        c.execute("DELETE FROM sender_profiles WHERE user_id = %s", [U])

    # two mails from one sender, saved the way api.py saves them
    s = load(U)
    assert not s["p"], "a fresh user must start with no profiles"
    for i, txt in enumerate(["Hi. See attached. Thanks.", "Minutes are attached."]):
        f = behaviour.features(s, "Alice <a@corp.com>", "me", T, txt, i == 1)
        assert f["first_contact"] == float(i == 0), "first_contact must use the past"
        behaviour.update(s, "a@corp.com", "me", T + dt.timedelta(hours=i),
                         behaviour.style_of(txt))
        save(U, "a@corp.com", s, when=T + dt.timedelta(hours=i), has_url=(i == 1),
             display="alice", domain="corp.com")

    # ...and the store rebuilt from Postgres scores the next mail identically
    back = load(U)
    assert set(back["p"]) == {"a@corp.com"} and back["p"]["a@corp.com"]["n"] == 2
    fresh = behaviour.features(back, "Alice <a@corp.com>", "me", T, "hello", False)
    assert fresh["first_contact"] == 0 and fresh["new_domain"] == 0, \
        "a persisted sender is not a first contact"
    assert behaviour.features(back, "Bob <b@other.com>", "me", T, "hi", False)[
        "first_contact"] == 1, "an unknown sender still is"
    assert back["p"]["a@corp.com"]["times"] == [T, T + dt.timedelta(hours=1)], \
        "burst needs real datetimes back, not strings"
    assert len(back["p"]["a@corp.com"]["style"]) == 2

    row = profiles(U)[0]
    assert row[:5] == ("a@corp.com", "corp.com", "alice", 2, 1), row  # urls summed
    assert row[5] == T and row[6] == T + dt.timedelta(hours=1), row   # first/last seen

    # MIN_PROFILE is reachable across restarts, which is the whole point
    for i in range(behaviour.MIN_PROFILE):
        behaviour.update(s, f"s{i}@x.com", "me", T, behaviour.style_of("hello"))
        save(U, f"s{i}@x.com", s, when=T, domain="x.com")
    assert behaviour.profile_ready(load(U)), "profiles must survive a reload"

    # contacts persist, never count as senders, and survive the sender writing back
    n_before = behaviour.profile_count(load(U))
    save_contacts(U, ["pal@corp.com", "a@corp.com"])
    save_contacts(U, ["pal@corp.com"])                          # idempotent
    back = load(U)
    assert back["contacts"] == {"pal@corp.com", "a@corp.com"}
    assert behaviour.profile_count(back) == n_before, "a contact is not a sender"
    assert back["p"]["a@corp.com"]["n"] == 2, "marking a contact keeps its history"
    save(U, "a@corp.com", back, when=T, domain="corp.com")
    assert "a@corp.com" in load(U)["contacts"], "a later save keeps contacted"

    # crowd: distinct Gmail reporters within CROWD_DAYS, typed senders ignored
    S = "Spam King <king@crowd-selftest.tk>"
    wipe = """DELETE FROM scans WHERE user_id IN
                (SELECT id::text FROM users WHERE username LIKE '_crowdtest%%');
              DELETE FROM scans WHERE user_id = 'ghost-crowdtest';
              DELETE FROM users WHERE username LIKE '_crowdtest%%'"""
    try:
        with connect() as c:
            c.execute(wipe)
            ids = []
            #  source   feedback      days ago  dmarc   account age (days) at report
            for i, (src, fb, age, dmarc, acct) in enumerate([
                    ("gmail", "is_a_threat", 1, "pass", 30),     # 0 counts
                    ("gmail", "is_a_threat", 2, "pass", 30),     # 1 counts
                    ("gmail", None, 1, "pass", 30),              # 2 receiver only
                    ("web", "is_a_threat", 1, "pass", 30),       # 3 typed sender
                    ("gmail", "is_a_threat", 200, "pass", 30),   # 4 expired
                    ("gmail", "is_a_threat", 1, "none", 30),     # 5 possible spoof
                    ("gmail", "is_a_threat", 1, "pass", 3)]):    # 6 fresh account
                uid = str(c.execute("""INSERT INTO users (username, password_hash, created_at)
                                       VALUES (%s, 'x', now() - make_interval(days => %s))
                                       RETURNING id""",
                                    [f"_crowdtest{i}", age + acct]).fetchone()[0])
                ids.append(uid)
                c.execute("""INSERT INTO scans (user_id, channel, preview, verdict, risk,
                                 detail, source, sender, feedback, feedback_at, created_at)
                             VALUES (%s,'email','','SAFE',0,%s,%s,%s,%s,
                                     now() - make_interval(days => %s),
                                     now() - make_interval(days => %s))""",
                          [uid, Jsonb({"dmarc": dmarc}), src, S, fb, age, age])
            # a report from a user_id with no account row cannot be aged: never counts
            c.execute("""INSERT INTO scans (user_id, channel, preview, verdict, risk, detail,
                             source, sender, feedback, feedback_at)
                         VALUES ('ghost-crowdtest','email','','SAFE',0,%s,'gmail',%s,
                                 'is_a_threat',now())""", [Jsonb({"dmarc": "pass"}), S])
        # Reporters: rows 0 and 1 only. Receivers: recent Gmail rows 0,1,2,5,6,ghost.
        assert crowd(S) == (2, 6), crowd(S)
        assert crowd(S, caller=ids[0]) == (1, 5), "the caller never counts"
        assert crowd("nobody@crowd-selftest.tk") == (0, 0)
        with connect() as c:
            c.execute(wipe)
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        print("scans not migrated yet (web app not run since) - crowd check skipped")

    # risky_sms: the same number however it was typed, this user's SMS only
    with connect() as c:
        c.execute("DELETE FROM scans WHERE user_id IN (%s, '_selftest2')", [U])
        for user, ch, snd, risk, age in [(U, "sms", "+91 98765-43210", 0.8, 1),
                                         (U, "sms", "9876543210", 0.2, 2),
                                         (U, "sms", "9876543210", 0.99, 40),   # too old
                                         (U, "voice", "9876543210", 0.99, 1),  # not SMS
                                         ("_selftest2", "sms", "9876543210", 0.99, 1)]:
            c.execute("""INSERT INTO scans (user_id, channel, preview, verdict, risk,
                             detail, sender, created_at)
                         VALUES (%s,%s,'','SAFE',%s,'{}',%s,
                                 now() - make_interval(days => %s))""",
                      [user, ch, risk, snd, age])
    assert risky_sms(U, "098765 43210") == 0.8, risky_sms(U, "098765 43210")
    assert risky_sms(U, "9123456789") == 0.0 and risky_sms(U, "withheld") is None
    with connect() as c:
        c.execute("DELETE FROM scans WHERE user_id IN (%s, '_selftest2')", [U])

    with connect() as c:
        c.execute("DELETE FROM sender_profiles WHERE user_id = %s", [U])
    print("ok")
