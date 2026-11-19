"""Google Safe Browsing v5 (hashes.search) - a WEBSITE-ONLY link layer.

Privacy: no URL leaves this machine. Each link is canonicalised into Google's
host-suffix / path-prefix expressions, SHA-256'd, and only the 4-byte prefixes
are sent; Google returns every full hash sharing a prefix and the match is done
here. (developers.google.com/safe-browsing/reference/URLs.and.Hashing)
What Google CAN learn: a host-only expression (bank.com/) has a guessable
prefix, so it can tell which known domains appear in scanned messages - not
paths, query strings or tokens. Privacy review 2026-10-11.

How it is used (api._links): a listed link's risk is raised to LISTED_RISK
with Google's reason; an unlisted link is left alone, because "not on Google's
list" is not evidence of safety - fresh phishing is not listed yet. No key, no
network, an error: everything abstains and scanning is unchanged.

Terms: free for non-commercial use; results must carry "Advisory provided by
Google"; a warning may only rest on data fetched in the last 30 minutes - a
stored scan therefore says the link was listed WHEN SCANNED.

Key: GOOGLE_SAFE_BROWSING in web/.env.local (store.env). The phone never calls
this: it scores offline.
"""
import base64
import hashlib
import posixpath
import re
import time
import urllib.parse
import urllib.request

import store
import url_model

ENDPOINT = "https://safebrowsing.googleapis.com/v5/hashes:search"
LISTED_RISK = 0.95
TIMEOUT = 3          # seconds: a slow Google must not hold a scan up
SAY = {"SOCIAL_ENGINEERING": "phishing", "MALWARE": "malware",
       "UNWANTED_SOFTWARE": "unwanted software", "POTENTIALLY_HARMFUL_APPLICATION": "harmful app"}
_cache = {}          # 4-byte prefix -> (expires_at, {full_hash: threat_type})


def expressions(url):
    """-> Google's host-suffix / path-prefix expressions for one URL.

    ponytail: the spec's canonicalisation minus the rare cases - octal/hex IPv4
    and IPv6 normalisation, and IDN beyond Python's idna codec. A link using
    those is simply checked in the form it was written.
    """
    url = re.sub(r"[\t\r\n]", "", url.strip())
    if "://" not in url:
        url = "http://" + url
    url = url.split("#", 1)[0]
    for _ in range(10):                                   # repeated unescape
        un = urllib.parse.unquote(url)
        if un == url:
            break
        url = un
    try:
        p = urllib.parse.urlsplit(url)
        host = (p.hostname or "")
    except ValueError:
        return []
    host = re.sub(r"\.+", ".", host.strip(".")).lower()
    try:
        host = host.encode("idna").decode()
    except UnicodeError:
        pass
    if not host:
        return []
    path = re.sub(r"/+", "/", p.path or "/")
    path = posixpath.normpath(path) + ("/" if path.endswith("/") and path != "/" else "")
    path = "/" if path in (".", "//") else path
    esc = lambda s: "".join(c if 32 < ord(c) < 127 and c not in "#%" else
                            "".join(f"%{b:02X}" for b in c.encode()) for c in s)
    host, path, query = esc(host), esc(path), esc(p.query)

    if url_model.IP_RE.match(host):
        hosts = [host]
    else:
        reg = url_model.split(host)[1]
        labels = host.split(".")
        n = len(reg.split("."))
        hosts = [".".join(labels[-k:]) for k in range(n, min(len(labels), n + 4) + 1)]
        hosts = list(dict.fromkeys([host] + hosts[:4] + [reg]))[:5]
    parts = [s for s in path.split("/")[1:-1] if s]
    paths = ([path + "?" + query] if query else []) + [path] + \
            ["/" + "".join(s + "/" for s in parts[:i]) for i in range(min(len(parts), 3) + 1)]
    paths = list(dict.fromkeys(paths))[:6]
    return [h + q for h in hosts for q in paths]


# v5 hashes.search answers in protobuf only (alt=json -> 400 "Unsupported
# Output Format"). The message is three levels deep, so a 20-line decoder
# beats a protobuf dependency plus vendored .proto files:
#   SearchHashesResponse { repeated FullHash full_hashes = 1; Duration cache_duration = 2; }
#   FullHash { bytes full_hash = 1; repeated FullHashDetail full_hash_details = 2; }
#   FullHashDetail { ThreatType threat_type = 1; repeated ThreatAttribute attributes = 2; }
THREAT = {1: "MALWARE", 2: "SOCIAL_ENGINEERING", 3: "UNWANTED_SOFTWARE",
          4: "POTENTIALLY_HARMFUL_APPLICATION"}
SKIP_ATTR = {1, 2}       # CANARY (do not enforce), FRAME_ONLY (not a top-level threat)
# One URL can be listed under several types, and Google's order varies (the
# malware test page came back as malware, then as harmful app): always report
# the most serious one.
SEVERITY = ["SOCIAL_ENGINEERING", "MALWARE", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"]


def worst(kinds):
    return min(kinds, key=SEVERITY.index)


def _fields(buf):
    """-> [(field number, value)] of one protobuf message: ints for varints,
    bytes for length-delimited fields (the only two wire types used here)."""
    out, i = [], 0

    def varint():
        nonlocal i
        v = shift = 0
        while True:
            b = buf[i]
            i += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                return v
    while i < len(buf):
        key = varint()
        if key & 7 == 0:
            out.append((key >> 3, varint()))
        elif key & 7 == 2:
            n = varint()
            out.append((key >> 3, buf[i:i + n]))
            i += n
        else:
            raise ValueError(f"unexpected protobuf wire type {key & 7}")
    return out


def _decode(buf):
    """-> ({full_hash: threat_type}, cache seconds) from a SearchHashesResponse."""
    got, ttl = {}, 300.0
    for f, v in _fields(buf):
        if f == 2:
            ttl = float(dict(_fields(v)).get(1, 300))
        elif f == 1:
            fh = _fields(v)
            full = next((x for k, x in fh if k == 1), b"")
            kinds = []
            for k, d in fh:
                if k != 2:
                    continue
                det = _fields(d)
                attrs = {x for a, x in det if a == 2 and isinstance(x, int)} | \
                        {b for a, x in det if a == 2 and isinstance(x, bytes) for b in x}  # packed
                kind = THREAT.get(dict(det).get(1))
                if kind and not attrs & SKIP_ATTR:
                    kinds.append(kind)
            if kinds:
                got[full] = worst(kinds)
    return got, ttl


def _search(prefixes):
    """-> {full_hash: threat_type} for these prefixes, cached per Google's
    cacheDuration. Raises on network/HTTP errors (check() abstains)."""
    now, found, ask = time.time(), {}, []
    for pre in prefixes:
        hit = _cache.get(pre)
        if hit and hit[0] > now:
            found.update(hit[1])
        else:
            ask.append(pre)
    if ask:
        q = urllib.parse.urlencode([("key", store.env("GOOGLE_SAFE_BROWSING"))] +
                                   [("hashPrefixes", base64.b64encode(p).decode()) for p in ask])
        with urllib.request.urlopen(f"{ENDPOINT}?{q}", timeout=TIMEOUT) as r:
            got, ttl = _decode(r.read())
        for pre in ask:
            _cache[pre] = (now + ttl, {h: t for h, t in got.items() if h[:4] == pre})
        found.update(got)
    return found


def check(urls):
    """-> {url: threat in plain words} for every listed URL. {} when there is
    no key, Google is unreachable, or nothing is listed - never an exception."""
    if not urls or not store.env("GOOGLE_SAFE_BROWSING"):
        return {}
    try:      # expressions() too: a lone surrogate in a pasted URL must not 500 the scan
        full = {u: [hashlib.sha256(e.encode()).digest() for e in expressions(u)] for u in urls}
        prefixes = sorted({h[:4] for hs in full.values() for h in hs})[:1000]
        listed = _search(prefixes)
    except Exception as e:                  # network, quota, bad key: abstain
        print(f"safe browsing skipped ({type(e).__name__}: {e})")
        return {}
    out = {}
    for u, hs in full.items():
        kinds = [listed[h] for h in hs if h in listed]
        if kinds:
            out[u] = SAY[worst(kinds)]
    return out


if __name__ == "__main__":
    # The spec's own worked example.
    e = expressions("http://a.b.com/1/2.html?param=1")
    assert set(e) == {"a.b.com/1/2.html?param=1", "a.b.com/1/2.html", "a.b.com/", "a.b.com/1/",
                      "b.com/1/2.html?param=1", "b.com/1/2.html", "b.com/", "b.com/1/"}, e
    assert set(expressions("http://1.2.3.4/1/")) == {"1.2.3.4/1/", "1.2.3.4/"}
    assert len(expressions("http://a.b.c.d.e.f.g/1.html")) <= 30
    assert expressions("HTTP://WWW.Google.COM./#frag") == ["www.google.com/", "google.com/"]
    # The decoder, on the exact shape Google returned 2026-10-11: one full hash,
    # threat_type 2, cacheDuration 300 s.
    h = bytes(range(32))
    assert _decode(b"\n&\n " + h + b"\x12\x02\x08\x02" + b"\x12\x03\x08\xac\x02") == \
        ({h: "SOCIAL_ENGINEERING"}, 300.0)
    assert _decode(b"\n(\n " + h + b"\x12\x04\x08\x02\x10\x01") == ({}, 300.0), "CANARY ignored"
    assert _decode(b"") == ({}, 300.0)
    two = b"\n*\n " + h + b"\x12\x02\x08\x04" + b"\x12\x02\x08\x01"   # harmful app, then malware
    assert _decode(two) == ({h: "MALWARE"}, 300.0), "most serious type wins"
    print("expressions + decoder ok")
    # Live: Google's own always-listed test URLs, and one that never is.
    if store.env("GOOGLE_SAFE_BROWSING"):
        test = "http://testsafebrowsing.appspot.com/s/phishing.html"
        mal = "http://testsafebrowsing.appspot.com/s/malware.html"
        r = check([test, mal, "https://www.google.com/"])
        print("live check:", r or "(nothing listed - see message above if it was skipped)")
        assert r == {test: "phishing", mal: "malware"}, r
        # privacy review 2026-10-11: a lone surrogate used to raise out of check()
        assert check(["http://a" + chr(0xD800) + "b.com/x"]) == {}
        print("safe browsing check passed")
