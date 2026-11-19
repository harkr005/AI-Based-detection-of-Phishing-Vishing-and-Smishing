"""python scratch/_url_mail_check.py - url_model's MAIL link path (url_xgb_v4).

The PSL split is what the whole mail model rests on: get it wrong and a
tracking subdomain is scored as a domain again, or an attacker's label on
shared hosting is thrown away. Then one sanity row per model, so a missing
file or a feature-order slip shows up as a wrong number, not a silent fallback.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402,F401  (before xgboost, see CLAUDE.md)
import url_model as um  # noqa: E402

split = um.split
assert split("t2.service.icicisecurities.com") == ("t2.service", "icicisecurities.com")
assert split("www.amazon.in") == ("", "amazon.in")
assert split("lnk.custcomm.icici.bank.in") == ("lnk.custcomm", "icici.bank.in"), "bank.in is a suffix"
assert split("greekturtle-prod.s3.amazonaws.com")[0] == "", "s3 is a PRIVATE suffix"
assert split("paypal-login.github.io") == ("", "paypal-login.github.io")
assert split("hdfcbank.com.evil.tk") == ("hdfcbank.com", "evil.tk"), "brand in front, attacker's domain"
assert split("github.io") == ("", "github.io") and split("192.168.1.4") == ("", "192.168.1.4")

f = um.mail_features
assert f("https://vjockfvt.20-218-142-124.cloud-xip.com")["sub_ip"] == 1
assert f("https://t2.service.icicisecurities.com/x")["sub_ip"] == 0
assert f("http://kyc-verify.paytm.evil.tk")["sub_lure"] >= 2
# the registrable domain is scored with url_model's own 15 features
assert {k: v for k, v in f("https://t2.service.icicisecurities.com").items()
        if k in um.FEATURES} == um.features("https://icicisecurities.com")

assert um.load_mail() is not None, "models/url_xgb_v4.json or the PSL file is missing"
real = ["https://myaccount.google.com/x", "https://links.mail.crunchyroll.com/y",
        "https://t2.service.icicisecurities.com/z"]
fake = ["http://paytm-kyc-update.verify-now.tk/login", "http://192.168.1.4:8080/secure-login"]
old, new = um.score(real), um.score(real, mail=True)
print("genuine tracking links  v1", old.round(2), "  mail", new.round(2))
print("phishing                v1", um.score(fake).round(2), "  mail", um.score(fake, mail=True).round(2))
assert (new < .5).all(), "the mail model must clear real tracking links"
assert (um.score(fake, mail=True) > .5).all(), "...and still flag obvious phishing"
assert um.score(["https://hdfcbank.com/x"], mail=True)[0] == 0.0, "registry pin applies to mail too"

# Layer A, brand mismatch: the brand must be a whole token, and the brand's
# own domains (any of them) never match.
bm = um.brand_mismatch
assert bm("http://paypal-login.evil.tk/x") == "paypal"
assert bm("https://hdfcbank.com.evil.tk/login") == "hdfcbank"
assert bm("https://myaccount.google.com") is None and bm("https://www.paypal.com") is None
assert bm("https://www.facebookmail.com") is None, "brand inside a longer token is not a claim"
assert bm("https://lnk.custcomm.icici.bank.in") is None, "registry domains belong to their brand"
assert bm("https://login.secure-update.xyz") is None, "generic words are not brands"
assert um.score(["http://paypal-login.evil.tk/x"], mail=True)[0] >= um.BRAND_FLOOR
assert "paypal" in um.explain("http://paypal-login.evil.tk/x", mail=True)[1]
u = "http://paypal-login.evil.tk/x"
assert um.score([u])[0] == float(um.load().predict_proba(um.featurize([u]))[0, 1]), "SMS path: v1, no layers"

# Layer B, the authenticated sender's own link: DMARC pass AND top-1M domain.
so = um.sender_owner
assert so('"Google" <no-reply@accounts.google.com>', "pass") == "google.com"
assert so('"Google" <no-reply@accounts.google.com>', "fail") is None
assert so("alerts@hdfc-secure-x93k.tk", "pass") is None, "a domain the attacker owns is not established"
assert so("", "pass") is None and so("Google <no-reply@accounts.google.com>", None) is None
deep = "https://ab12-cd34.track.mail.google.com/x"
assert um._own(deep, "google.com") == "https://google.com" and um._own(deep, None) == deep
assert um._own("https://evil.tk/x", "google.com") == "https://evil.tk/x", "only the sender's own domain"
print("url mail check passed")
