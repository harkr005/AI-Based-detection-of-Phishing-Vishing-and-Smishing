"""python scratch/_mail_parser_check.py - data._pot_body on the four mail shapes
real Gmail sent that the old parser got wrong (2026-09-23):

  - HTML-only mail: <style>/<script>/<head> contents reached the model as text
    (CSS rules, `document.title = ...`), because tag stripping keeps what is
    between the tags;
  - empty text/plain beside a full text/html: the body came back "";
  - HTML inside a part labelled text/plain (a job-portal mail);
  - &zwnj;/&shy; preheader padding, hundreds of invisible characters.
"""
import email
import sys
from email import policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402


def body(*parts):
    m = MIMEMultipart("alternative")
    for text, sub in parts:
        m.attach(MIMEText(text, sub))
    return data._pot_body(email.message_from_bytes(m.as_bytes(), policy=policy.compat32))


PAGE = ('<!DOCTYPE html><html><head><title>T</title><style>#outlook a{padding:0}'
        '</style></head><body><!-- c --><script>document.title="x"</script>'
        '<p>Your order &amp; receipt</p>&zwnj;&zwnj;&shy;'
        '<a href="http://x.com">track</a></body></html>')
WANT = "Your order & receipt track"

assert body((PAGE, "html")) == (WANT, True), body((PAGE, "html"))
assert body(("", "plain"), (PAGE, "html"))[0] == WANT, "empty plain must fall back to HTML"
assert body((PAGE, "plain"))[0] == WANT, "HTML inside text/plain"
assert body(("Hi Bob,\n see <you> at 5", "plain"), (PAGE, "html"))[0] == "Hi Bob, see <you> at 5", \
    "a real plain part is used as-is"
assert body(("", "plain")) == ("", False)


# readable_body - what View shows a person: lines kept, links show their host.
def readable(*parts):
    m = MIMEMultipart("alternative")
    for text, sub in parts:
        m.attach(MIMEText(text, sub))
    return data.readable_body(email.message_from_bytes(m.as_bytes(), policy=policy.compat32))


MAIL = ('<html><head><style>p{color:red}</style></head><body>'
        '<p>Dear customer,</p><div><br></div><div><br></div>'
        '<p>Your account is on hold.<br><a href="https://hdfc-secure-in.tk/v?id=1">Verify now</a></p>'
        '<p>Visit <a href="https://www.hdfcbank.com">www.hdfcbank.com</a></p>'
        '<a href="https://t.example.com/px"><img src="x.png"></a></body></html>')
assert readable((MAIL, "html")) == (
    "Dear customer,\n\n"
    "Your account is on hold.\nVerify now (hdfc-secure-in.tk)\n"
    "Visit www.hdfcbank.com"), repr(readable((MAIL, "html")))
SPOOF = '<p><a href="https://evil.tk/x">https://www.hdfcbank.com/?r=evil.tk</a></p>'
assert readable((SPOOF, "html")) == "https://www.hdfcbank.com/?r=evil.tk (evil.tk)",     "a host merely MENTIONED in the link text must still be shown"
assert readable(("Hi Bob,\n\n\n\nSee you   at 5.\nMohak", "plain")) == "Hi Bob,\n\nSee you at 5.\nMohak"

# data.safe_html: the View's "as sent" frame. Every hole the 2026-10-11 privacy
# review found must stay shut; styling must survive.
EVIL = ('<html><head><meta http-equiv="refresh" content="0;url=https://evil.tk/r">'
        '<meta name="referrer" content="unsafe-url"><base target="_top">'
        '<link rel="preconnect" href="https://u1.evil.tk"><style>a > b { color: red }</style>'
        '<script>alert(1)</script></head><body onload="x()">'
        '<a href="https://evil.tk/v" target="_self" ping="https://evil.tk/ping">Verify</a>'
        '<iframe srcdoc="<b>x</b>"><p>hidden</p></iframe><svg><a xlink:href="https://evil.tk/s">s</a></svg>'
        '<p style="color:blue">5 &lt; 6 &amp; ok</p><img src="https://evil.tk/px.gif"></body></html>')
safe = data.safe_html(EVIL)
for bad in ("evil.tk/r", "evil.tk/v", "evil.tk/ping", "evil.tk/s", "u1.evil", "<meta", "<link",
            "<base", "<script", "alert(1)", "onload", "target=", "<iframe", "srcdoc", "hidden"):
    assert bad not in safe, (bad, safe)
assert "a > b { color: red }" in safe, "CSS must pass through unescaped"
assert 'style="color:blue"' in safe and "5 &lt; 6 &amp; ok" in safe and ">Verify</a>" in safe
assert 'src="https://evil.tk/px.gif"' in safe, "images stay: the frame's CSP gates them"
# Foreign content: a browser reads <style> inside <svg>/<math> as markup.
for evil in ('<svg><style><a href="https://evil.tk/v" target="_self">x</a>'
             '<meta http-equiv="refresh" content="0;url=https://evil.tk/r"></style></svg>',
             '<svg><a><set attributeName="href" to="https://evil.tk/s"/>s</a></svg>',
             '<math><style><a href="https://evil.tk/m">m</a></style></math>'):
    assert "evil.tk" not in data.safe_html(evil), evil
print("mail parser self-check passed")
