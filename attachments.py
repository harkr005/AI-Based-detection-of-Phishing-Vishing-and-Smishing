"""Email attachments: the payload a phishing mail carries besides its body.

Two checks, both in memory - an attachment is never written to disk or stored:
  1. File type. A file that runs code, installs settings or mounts as a drive
     the moment it is opened (RUNS), an Office file with macros (MACRO), or a
     zip holding one / locked with a password. One of those is treated like a
     link Google has listed: it fills the link slot at safe_browsing.LISTED_RISK.
  2. Links inside PDF and text-like attachments (HTML, SVG, calendar invites),
     handed to the same mail link model as links in the body (api._links).

What was in the mail decided the lists (scratch/_attach_eval.py has the counts):
phishing_pot carries .mobileconfig, .iso, .ics invites and link-bearing PDFs;
the personal mailboxes carry 292 mails with .htm attachments, so an HTML file is
NOT flagged for its type, only for the links inside it.

ponytail: .rar/.7z are not opened (no stdlib reader) and abstain; QR codes in
images are not read (needs an image library) - phishing_pot has ~230 mails
each with .jpg/.png, so that is the next layer if measured worth it.
"""
import email
import html
import io
import os
import zipfile
from email import policy

import data

RUNS = {".exe", ".scr", ".com", ".pif", ".bat", ".cmd", ".msi", ".msix", ".appx",
        ".js", ".jse", ".vbs", ".vbe", ".wsf", ".wsh", ".hta", ".ps1", ".lnk", ".url",
        ".cpl", ".jar", ".apk", ".reg", ".chm", ".inf", ".scf", ".xll", ".one",
        ".iso", ".img", ".vhd", ".vhdx", ".dmg", ".mobileconfig", ".application"}
MACRO = {".docm", ".dotm", ".xlsm", ".xltm", ".xlam", ".pptm", ".potm", ".ppam", ".sldm"}
TEXTY = {".html", ".htm", ".shtml", ".xhtml", ".svg", ".ics", ".xml", ".txt"}
MAX_PDF_PAGES = 30              # a statement runs a few pages; a 500-page PDF is not read whole

SAY = {**{e: "a file that runs a program or changes your device when opened" for e in RUNS},
       **{e: "an Office file with macros, which can run code when opened" for e in MACRO},
       ".iso": "a disk image, a common way to slip malware past mail filters",
       ".img": "a disk image, a common way to slip malware past mail filters",
       ".vhd": "a disk image, a common way to slip malware past mail filters",
       ".vhdx": "a disk image, a common way to slip malware past mail filters",
       ".mobileconfig": "an iPhone/Mac settings profile, which can take control of the device",
       ".lnk": "a Windows shortcut that can run a hidden command",
       ".one": "a OneNote file, a common way to hide a program behind a fake button"}


def _ext(name):
    return os.path.splitext(name)[1].lower()


def _why(name):
    """-> plain reason this filename is dangerous, or None."""
    e = _ext(name)
    if e not in RUNS and e not in MACRO:
        return None
    inner = _ext(os.path.splitext(name)[0])
    disguise = f", disguised as {inner}" if inner and inner not in RUNS | MACRO else ""
    return SAY[e] + disguise


def _zip(blob):
    """-> reason the zip is dangerous, or None. Names only; nothing is extracted."""
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
        infos = z.infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        return None                                   # unreadable: abstain
    for i in infos:
        if _why(i.filename):
            return f"a zip holding {_why(i.filename)}"
    if any(i.flag_bits & 1 for i in infos):
        return "a password-protected zip, which mail scanners cannot look inside"
    return None


def _pdf_urls(blob):
    import logging
    import pypdf
    logging.getLogger("pypdf").setLevel(logging.ERROR)   # broken PDFs are routine, not news
    try:
        r = pypdf.PdfReader(io.BytesIO(blob))
        if r.is_encrypted and not r.decrypt(""):
            return []                                 # bank statements: locked, abstain
        out = []
        for page in r.pages[:MAX_PDF_PAGES]:
            for a in page.get("/Annots") or []:
                act = a.get_object().get("/A") or {}
                uri = act.get("/URI") if hasattr(act, "get") else None
                if uri:
                    out.append(str(uri))
        return out
    except Exception:                                 # malformed PDFs are routine in phishing
        return []


def scan(raw):
    """RFC 822 bytes -> {"count": n attachments, "bad": [{"type", "why"}],
    "urls": [(url, file type)]}. No filename is ever returned, only its type
    (".zip"): the scan result is stored, and a name like "Form16_<PAN>.zip"
    would carry the user's private data into it."""
    m = email.message_from_bytes(raw, policy=policy.compat32)
    out = {"count": 0, "bad": [], "urls": []}
    for p in m.walk():
        name = p.get_filename()
        if p.is_multipart() or not name:
            continue
        name = data._hdr(name).strip()
        out["count"] += 1
        blob = p.get_payload(decode=True) or b""
        e = _ext(name)
        why = _why(name) or (_zip(blob) if e == ".zip" else None)
        if why:
            out["bad"].append({"type": e, "why": why})
        if e == ".pdf":
            urls = _pdf_urls(blob)
        elif e in TEXTY:
            urls = data._HREF_URL.findall(html.unescape(blob.decode("utf-8", "replace")))
        else:
            urls = []
        out["urls"] += [(u, e) for u in dict.fromkeys(urls)]
    return out


def _mail(*files):
    from email.message import EmailMessage
    m = EmailMessage()
    m["From"], m["Subject"] = "a@b.com", "hi"
    m.set_content("see attached")
    for name, blob in files:
        m.add_attachment(blob, maintype="application", subtype="octet-stream", filename=name)
    return m.as_bytes()


if __name__ == "__main__":
    import pypdf
    from pypdf.annotations import Link

    w = pypdf.PdfWriter()
    w.add_blank_page(200, 200)
    w.add_annotation(page_number=0, annotation=Link(rect=(0, 0, 50, 50), url="http://evil.tk/login"))
    pdf = io.BytesIO()
    w.write(pdf)

    z_bad, z_lock, z_ok = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(z_bad, "w") as z:
        z.writestr("invoice.pdf.exe", b"MZ")
    with zipfile.ZipFile(z_lock, "w") as z:
        z.writestr("doc.pdf", b"x")
    lock = bytearray(z_lock.getvalue())               # zipfile cannot encrypt: set the "encrypted"
    lock[lock.find(b"PK\x01\x02") + 8] |= 1          # bit in the central directory by hand
    with zipfile.ZipFile(z_ok, "w") as z:
        z.writestr("report.csv", b"a,b")

    r = scan(_mail(("Invoice.PDF.exe", b"MZ"), ("q.xlsm", b"x"), ("a.mobileconfig", b"x"),
                   ("b.zip", z_bad.getvalue()), ("c.zip", bytes(lock)), ("d.zip", z_ok.getvalue()),
                   ("statement.pdf", pdf.getvalue()), ("broken.pdf", b"%PDF-1.4 garbage"),
                   ("invite.ics", b"DESCRIPTION:join https://evil.tk/meet?a=1&amp;b=2\r\n"),
                   ("page.htm", b'<a href="https://ok.com/x">x</a>'), ("photo.jpg", b"\xff\xd8")))
    bad = [(b["type"], b["why"]) for b in r["bad"]]
    assert r["count"] == 11, r["count"]
    assert [t for t, _ in bad] == [".exe", ".xlsm", ".mobileconfig", ".zip", ".zip"], bad
    assert "disguised as .pdf" in bad[0][1]
    assert ".pdf" in bad[3][1] and "password" in bad[4][1]
    assert ("http://evil.tk/login", ".pdf") in r["urls"]
    assert ("https://evil.tk/meet?a=1&b=2", ".ics") in r["urls"]
    assert ("https://ok.com/x", ".htm") in r["urls"] and len(r["urls"]) == 3
    assert scan(_mail()) == {"count": 0, "bad": [], "urls": []}
    print("self-check ok")
