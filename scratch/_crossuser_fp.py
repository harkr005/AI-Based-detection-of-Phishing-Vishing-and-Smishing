"""What does the modern email model actually false-positive ON, in a mailbox it
has never seen? A rate is not a finding; the sender list is.

Loads ONE person, so it runs in seconds. Prints sender domains only - no message
text leaves this script.
"""
import pathlib

import data                      # before torch
import pandas as pd
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
d = data.load_personal_mail(person="user2")
ham = d[d.label_3 == "ham"]

for name in ("email_modern", "email_distilbert_v2"):
    em.OUT, _ = ROOT / "models" / name, em.load.cache_clear()
    tok, model = em.load()
    p, _ = em.predict(tok, model, ham.text)
    fp = ham[p.argmax(1) == em.LABELS.index("phishing")]
    dom = fp.sender.str.extract(r"@([\w.-]+)", expand=False).str.lower()
    all_dom = ham.sender.str.extract(r"@([\w.-]+)", expand=False).str.lower()
    print(f"\n{name}: {len(fp)}/{len(ham)} = {100*len(fp)/len(ham):.1f}% flagged, "
          f"{dom.nunique()} distinct sender domains of {all_dom.nunique()}")
    t = pd.DataFrame({"flagged": dom.value_counts()}).join(
        all_dom.value_counts().rename("total")).head(12)
    t["rate"] = (100 * t.flagged / t.total).round(0)
    print(t.to_string())
