"""What does relabelling Gmail's Promotions box as SPAM cost the ham side?

THE POLICY UNDER TEST. `data.load_personal_mail` calls everything outside
Spam/Trash/Junk ham, so `Category promotions` (5,231 msgs here) trains and scores
as ham, and `_train_unified.py` HAM_BOXES puts it in v3's modern ham half. The
proposal is to move that box to the spam class while keeping Gmail's other four
boxes (Inbox / social / updates / Archived) as ham and keeping the whole email
stack unchanged - content model, URL model, solicitation rule, fusion.

WHY THIS IS NOT FREE. `updates` is the SAME GENRE as `promotions`: bulk HTML mail
from brands. Gmail separates them by intent (offer vs receipt/notice), not by
anything a 256-token DistilBERT sees. So the question is not whether the model
can learn "promotions", it is how much of `updates` and `Inbox` it drags across
with it. That is the ham cost, and it is what this script measures.

TWO PHASES.
  (default)  score the checkpoints already on disk against the RElabelled truth.
             Answers "what does today's stack do", and sizes the denominator
             change - the ham set loses 5,231 messages the moment you relabel.
  --train    retrain the v3 recipe with promotions in the spam class and score
             the ham boxes again. This is the number that actually answers the
             question; the phase above cannot, because v3 was TRAINED on
             promotions as ham.

HELD OUT. `Inbox`, `Category social` and `Archived` were never in any training
run (v3's modern ham is promotions+updates <=2024 only), so they are clean ham
under both labellings. `updates` 2025-26 is held out by year. user2 is a second
mailbox, promotions+updates only.

PRIVATE DATA - rates and sender domains may go in the report, message text may not.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import data                      # noqa: E402  # before torch, see sms_model
import numpy as np
import pandas as pd

import adversarial
import email_model as em

ROOT = pathlib.Path(data.__file__).resolve().parent
PROMO, UPDATES = "Category promotions", "Category updates"
HAM, SPAM, PHISH = (em.LABELS.index(x) for x in ("ham", "spam", "phishing"))
EN = set("the and you your account for that with this have from will not are was "
         "please click here".split())
english = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(
    re.findall(r"[a-z]+", str(t).lower()[:2000])) >= 0.04


def mail(person, cache):
    """Personal mail for one person, parquet-cached - the mbox key scan is GBs."""
    f = data.PERSONAL / cache
    if f.exists():
        return pd.read_parquet(f)
    d = data.load_personal_mail(person=person, limit=6000)
    d.to_parquet(f)
    return d


def boxes():
    """-> DataFrame of all real mail on disk, with both labellings attached."""
    d = pd.concat([mail("self", "_self_6000.parquet"),
                   mail("user2", "_user2_6000.parquet")], ignore_index=True)
    d = d[d.text.str.len() > 5].copy()
    # label_now = what ships today.  label_new = the policy under test.
    d["label_now"] = d.label_3
    d["label_new"] = np.where(d.mbox == PROMO, "spam", d.label_3)
    return d


def score_all(d, models):
    """-> {tag: pred array} aligned to d. One checkpoint load per model."""
    out = {}
    for tag, name in models:
        em.OUT = ROOT / "models" / name
        em.load.cache_clear()
        tok, model = em.load()
        pred = em.predict(tok, model, d.text)[0].argmax(1)
        # The shipped LABEL is not argmax alone: api.py promotes ham -> spam when
        # the solicitation rule fires. Same stack here or this is not the stack.
        sol = np.array([bool(em.solicitation(t)) for t in d.text])
        out[tag] = np.where((pred == HAM) & sol, SPAM, pred)
    return out


def table(d, preds, models, slices):
    w = 17
    print(f"\n{'slice':44s} {'n':>6}" + "".join(f"{t:>{w}s}" for t, _ in models))
    print(f"{'':44s} {'':>6}" + "".join(f"{'ham/spam/phish%':>{w}s}" for _ in models))
    for label, mask in slices:
        if mask is None:
            print(label)
            continue
        f = d[mask]
        if not len(f):
            continue
        row = f"{label:44s} {len(f):6d}"
        for t, _ in models:
            p = preds[t][mask.values]
            row += (f"{100 * (p == HAM).mean():6.1f}/"
                    f"{100 * (p == SPAM).mean():5.1f}/"
                    f"{100 * (p == PHISH).mean():4.1f}")
        print(row, flush=True)


def ham_summary(d, preds, models, col):
    """The answer to the question: FP rate over whatever `col` calls ham."""
    m = (d[col] == "ham").values
    print(f"\n  ham side under `{col}`: {m.sum():,} messages "
          f"({', '.join(f'{k} {v:,}' for k, v in d[m].mbox.value_counts().items())})")
    for t, _ in models:
        p = preds[t][m]
        print(f"    {t:22s} FP {100 * (p != HAM).mean():5.2f}%  "
              f"(spam {100 * (p == SPAM).mean():5.2f}%, phish {100 * (p == PHISH).mean():5.2f}%)")


def train_variant():
    """v3's recipe from _train_unified.py, one line changed: promotions -> spam."""
    e = data.load_email()
    aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
    aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
    ph_tr, _ = em.phishing_split(e)
    old_tr = pd.concat([aug[aug.date.dt.day <= 7], ph_tr])

    pot = data.load_phishing_pot()
    pot = pot[pot.text.map(english) & pot.date.dt.year.between(2022, 2026)].copy()
    pot["label_3"] = "phishing"

    d = boxes()
    m = d[d.mbox.isin({PROMO, UPDATES}) & (d.label_3 == "ham")
          & d.date.dt.year.between(2021, 2026)].copy()
    m["label_3"] = m.label_new          # <- THE ONE CHANGE. promotions is spam now.
    modern = pd.concat([pot, m], ignore_index=True)
    tr = pd.concat([old_tr, modern[modern.date.dt.year <= 2024]], ignore_index=True)
    tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)
    print("train", len(tr), dict(tr.label_3.value_counts()), flush=True)

    em.OUT = ROOT / "models" / "email_promospam"
    em.load.cache_clear()
    return em.train(tr)


if __name__ == "__main__":
    if "--train" in sys.argv:
        train_variant()

    d = boxes().reset_index(drop=True)
    print(d.groupby(["person", "mbox", "label_3"]).size().to_string(), flush=True)

    models = [("v3 SHIPPED", "email_distilbert_v3")]
    if (ROOT / "models" / "email_promospam" / "model.pt").exists():
        models.append(("promo=spam", "email_promospam"))

    self_, har = d.person == "self", d.person == "user2"
    y25 = d.date.dt.year >= 2025
    slices = [
        ("SELF Inbox ham (never trained, any model)", self_ & (d.mbox == "Inbox") & (d.label_3 == "ham")),
        ("SELF social ham (never trained, any model)", self_ & (d.mbox == "Category social")),
        ("SELF updates ham 2025-26 (held out by year)", self_ & (d.mbox == UPDATES) & y25),
        ("SELF updates ham <=2024 (IN v3 train)", self_ & (d.mbox == UPDATES) & ~y25),
        ("USER2 updates ham 2025-26", har & (d.mbox == UPDATES) & y25),
        ("USER2 updates ham <=2024 (IN v3 train)", har & (d.mbox == UPDATES) & ~y25),
        ("--- the relabelled box (was ham, now spam) ---", None),
        ("SELF promotions 2025-26 (held out by year)", self_ & (d.mbox == PROMO) & y25),
        ("SELF promotions <=2024", self_ & (d.mbox == PROMO) & ~y25),
        ("USER2 promotions 2025-26", har & (d.mbox == PROMO) & y25),
        ("USER2 promotions <=2024", har & (d.mbox == PROMO) & ~y25),
        ("--- Gmail's own spam boxes ---", None),
        ("Gmail Spam, both people", d.label_3 == "spam"),
    ]
    preds = score_all(d, models)
    table(d, preds, models, slices)

    print("\n" + "=" * 78)
    print("THE HAM SIDE - same predictions, two definitions of what ham is")
    ham_summary(d, preds, models, "label_now")
    ham_summary(d, preds, models, "label_new")

    # A rate is not a finding; the senders are (same move as _inbox_eval.py).
    keep = d[d.label_new == "ham"]
    tot = keep.sender.str.extract(r"@([\w.-]+)", expand=False).str.lower().value_counts()
    for t, _ in models:
        p = preds[t][(d.label_new == "ham").values]
        fp = keep[p != HAM]
        if len(fp):
            dom = fp.sender.str.extract(r"@([\w.-]+)", expand=False).str.lower()
            print(f"\n{t} flags {len(fp)}/{len(keep)} of the NEW ham set: " + ", ".join(
                f"{k} {v}/{tot.get(k, v)}" for k, v in dom.value_counts().head(8).items()))
