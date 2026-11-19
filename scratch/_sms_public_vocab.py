"""SMS model whose word list comes from PUBLIC SMS only (2026-10-28).

The shipped sms_charlr stores its char n-gram vocabulary in plain text, inside
the APK, and that vocabulary was fitted on every training row - including the
private inbox, so name fragments of the inbox owner are in it. Here the
TfidfVectorizer (same settings) is fitted on non-inbox training rows only; the
logistic regression is still fitted on ALL training rows, inbox included. A
private-only string can then never become a feature: every term must occur in
>= 3 public rows. The inbox still teaches weights over public n-grams.

Pass rule, written before the first run. The candidate is promotable only if:
  1. whole-scan false alarms on the 943-message inbox (app-reports/audit.jsonl,
     the _inbox_verdicts.py replay): candidate <= shipped + 2
  2. held-out smishing caught (Dataset_5971, the _smish_verdicts.py replay):
     candidate >= shipped - 2
  3. cross-source macro-F1 >= 0.865 (shipped 0.870)
  4. zero vocabulary terms that occur only in inbox rows

Counts only. No message text is printed. Replays run with Safe Browsing and the
Postgres profile store stubbed out, identically for both models, so nothing
leaves the machine and no "replay" profiles are written.

Writes the candidate to Datasets/candidates/sms_charlr_publicvocab.pkl, never
models/. Promotion is /promote-model's job.

Result, 2026-10-28 (shipped -> candidate):
  inbox false alarms 1/943 -> 0/943          PASS
  held-out smishing  142/187 -> 139/187      FAIL (rule allowed 140)
  cross-source F1    0.870 -> 0.859          FAIL
  in-source F1       0.984 -> 0.980
  vocabulary         75,839 -> 62,527 terms; "mohak"/"kukal" terms 2 -> 0
  rule 4             PASS - every candidate term is in >= 3 public training
                     rows. The first run printed 4,494 here because it probed
                     the raw public corpus, not the scrambled public twins that
                     adversarial.augment adds and the vectorizer also saw; the
                     check below now probes the training rows.
It fails the rule set before the run. The cost is the Indian bank/UPI wording
that only the inbox supplied (13k terms dropped). Mohak's call, 2026-10-28:
accept it anyway - the repo is going public, and no private string may ship.
sms_model.train() now fits the vocabulary this way.
"""
import collections
import datetime as dt
import json
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import data  # before anything that imports torch
import adversarial
import api
import behaviour
import sms_model as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

ROOT = pathlib.Path(data.__file__).resolve().parent
SHIPPED = ROOT / "models" / "sms_charlr.pkl"
CAND = ROOT / "Datasets" / "candidates" / "sms_charlr_publicvocab.pkl"
AUDIT = ROOT / "app-reports" / "audit.jsonl"
WHEN = dt.datetime(2026, 9, 23, 14, 30, tzinfo=dt.timezone.utc)

# Same scan path, nothing external: no Google lookups, no Postgres profiles.
api.safe_browsing.check = lambda urls: {}
api.store.load = lambda user: behaviour.new_store()
api.store.save = lambda *a, **k: None


def split():
    """The exact split sms_model.__main__ uses."""
    s = data.load_sms()
    held = s[s.source == "Dataset_5971"]
    pool = s[s.source != "Dataset_5971"]
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
                      .split(pool, groups=pool.group))
    tr, te = pool.iloc[i_tr], pool.iloc[i_te]
    tr = data.pd.concat([tr, adversarial.augment(tr)], ignore_index=True)
    return s, tr, te, held


def train(tr):
    pipe = sm.build()
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    public = tr[tr.source != "inbox"]
    vec.fit(sm._clean(public.text.fillna("")))
    clf.fit(vec.transform(sm._clean(tr.text.fillna(""))),
            tr.label_3.map(sm.LABELS.index).values)
    return pipe


def inbox_only_terms(pipe, tr):
    """Vocabulary terms in fewer than 3 public TRAINING rows (scrambled twins
    included, since the vectorizer saw them) - i.e. terms only private text
    could have put there. min_df=3 makes this 0 by construction for the
    candidate; the check guards a mislabelled `source`."""
    vocab = pipe.steps[0][1].vocabulary_
    probe = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), vocabulary=vocab)
    public = probe.transform(sm._clean(tr[tr.source != "inbox"].text.fillna("")))
    return int((public.getnnz(axis=0) < 3).sum())


def false_alarms(pkl):
    sm.OUT = pkl
    sm.load.cache_clear()
    v = collections.Counter()
    for line in AUDIT.open(encoding="utf-8"):
        m = json.loads(line)
        body = " ".join((m.get("body") or "").split())
        if not body:
            continue
        api._STORES.clear()
        v[api.sms(api.SmsIn(
            text=body, sender=m["sender"], first_contact=False,
            sent_at=dt.datetime.fromtimestamp(m["sent_at"] / 1000, dt.timezone.utc),
            user_id="replay", verified_sender=False))["verdict"]] += 1
    return sum(v.values()) - v["SAFE"], sum(v.values())


def smish_caught(pkl, rows):
    sm.OUT = pkl
    sm.load.cache_clear()
    caught = 0
    for text in rows:
        api._STORES.clear()
        caught += api.sms(api.SmsIn(
            text=text, sender="+919876501234", first_contact=True, sent_at=WHEN,
            user_id="replay", verified_sender=False))["verdict"] != "SAFE"
    return caught, len(rows)


if __name__ == "__main__":
    s, tr, te, held = split()
    print(f"train {len(tr):,} ({(tr.source == 'inbox').sum():,} inbox rows, "
          f"used for weights only)  in-source {len(te):,}  cross-source {len(held):,}")
    cand = train(tr)
    CAND.parent.mkdir(parents=True, exist_ok=True)
    with CAND.open("wb") as fh:
        pickle.dump(cand, fh)

    with SHIPPED.open("rb") as fh:
        shipped = pickle.load(fh)
    rows = {}
    for tag, pipe, pkl in (("shipped", shipped, SHIPPED), ("candidate", cand, CAND)):
        vocab = pipe.steps[0][1].vocabulary_
        r = rows[tag] = {
            "vocab": len(vocab),
            "inbox_only": inbox_only_terms(pipe, tr),
            "name_terms": sum(("mohak" in t or "kukal" in t) for t in vocab),
            "in_f1": sm.report(f"{tag} in-source", pipe, te),
            "cross_f1": sm.report(f"{tag} cross-source", pipe, held),
            "fa": false_alarms(pkl),
            "smish": smish_caught(pkl, held[held.label_3 == "smishing"].text.fillna("").tolist()),
        }
    print(f"\n{'':<24}{'shipped':>12}{'candidate':>12}")
    for k in ("vocab", "inbox_only", "name_terms", "in_f1", "cross_f1"):
        a, b = rows["shipped"][k], rows["candidate"][k]
        f = (lambda x: f"{x:.3f}") if isinstance(a, float) else (lambda x: f"{x:,}")
        print(f"{k:<24}{f(a):>12}{f(b):>12}")
    for k in ("fa", "smish"):
        a, b = rows["shipped"][k], rows["candidate"][k]
        print(f"{k:<24}{f'{a[0]}/{a[1]}':>12}{f'{b[0]}/{b[1]}':>12}")

    sh, cd = rows["shipped"], rows["candidate"]
    checks = {
        "1 inbox false alarms <= shipped + 2": cd["fa"][0] <= sh["fa"][0] + 2,
        "2 smishing caught >= shipped - 2": cd["smish"][0] >= sh["smish"][0] - 2,
        "3 cross-source macro-F1 >= 0.865": cd["cross_f1"] >= 0.865,
        "4 no inbox-only vocabulary terms": cd["inbox_only"] == 0,
    }
    print()
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"\ncandidate: {CAND}")
    sys.exit(0 if all(checks.values()) else 1)
