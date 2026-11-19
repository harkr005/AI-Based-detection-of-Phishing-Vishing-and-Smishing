"""What the SMS model does on one real inbox, before and after it has seen one.

Three numbers, always together:
  in-source     held-out groups of the corpora it trained on
  cross-source  Dataset_5971, a corpus it never sees
  inbox FP      % of 943 real, carrier-stamped, legitimate messages called
                smishing - the only number measured on traffic rather than on a
                dataset, and the one the user actually reported.

Run it before the retrain to get the baseline, and after to get the delta.
Held-out inbox groups are reported separately from the whole file: once the
corpus is in training, the whole-file number is partly in-sample and only the
held-out one is honest.
"""
import pathlib
import sys

import data  # must precede any torch/pandas-csv import; see project notes
import numpy as np
import pandas as pd
import sms_model as sm
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = pathlib.Path(data.__file__).resolve().parent
INBOX = ROOT / "Dataset(Indian)" / "sms" / "inbox_sms.csv"
SMISH = sm.LABELS.index("smishing")


def splits():
    """The exact split sms_model.__main__ trains on, so the numbers compare."""
    s = data.load_sms()
    held = s[s.source == "Dataset_5971"]
    pool = s[s.source != "Dataset_5971"]
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
                      .split(pool, groups=pool.group))
    return pool.iloc[i_tr], pool.iloc[i_te], held


def macro(model, df):
    pred = model.predict_proba(sm._clean(df.text.fillna(""))).argmax(1)
    return f1_score(df.label_3.map(sm.LABELS.index).values, pred, average="macro")


def smishing_recall(model, df):
    rows = df[df.label_3 == "smishing"]
    if rows.empty:
        return float("nan")
    pred = model.predict_proba(sm._clean(rows.text.fillna(""))).argmax(1)
    return (pred == SMISH).mean()


def inbox_rates(model, train_groups=frozenset()):
    inbox = pd.read_csv(INBOX)
    pred = model.predict_proba(sm._clean(inbox.text.fillna(""))).argmax(1)
    inbox["called_smishing"] = pred == SMISH
    unseen = inbox[~inbox.group.isin(train_groups)]
    return inbox, unseen


def run(tag, model, train_groups=frozenset()):
    tr, te, held = splits()
    inbox, unseen = inbox_rates(model, train_groups)
    print(f"\n=== {tag} " + "=" * (58 - len(tag)))
    print(f"  in-source macro-F1        {macro(model, te):.3f}   (n={len(te):,})")
    print(f"  cross-source macro-F1     {macro(model, held):.3f}   (n={len(held):,})")
    print(f"  smishing recall, cross    {smishing_recall(model, held):.3f}")
    print(f"  smishing recall, india    {smishing_recall(model, te[te.source == 'india']):.3f}")
    print(f"  INBOX false positives     {100 * inbox.called_smishing.mean():5.2f}%  "
          f"({inbox.called_smishing.sum()}/{len(inbox)})")
    if len(unseen) != len(inbox):
        print(f"    of which unseen groups  {100 * unseen.called_smishing.mean():5.2f}%  "
              f"({unseen.called_smishing.sum()}/{len(unseen)})")
    worst = inbox[inbox.called_smishing].text.str.slice(0, 70)
    for t in worst.head(8):
        print(f"      ! {t}")
    return inbox


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "current model on disk"
    model = sm.load()
    # Groups the shipped model saw, if the corpus is already wired into load_sms.
    tr, _, _ = splits()
    seen = frozenset(tr.group.unique()) if (tr.source == "inbox").any() else frozenset()
    run(tag, model, seen)
