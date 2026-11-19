"""The mirror of _inbox_verdicts.py: what the retrain COST on real smishing.

Cutting false alarms is only honest next to the number it was traded against,
so this runs the held-out corpus - Dataset_5971, which neither model has ever
trained on - through the same whole-scan path and counts how many of its
smishing messages still come out flagged.
"""
import datetime as dt
import pathlib

import data
import api
import sms_model as sm

ROOT = pathlib.Path(data.__file__).resolve().parent
BASELINE = pathlib.Path(
    r"C:\Users\DELL\AppData\Local\Temp\claude\E--major-project"
    r"\94ba24b8-492a-4e84-a34c-8653c56e999b\scratchpad\sms_charlr.baseline.pkl")
WHEN = dt.datetime(2026, 9, 23, 14, 30, tzinfo=dt.timezone.utc)


def replay(tag, pkl, rows):
    sm.OUT = pkl
    sm.load.cache_clear()
    caught = missed = 0
    examples = []
    for text in rows:
        api._STORES.clear()
        out = api.sms(api.SmsIn(text=text, sender="+919876501234", first_contact=True,
                                sent_at=WHEN, user_id="replay", verified_sender=False))
        if out["verdict"] == "SAFE":
            missed += 1
            examples.append(text[:70])
        else:
            caught += 1
    print(f"  {tag:<38} caught {caught}/{caught + missed} "
          f"({100 * caught / (caught + missed):.1f}%)")
    return set(examples)


if __name__ == "__main__":
    s = data.load_sms()
    smish = s[(s.source == "Dataset_5971") & (s.label_3 == "smishing")].text.fillna("").tolist()
    print(f"held-out smishing, never trained on either way (n={len(smish)})")
    before = replay("BEFORE - no inbox in training", BASELINE, smish)
    after = replay("AFTER  - inbox in training", ROOT / "models" / "sms_charlr.pkl", smish)
    newly_missed = after - before
    print(f"\n  newly missed by the retrain: {len(newly_missed)}")
    for t in sorted(newly_missed)[:10]:
        print(f"    - {t}")
    recovered = before - after
    print(f"  newly caught by the retrain: {len(recovered)}")
    for t in sorted(recovered)[:5]:
        print(f"    + {t}")
