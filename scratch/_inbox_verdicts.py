"""The number the user actually sees: whole /sms verdicts over one real inbox,
old model against new.

`_inbox_sms_eval.py` measures the content model alone. This runs all 943
messages back through `api.sms` - fusion, behaviour, URL model and the bands -
because a content-model false positive only matters if it survives fusion, and
a SUSPICIOUS badge on a bank message is what got reported.

Both models are scored with identical inputs, so the delta is clean even though
the replay cannot reproduce the phone's own first_contact for every sender.
"""
import datetime as dt
import json
import pathlib
import collections
import sys

import data  # before anything that imports torch
import api
import sms_model as sm

ROOT = pathlib.Path(data.__file__).resolve().parent
# argv[1] replays another dump, e.g. app-reports/audit2.jsonl (a second inbox).
AUDIT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "app-reports" / "audit.jsonl"
BASELINE = pathlib.Path(
    r"C:\Users\DELL\AppData\Local\Temp\claude\E--major-project"
    r"\94ba24b8-492a-4e84-a34c-8653c56e999b\scratchpad\sms_charlr.baseline.pkl")


def replay(tag, pkl):
    sm.OUT = pkl
    sm.load.cache_clear()
    verdicts, labels, flagged = collections.Counter(), collections.Counter(), []
    for line in AUDIT.open(encoding="utf-8"):
        m = json.loads(line)
        body = " ".join((m.get("body") or "").split())
        if not body:
            continue
        api._STORES.clear()
        out = api.sms(api.SmsIn(
            text=body, sender=m["sender"], first_contact=False,
            sent_at=dt.datetime.fromtimestamp(m["sent_at"] / 1000, dt.timezone.utc),
            user_id="replay", verified_sender=False))
        verdicts[out["verdict"]] += 1
        labels[out["label"]] += 1
        if out["verdict"] != "SAFE":
            flagged.append((out["risk"], m["sender"], body[:72]))
    total = sum(verdicts.values())
    print(f"\n=== {tag}  (n={total})")
    print("  verdicts:", dict(verdicts))
    print("  labels  :", dict(labels))
    print(f"  NOT SAFE: {total - verdicts['SAFE']} "
          f"({100 * (total - verdicts['SAFE']) / total:.1f}%)")
    for risk, sender, text in sorted(flagged, reverse=True)[:10]:
        print(f"    {risk:.3f}  {sender:<15} {text}")
    return verdicts


if __name__ == "__main__":
    before = replay("BEFORE - model that has never seen an inbox", BASELINE)
    after = replay("AFTER  - inbox corpus in training", ROOT / "models" / "sms_charlr.pkl")
    if len(sys.argv) == 1:  # audit2 is NOT all-legitimate: job scams on -P headers
        print("\n  every one of these messages is legitimate: carrier-stamped DLT")
        print("  headers, read and checked. SAFE is the only correct verdict.")
    print(f"  false alarms: {sum(before.values()) - before['SAFE']} -> "
          f"{sum(after.values()) - after['SAFE']}")
