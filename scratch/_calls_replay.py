"""27 real two-party calls, scored from the CACHED transcripts (no Whisper).

Same texts `--calls` gets from analyse(); word timings are faked at 2.5 wps, so
this is the transcript path, not the audio path - it reproduces the ranking, not
the exact alert timing. ponytail: replay, so the honest external voice number is
checkable in seconds instead of 16 minutes of GPU.
"""
import json

import data  # before torch
import numpy as np
import voice_model as vm
from sklearn.metrics import roc_auc_score

idx = json.load(open(data.DATA / "vishing" / "calls" / "index.json"))
txt = json.load(open(data.DATA / "vishing" / "calls" / "transcripts.json"))
rows = [(x["label"], vm.score_transcript(txt[x["id"]])) for x in idx if x["id"] in txt]
y = np.array([l for l, _ in rows])
pk = np.array([v["peak_risk"] for _, v in rows])
al = np.array([v["alert_at_s"] is not None for _, v in rows])
print(f"{len(y)} real calls: {y.sum()} scam / {(1 - y).sum()} legit")
print(f"  peak-risk AUC {roc_auc_score(y, pk):.3f}")
for lab, name in [(1, "scam"), (0, "legit")]:
    m = y == lab
    print(f"  {name:<6} peak mean {pk[m].mean():.3f}  alerts {al[m].mean():.0%}")
tp, fp, fn = (al & (y == 1)).sum(), (al & (y == 0)).sum(), (~al & (y == 1)).sum()
print(f"  alert P {tp / max(tp + fp, 1):.2f} R {tp / max(tp + fn, 1):.2f}")
