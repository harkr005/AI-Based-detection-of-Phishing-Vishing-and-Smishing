"""Audio-path A/B on the 27 recorded calls, without re-running Whisper each time.

`--calls` re-transcribes all 27 (16 min) for every configuration, which is why
the audio path was only ever measured once per change. The word TIMINGS are what
the transcript replay cannot fake - windows() cuts on them - so they are cached
here once and every later configuration scores in seconds.

Usage:  python _calls_audio.py <checkpoint-dir> <stream 0|1>
"""
import json
import sys
from pathlib import Path

import data                       # before torch
import numpy as np
import act_model
import voice_model as vm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]   # repo root: this file lives in scratch/
WORDS = data.DATA / "vishing" / "calls" / "words.json"
meta = {c["id"]: c for c in json.load(open(data.DATA / "vishing" / "calls" / "index.json"))}

if not WORDS.exists():
    print("transcribing once with timings (~16 min)", flush=True)
    out = {}
    for i, (cid, c) in enumerate(meta.items(), 1):
        out[cid] = [list(w) for w in vm.transcribe(ROOT / c["path"])]
        json.dump(out, open(WORDS, "w"))
        print(f"  {i}/{len(meta)}", flush=True)
words = {k: [tuple(w) for w in v] for k, v in json.load(open(WORDS)).items()}

ck, stream = sys.argv[1], sys.argv[2] == "1"
act_model.OUT = ROOT / "models" / ck
act_model.load.cache_clear()
vm.STREAM_ACTS = stream

rows = []
for cid, w in words.items():
    txt = " ".join(x for _, _, x in w)
    if sum(ord(c) > 127 for c in txt) / max(len(txt), 1) >= 0.02:    # English only
        continue
    v = vm.verdict(vm.windows(w), text=txt)
    rows.append((meta[cid]["label"], v["peak_risk"], v["alert_at_s"] is not None))
y, pk, al = (np.array(x) for x in zip(*rows))
print(f"{ck}  stream={int(stream)}  n={len(y)}  "
      f"AUC {roc_auc_score(y, pk):.3f}   "
      f"scam peak {pk[y == 1].mean():.3f} alerts {al[y == 1].mean():.0%}   "
      f"legit peak {pk[y == 0].mean():.3f} alerts {al[y == 0].mean():.0%}")
