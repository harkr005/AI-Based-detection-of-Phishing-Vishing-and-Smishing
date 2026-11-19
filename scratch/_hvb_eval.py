"""The 2x2 that the three fixes were built for.

Rows are the two changes made to the act layer:
  ACL BACKGROUND NEGATIVES - `act_distilbert_bg`, trained with the scam calls'
    evidence spans cut out as all-zero rows plus the spans alone as one-hot rows
    (act_model.background). The paper's claim is that this stops the detector
    fitting the narrative around the act.
  STREAMING - score acts per 15 s window and ratchet, instead of once on the
    finished transcript, so an alert can exist before the caller hangs up.

Columns are what actually matters:
  HVB FP     false alarms on 1,446 real recorded LEGITIMATE bank calls, the
             negatives this channel never had. Acted, but real speech through
             our own Whisper - see data.load_harpervalley.
  FTC catch  real scam robocalls caught. IN-SOURCE (act_model trained on LLM
             labels over this corpus) - it is a floor check, not validation.
  27 calls   the honest external number, unchanged in provenance.
  authored   the 50 written transcripts, dev/test.

FTC is a seeded 500-call subsample so the four configurations cost the same;
+-2.4 points at n=500, which is smaller than every difference worth acting on.
"""
import hashlib
import json
import sys
from pathlib import Path

import data                       # before torch
import numpy as np
import act_model
import voice_model as vm
from sklearn.metrics import roc_auc_score

BAND = 0.35                       # §3 SUSPICIOUS
HARD = 0.70                       # §3 PHISHING - the band the 3B acts cannot reach:
#                                   threat 0.45 + prize 0.35 never co-fire, so the
#                                   shipped model's ceiling is SUSPICIOUS by arithmetic
ROOT = Path(__file__).resolve().parents[1]   # repo root: this file lives in scratch/
rng = np.random.default_rng(0)

hvb = data.load_harpervalley()
ftc = data.load_robocalls()
ftc = ftc.iloc[rng.permutation(len(ftc))[:500]]
auth = data.load_transcripts()
meta = {c["id"]: c for c in json.load(open(data.DATA / "vishing" / "calls" / "index.json"))}
cached = json.load(open(data.DATA / "vishing" / "calls" / "transcripts.json"))
cids = sorted(i for i in cached if len(cached[i]) > 200)
cy = np.array([meta[i]["label"] for i in cids])
dev = np.zeros(len(cids), bool)
r2 = np.random.default_rng(7)
for lab in (0, 1):                # the same seeded halves act_model.__main__ uses
    w = np.flatnonzero(cy == lab)
    dev[r2.permutation(w)[:len(w) // 2]] = True
print(f"HVB legit {len(hvb)}   FTC scam {len(ftc)}   real calls {len(cids)}   "
      f"authored {len(auth)}", flush=True)


def run(texts):
    v = [vm.score_transcript(t) for t in texts]
    return (np.array([x["peak_risk"] for x in v]),
            np.array([x["alert_at_s"] is not None for x in v]))


print(f"\n{'configuration':<26} {'HVB FP':>8} {'alerts':>7} | {'FTC':>6} | "
      f"{'27 dev':>7} {'test':>6} {'all':>6} | {'auth dev':>9} {'test':>6}")
# Default rows are the 2x2 above, collapsed to what shipped plus the checkpoint
# under test. Any checkpoint directory name given on the command line replaces
# them - `python -m scratch._hvb_eval act_distilbert_7b` scores just that one.
CONFIGS = [(c, c, False) for c in sys.argv[1:] if not c.startswith("--")] or [
    ("v1 acts (3B labels)", "act_distilbert_prev", False),
    ("shipped: 3B + ACL bg", "act_distilbert", False),
]
for name, out, stream in CONFIGS:
    act_model.OUT = ROOT / "models" / out
    act_model.load.cache_clear()
    w = next(iter(sorted((act_model.OUT).glob("model*"))), None)
    tag = hashlib.md5(w.read_bytes()).hexdigest()[:8] if w else "missing"
    vm.STREAM_ACTS = stream
    hp, ha = run(hvb.transcript)
    fp, _ = run(ftc.transcript)
    cp, _ = run([cached[i] for i in cids])
    ap, _ = run(auth.text)
    ay = auth.label.to_numpy()
    ad = (auth.split == "dev").to_numpy()
    print(f"{name+chr(32)+tag:<26} {(hp >= BAND).mean():>7.1%} {ha.mean():>7.1%} | "
          f"{(fp >= BAND).mean():>5.1%} {(fp >= HARD).mean():>6.1%} | "
          f"{roc_auc_score(cy[dev], cp[dev]):>7.3f} {roc_auc_score(cy[~dev], cp[~dev]):>6.3f} "
          f"{roc_auc_score(cy, cp):>6.3f} | "
          f"{roc_auc_score(ay[ad], ap[ad]):>9.3f} {roc_auc_score(ay[~ad], ap[~ad]):>6.3f}",
          flush=True)
