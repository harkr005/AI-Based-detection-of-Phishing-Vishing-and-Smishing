"""What the voice channel does on LIVE two-party scam calls, and what acts are
actually in them.

Two modes, both read `data.load_scambait()`:

  (default)  Catch rate through the shipped pipeline (`vm.score_transcript`),
             per checkpoint named on the command line, at both §3 bands. Every
             row is label 1, so this is a CATCH RATE and never an accuracy - the
             false-alarm half stays HarperValley (`_hvb_eval.py`).

  --judge N  Run the LLM judge over N calls (all seven acts, criteria prompt)
             and print how often each fires. This is the number the whole
             exercise is for: on FTC robocalls credential/payment/remote fire on
             17 calls out of 1,353, because the recording stops before the ask.
             If they fire here, this corpus is the training data act_model has
             never had, and the labels are written out ready for it.

Run the judge with the same 7B + 4-bit teacher `_label7b.py` used, or the
comparison to the FTC label file is not like-for-like:
    LLM_PRIMITIVES_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_PRIMITIVES_4BIT=1
"""
import json
import sys
import time
from pathlib import Path

import data                       # before torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAND, HARD = 0.35, 0.70

d = data.load_scambait()
print(f"{len(d)} scambait calls, {d.dur.sum()/3600:.1f} h, "
      f"median {int(np.median([len(t.split()) for t in d.transcript]))} words")
print(d.channel.value_counts().to_string())

if "--judge" in sys.argv:
    import llm_primitives as lp
    n = int(sys.argv[sys.argv.index("--judge") + 1])
    sub = d.sample(min(n, len(d)), random_state=0)   # across channels, not the first channel
    rows = [(r.id, c) for r in sub.itertuples() for c in lp.chunks(r.transcript)]
    print(f"\njudging {len(sub)} calls, {len(rows)} chunks with {lp.MODEL}"
          f"{' 4bit' if lp.FOURBIT else ''}", flush=True)
    t0 = time.time()
    out = lp.detect([c for _, c in rows], keys=lp.ORDER, criteria=True,
                    max_new=72, bs=8)
    per_call = {}
    for (cid, _), row in zip(rows, out):        # an act anywhere in the call
        per_call[cid] = np.logical_or(per_call.get(cid, np.zeros(len(lp.ORDER), bool)), row)
    Y = np.array([per_call[i] for i in sub.id])
    print(f"labelled in {(time.time()-t0)/60:.1f} min\n")
    print(f"{'act':<12} {'scambait':>9} {'FTC robocalls':>14}")
    ftc = json.load(open(data.DATA / "vishing" / "robocall" / "llm_acts_7b.json"))
    F = np.array(ftc["labels"], bool)
    for j, a in enumerate(lp.ORDER):
        f = F[:, ftc["keys"].index(a)].mean() if a in ftc["keys"] else float("nan")
        print(f"{a:<12} {Y[:, j].mean():>8.1%} {f:>13.1%}")
    print(f"{'ANY':<12} {Y.any(1).mean():>8.1%} {F.any(1).mean():>13.1%}")
    p = data.DATA / "vishing" / "scambait" / "llm_acts.json"
    json.dump({"id": sub.id.tolist(), "keys": lp.ORDER,
               "labels": Y.astype(int).tolist()}, open(p, "w"))
    print(f"\n-> {p.name}   (train act_model on these once the corpus is complete)")
    sys.exit()

import act_model
import voice_model as vm
import hashlib

ckpts = [a for a in sys.argv[1:] if not a.startswith("--")] or ["act_distilbert"]

# LENGTH CONTROL. These calls run a median 2,782 words against FTC's 53, and
# peak_risk is a MAXIMUM over windows - the same inflation that sank streaming
# acts (2026-08-24). `--prefix N` truncates every transcript to N words so the
# catch rate is read at a comparable length instead of over ~500 windows.
prefix = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 0
texts = [" ".join(t.split()[:prefix]) if prefix else t for t in d.transcript]
if prefix:
    print(f"LENGTH-CONTROLLED: first {prefix} words only")
print(f"\n{'checkpoint':<30} {'>=.35':>7} {'>=.70':>7} {'alerts':>7}")
for c in ckpts:
    act_model.OUT = ROOT / "models" / c
    act_model.load.cache_clear()
    w = next(iter(sorted(act_model.OUT.glob("model*"))), None)
    tag = hashlib.md5(w.read_bytes()).hexdigest()[:8] if w else "MISSING"
    v = [vm.score_transcript(t) for t in texts]
    peak = np.array([x["peak_risk"] for x in v])
    alert = np.array([x["alert_at_s"] is not None for x in v])
    line = (f"{c + ' ' + tag:<30} {(peak >= BAND).mean():>6.1%} "
            f"{(peak >= HARD).mean():>6.1%} {alert.mean():>6.1%}")
    if "--windows" in sys.argv:
        # peak_risk is a MAXIMUM, so a 17-minute call gets ~500 chances to
        # exceed the band and a 30-second robocall gets 7. The per-window rate
        # is the only length-normalised figure available, and it is only
        # readable next to the same number on the legitimate corpus.
        def wrate(scored):
            cs = [c for x in scored for c in x["curve"]]
            return sum(y >= BAND for y in cs) / max(len(cs), 1)
        hvb = [vm.score_transcript(t) for t in data.load_harpervalley().transcript]
        line += f" | windows {wrate(v):>6.1%} scam  {wrate(hvb):>6.1%} HVB legit"
    print(line, flush=True)
