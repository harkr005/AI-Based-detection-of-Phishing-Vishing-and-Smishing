"""Does letting the judge say "unclear" cut false alarms on real legitimate calls?

CHI 2025 (arXiv 2502.03964) ran an LLM turn-by-turn on live calls and found the
alerts that annoyed users came from three things legitimate callers genuinely do:
identity verification, financial-information sharing, manufactured urgency. This
project measured the same failure independently (a real bank call at 0.994). Their
fix is a third answer, so the judge can decline instead of guessing.

Both arms use the SAME model, quantisation and criteria text; only the answer set
differs (true/false vs yes/no/unclear), so the comparison is the third option and
nothing else. One 250-word chunk per call, which is what call_risk scores first.
"""
import json
import os
import sys
from pathlib import Path

import data                       # before torch
import numpy as np

os.environ.setdefault("LLM_PRIMITIVES_4BIT", "1")   # coexist with a running Whisper
import llm_primitives as lp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
KEYS = lp.ORDER
BAND = 0.35
rng = np.random.default_rng(0)

hvb = data.load_harpervalley()
hvb = hvb.iloc[rng.permutation(len(hvb))[:N]]
ftc = data.load_robocalls()
ftc = ftc.iloc[rng.permutation(len(ftc))[:N]]
segs = ([lp.chunks(t)[0] for t in hvb.transcript if lp.chunks(t)]
        + [lp.chunks(t)[0] for t in ftc.transcript if lp.chunks(t)])
y = np.array([0] * sum(1 for t in hvb.transcript if lp.chunks(t))
             + [1] * sum(1 for t in ftc.transcript if lp.chunks(t)))
print(f"{(y == 0).sum()} legit HVB + {(y == 1).sum()} scam FTC, {lp.MODEL} "
      f"{'4-bit' if lp.FOURBIT else 'fp16'}", flush=True)

W = np.array([lp.ACTS[k][1] for k in KEYS])
rows = []
for tag, kw in [("two-valued (shipped)", dict(criteria=True)),
                ("three-valued (+unclear)", dict(uncertain=True))]:
    out = lp.detect(segs, keys=KEYS, max_new=96, bs=8, **kw)
    fired = (out == "yes") if kw.get("uncertain") else out.astype(bool)
    unclear = float((out == "unclear").mean()) if kw.get("uncertain") else 0.0
    risk = np.minimum(fired @ W, 1.0)
    rows.append((tag, fired, risk, unclear))
    print(f"\n{tag}")
    print(f"  {'act':<11} {'scam':>7} {'legit':>7}")
    for j, k in enumerate(KEYS):
        print(f"  {k:<11} {fired[y == 1, j].mean():>6.1%} {fired[y == 0, j].mean():>7.1%}")
    print(f"  risk >= {BAND}:  scam {100 * (risk[y == 1] >= BAND).mean():.1f}% caught, "
          f"legit {100 * (risk[y == 0] >= BAND).mean():.1f}% false alarms"
          + (f"   ({unclear:.1%} of answers were 'unclear')" if unclear else ""))

# HYBRID. The two arms disagree in opposite directions per act, and the split is
# exactly the one this project already found by hand: threat / secrecy / prize are
# performed by NO legitimate caller, so making the judge cautious about them only
# throws catch away; credential / payment / remote / urgency are things real banks,
# collectors and sales agents do, and those are where declining pays. Note payment
# in the two-valued arm - it fires MORE on legitimate calls than on scams.
CLEAN = ["threat", "secrecy", "prize"]
two, three = rows[0][1], rows[1][1]
hyb = np.where(np.array([k in CLEAN for k in KEYS]), two, three)
risk = np.minimum(hyb @ W, 1.0)
print("\nhybrid (two-valued for " + "/".join(CLEAN) + ", three-valued for the rest)")
print(f"  {'act':<11} {'scam':>7} {'legit':>7}")
for j, k in enumerate(KEYS):
    print(f"  {k:<11} {hyb[y == 1, j].mean():>6.1%} {hyb[y == 0, j].mean():>7.1%}")
print(f"  risk >= {BAND}:  scam {100 * (risk[y == 1] >= BAND).mean():.1f}% caught, "
      f"legit {100 * (risk[y == 0] >= BAND).mean():.1f}% false alarms")
