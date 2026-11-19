"""ACL-2026 evidence LOCATOR over the FTC scam calls -> `llm_spans.json`.

Only instance type 1 of the paper was ever built here (whole transcript -> acts,
`act_model.py`). This produces what types 2 and 3 need: the caller's verbatim
words for each act, so the evidence can be CUT OUT of the transcript. What is
left is a background negative from the SAME corpus, same recording conditions,
same Whisper pass - the one construction that cannot be shortcut by register,
which is what broke `sms_model`, `speech_model` and phrase mining on this channel.

Runs only on the 230 calls the judge already labelled positive: a call with no
act has no evidence to cut, so it is already a background example.

A quote the model paraphrased instead of copying is DROPPED, not fuzzy-matched.
An unfindable span cannot be removed, so keeping it would silently leave the
evidence inside the "background" negative - the exact leak this is meant to close.
"""
import json
import sys
import time

import data
import numpy as np
import llm_primitives as lp

# The acts act_model actually distils - which teacher labelled them decides
# which two survive MIN_POS. `--7b` follows act_model.py --7b.
tag = next((a[2:] for a in sys.argv if a in ("--7b", "--union")), "")
KEYS = {"": ["threat", "prize"],
        "7b": ["threat", "urgency"],
        "union": ["threat", "prize", "urgency"]}[tag]
LAB = data.DATA / "vishing" / "robocall" / f"llm_acts{'_' + tag if tag else ''}.json"
OUT = data.DATA / "vishing" / "robocall" / f"llm_spans{'_' + tag if tag else ''}.json"
print(f"{tag or '3b'} labels, acts {KEYS}", flush=True)

j = json.load(open(LAB))
lab = np.array(j["labels"])
cols = [j["keys"].index(k) for k in KEYS]
ids = [i for i, p in zip(j["id"], lab[:, cols].any(1)) if p]
d = data.load_robocalls().set_index("id")
rows = [(cid, c) for cid in ids for c in lp.chunks(d.loc[cid].transcript)]
print(f"{len(ids)} calls with an act, {len(rows)} chunks", flush=True)

t0 = time.time()
q = lp.detect([c for _, c in rows], keys=KEYS, criteria=True, spans=True,
              max_new=160, bs=8)

out, kept, para = {}, 0, 0
for (cid, chunk), row in zip(rows, q):
    for k, quote in zip(KEYS, row):
        if not quote:
            continue
        if quote.lower() in chunk.lower():
            out.setdefault(cid, {}).setdefault(k, []).append(quote)
            kept += 1
        else:
            para += 1
json.dump(out, open(OUT, "w"), indent=1)
print(f"{kept} verbatim spans over {len(out)} calls, {para} paraphrased and dropped "
      f"({(time.time() - t0) / 60:.1f} min)")
for k in KEYS:
    n = sum(len(v.get(k, [])) for v in out.values())
    print(f"  {k:<8} {n:>4} spans")
