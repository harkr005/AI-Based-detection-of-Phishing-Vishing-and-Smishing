"""ACL-2026 evidence LOCATOR over the live scambait calls -> scambait/llm_spans.json.

`_bg_spans.py` did this for the FTC robocalls, which only carry threat, prize and
urgency. Live calls are where credential, payment and remote actually happen,
and those are the acts a real bank call also talks about - so they are the
spans worth pasting into HarperValley calls (`_inject_eval.py`, act_model
--inject). Only chunks the 7B judge already labelled are asked, and only for
the acts it said fired there. A paraphrased quote is dropped, as in _bg_spans.

Covers BOTH train and held-out channels: held-out spans are the injection TEST,
train-channel spans the injection training rows. Resumable: chunks already in
the output are skipped.

    LLM_PRIMITIVES_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_PRIMITIVES_4BIT=1 \
        python -m scratch._scambait_spans
"""
import json
import time

import data
import llm_primitives as lp

SB = data.DATA / "vishing" / "scambait"
OUT = SB / "llm_spans.json"
j = json.load(open(SB / "llm_chunk_acts.json"))
keys = j["keys"]
d = data.load_scambait().set_index("id")
out = json.load(open(OUT)) if OUT.exists() else {}

rows = []                                   # (call id, chunk index, chunk, acts fired)
for cid, labs in j["calls"].items():
    segs = lp.chunks(d.loc[cid].transcript)
    assert len(segs) == len(labs), f"{cid}: transcript changed since it was judged"
    for k, (seg, lab) in enumerate(zip(segs, labs)):
        fired = [a for a, v in zip(keys, lab) if v]
        if fired and str(k) not in out.get(cid, {}):
            rows.append((cid, k, seg, fired))
print(f"{len(rows)} labelled chunks left to locate", flush=True)

t0, kept, para = time.time(), 0, 0
for b in range(0, len(rows), 80):
    batch = rows[b:b + 80]
    q = lp.detect([r[2] for r in batch], keys=keys, criteria=True, spans=True,
                  max_new=256, bs=8)
    for (cid, k, seg, fired), row in zip(batch, q):
        got = {}
        for a, quote in zip(keys, row):
            if a not in fired or not quote:
                continue
            if quote.lower() in seg.lower():
                got.setdefault(a, []).append(quote)
                kept += 1
            else:
                para += 1
        out.setdefault(cid, {})[str(k)] = got   # {} = asked, nothing verbatim
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"  {b + len(batch)}/{len(rows)}  {kept} kept  {para} paraphrased  "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)

n = {a: sum(len(v.get(a, [])) for c in out.values() for v in c.values()) for a in keys}
print("spans per act:", n)
