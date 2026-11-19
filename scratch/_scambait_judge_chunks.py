"""7B judge over every transcribed scambait call, labels kept PER CHUNK.

`_scambait_eval.py --judge` ORs the chunks into one row per call, which is right
for a catch rate and wrong for training: these calls run ~2,600 words and
act_model reads 256 tokens, so a call-level "payment" label would be pinned on
~10 chunks that mostly do not contain it. Per-chunk labels also give the ACL
background negatives for free - the same call's chunks without the act - so no
span locator pass is needed on this corpus.

Resumable: rerun after more calls are transcribed and it judges only the new
ones. Chunks are `lp.chunks(transcript)`, recomputed deterministically by the
reader, so only the labels are stored.

    LLM_PRIMITIVES_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_PRIMITIVES_4BIT=1 \\
        python -m scratch._scambait_judge_chunks
"""
import json
import time

import data                       # before torch
import llm_primitives as lp

OUT = data.DATA / "vishing" / "scambait" / "llm_chunk_acts.json"
SAVE_EVERY = 5

assert "7B" in lp.MODEL and lp.FOURBIT, "use the same 7B 4-bit judge as the FTC labels"
d = data.load_scambait()
done = json.load(open(OUT)) if OUT.exists() else {"keys": lp.ORDER, "calls": {}}
assert done["keys"] == lp.ORDER
todo = [r for r in d.itertuples() if r.id not in done["calls"]]
print(f"{len(d)} transcribed, {len(done['calls'])} judged, judging {len(todo)}", flush=True)

t0 = time.time()
for k, r in enumerate(todo, 1):
    out = lp.detect(lp.chunks(r.transcript), keys=lp.ORDER, criteria=True,
                    max_new=72, bs=8)
    done["calls"][r.id] = out.astype(int).tolist()
    if k % SAVE_EVERY == 0 or k == len(todo):
        json.dump(done, open(OUT, "w"))
        m = (time.time() - t0) / 60
        print(f"  {k}/{len(todo)}  {m:.1f} min  eta {m / k * (len(todo) - k):.0f} min",
              flush=True)
