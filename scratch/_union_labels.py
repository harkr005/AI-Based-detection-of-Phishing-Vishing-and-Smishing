"""3B labels OR 7B labels, per act -> `llm_acts_union.json`.

The two judges disagree in one specific, measurable way. The 7B + criteria
prompt is stricter everywhere: it finds urgency (194) that the 3B never asked
about, and it collapses prize from 45 calls to 2. On the FTC corpus, where
"you have an unclaimed refund" is one of the commonest openers, 2 is the wrong
number - the stricter prompt reads a recorded prize claim as a topic, not an act.

Dropping prize costs real catches, because prize (0.35) is exactly SUSPICIOUS
while urgency (0.30) alone is below the band and only pays off when it lands
with threat. So the union keeps both: an act fires if EITHER judge saw it.

This is a recall-first labelling rule and it must be declared as one - two
teachers over the same corpus, never a single-model ablation.
"""
import json

import data
import numpy as np

ROOT = data.DATA / "vishing" / "robocall"
a = json.load(open(ROOT / "llm_acts_3b.json"))
b = json.load(open(ROOT / "llm_acts_7b.json"))
assert a["id"] == b["id"], "different corpora - the union would be nonsense"

keys = list(dict.fromkeys(a["keys"] + b["keys"]))
A, B = np.array(a["labels"], bool), np.array(b["labels"], bool)
out = np.zeros((len(a["id"]), len(keys)), bool)
for src, K in ((A, a["keys"]), (B, b["keys"])):
    for j, k in enumerate(K):
        out[:, keys.index(k)] |= src[:, j]

json.dump({"id": a["id"], "keys": keys, "labels": out.astype(int).tolist()},
          open(ROOT / "llm_acts_union.json", "w"))
for j, k in enumerate(keys):
    print(f"  {k:<11} {out[:, j].sum():>5}  ({out[:, j].mean():.1%})")
print(f"calls with no act: {(~out.any(1)).mean():.1%}")
