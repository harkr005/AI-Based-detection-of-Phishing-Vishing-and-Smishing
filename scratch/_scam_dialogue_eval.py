"""Voice channel ACCURACY on a balanced English set where both classes share a
source - the one thing the 27 recorded calls give and the HVB-vs-scambait
numbers cannot.

Scam Dialogue (huggingface.co/datasets/BothBosu/scam-dialogue, Apache-2.0):
1,600 two-party phone dialogues written by Llama-3-70B-Instruct, 800 scam
(ssn / refund / support / reward) and 800 legit (delivery / insurance /
telemarketing / wrong number). Same generator for both halves, so the model
cannot win by spotting the corpus. Nothing here trained, tuned or judged on it:
all 1,600 rows are test.

CAVEAT that travels with every number: LLM-written, not recorded. Report it as
a synthetic benchmark. Text only, so this is the content layer at 100% weight -
the case of a pasted transcript with no audio and no caller number.

Same rows for the 27 recorded calls and the 50 authored scripts, so all three
sit at the deployed thresholds side by side. Rates and counts only.

    CUDA_VISIBLE_DEVICES="" python -m scratch._scam_dialogue_eval
"""
import json

import data                       # before torch
import numpy as np
import pandas as pd
import voice_model as vm
from sklearn.metrics import roc_auc_score

BAND, HARD = 0.35, 0.70
SD = data.DATA / "vishing" / "scam_dialogue" / "scam-dialogue_all.csv"


def row(name, y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = [f"{name:<26} n={len(y):>4} ({y.sum()} scam)  AUC {roc_auc_score(y, p):.3f}"]
    for t, tag in ((BAND, "SUSPICIOUS+"), (HARD, "PHISHING   ")):
        f = p >= t
        out.append(f"   flag at {tag} acc {(f == y).mean():6.1%}  caught {f[y == 1].mean():6.1%}  "
                   f"false alarms {f[y == 0].mean():6.1%}  precision {f[f].size and y[f].mean():.1%}")
    print("\n".join(out), flush=True)


def peaks(texts):
    return np.array([vm.score_transcript(t)["peak_risk"] for t in texts])


if __name__ == "__main__":
    sd = pd.read_csv(SD).drop_duplicates("dialogue")
    sd["p"] = peaks(sd.dialogue)
    row("Scam Dialogue (synthetic)", sd.label, sd.p)
    print("   flag rate at SUSPICIOUS+ by type: " + ", ".join(
        f"{t} {g.p.ge(BAND).mean():.0%}" for (_, t), g in sd.groupby(["label", "type"])))

    meta = {c["id"]: c for c in json.load(open(data.DATA / "vishing" / "calls" / "index.json"))}
    cached = json.load(open(data.DATA / "vishing" / "calls" / "transcripts.json"))
    cids = sorted(i for i in cached if len(cached[i]) > 200)     # = _voice_llm_eval
    row("27 recorded calls (real)", [meta[i]["label"] for i in cids], peaks([cached[i] for i in cids]))

    auth = data.load_transcripts()
    row("50 authored scripts", auth.label, peaks(auth.text))
