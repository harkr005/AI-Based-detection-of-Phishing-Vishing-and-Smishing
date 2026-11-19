"""Small distilled act model vs the 7B judge itself, through the shipped voice
pipeline (`vm.score_transcript`), on data no candidate trained on.

    HVB B        723 real legitimate bank calls - FALSE ALARMS. Half A is
                 reserved: the MONEY fix was diagnosed on it and the scambait
                 retrain may use it as negatives.
    scambait     live scam calls from the HELD-OUT channels only - CATCH RATE.
                 Fixed 2026-10-10 before any result was seen; never train on them.
    27 calls     recorded two-party calls, AUC.
    authored     the 50 written scripts, dev / test AUC.

Arguments are checkpoint names under models/, or `llm` (hybrid prompt) /
`llm2` (two-valued prompt). The llm rows need
    LLM_PRIMITIVES_MODEL=Qwen/Qwen2.5-7B-Instruct LLM_PRIMITIVES_4BIT=1
and cache every call's act risk in Datasets/vishing/llm_call_cache.json
(Datasets/ is gitignored), so a crash or a rerun costs nothing twice.

    python -m scratch._voice_llm_eval act_distilbert act_distilbert_union_bg llm
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import data                       # before torch
import numpy as np
import act_model
import llm_primitives as lp
import voice_model as vm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TEST_CHANNELS = act_model.SCAMBAIT_TEST_CHANNELS
BAND, HARD = 0.35, 0.70
CACHE = data.DATA / "vishing" / "llm_call_cache.json"

hvb = data.load_harpervalley()
hvb_b = [t for t, a in zip(hvb.transcript,
                           np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2)
         if not a]                # same seed-0 halves as _hvb_rules.py
sb = data.load_scambait()
sb_test = sb[sb.channel.isin(TEST_CHANNELS)].transcript.tolist()
meta = {c["id"]: c for c in json.load(open(data.DATA / "vishing" / "calls" / "index.json"))}
cached = json.load(open(data.DATA / "vishing" / "calls" / "transcripts.json"))
cids = sorted(i for i in cached if len(cached[i]) > 200)
cy = np.array([meta[i]["label"] for i in cids])
auth = data.load_transcripts()
ad, ay = (auth.split == "dev").to_numpy(), auth.label.to_numpy()
print(f"HVB B {len(hvb_b)}   scambait held-out {len(sb_test)} "
      f"({', '.join(sorted(TEST_CHANNELS))})   27 calls {len(cids)}   authored {len(auth)}",
      flush=True)

cache = json.load(open(CACHE)) if CACHE.exists() else {}
_llm = vm._llm_risk
calls = {"n": 0, "s": 0.0}


def cached_llm(text):
    k = f"{vm.LLM_HYBRID}:{hashlib.md5(text.encode()).hexdigest()}"
    if k not in cache:
        t0 = time.time()
        cache[k] = _llm(text)
        calls["n"] += 1
        calls["s"] += time.time() - t0
        if calls["n"] % 20 == 0:
            json.dump(cache, open(CACHE, "w"))
    return cache[k]


vm._llm_risk = cached_llm


def peaks(texts):
    v = [vm.score_transcript(t) for t in texts]
    return (np.array([x["peak_risk"] for x in v]),
            np.array([x["alert_at_s"] is not None for x in v]))


# Held-out-channel scam spans pasted into HVB B calls (act_model.injected): a
# live scammer's act inside real bank talk. CATCH RATE at >=.35; the plain HVB B
# column is the same calls without the span.
inj = act_model.injected(list(lp.ACTS), test=True)[0] if act_model.SCAMBAIT_SPANS.exists() else []
# Two wider catch-rate tests added 2026-10-24, nothing trained or judged on them:
# seven new scam-baiting channels, and KorCCVi's real Korean FSS calls (MT'd).
sb_new = sb[sb.channel.isin(act_model.SCAMBAIT_NEW_CHANNELS)].transcript.tolist()
kor = data.load_korccvi().transcript.tolist() if (
    data.DATA / "vishing" / "korccvi" / "translated.json").exists() else []
print(f"injected bank calls {len(inj)}   new-channel scambait {len(sb_new)}   Korean {len(kor)}")


def rate(texts):
    return f"{(peaks(texts)[0] >= BAND).mean():.1%}" if texts else "-"


print(f"\n{'config':<32} {'HVB B >=.35':>11} {'alerts':>7} | {'scam >=.35':>10} "
      f"{'>=.70':>6} | {'inj >=.35':>9} {'new ch':>6} {'Korean':>6} | {'27 AUC':>6} | "
      f"{'auth dev':>8} {'test':>6} | s/call")
for c in [a for a in sys.argv[1:] if not a.startswith("--")]:
    # "+c" = the same row with llm_primitives.COMBINE on (a lone dual-use act
    # counts half). Distilled rows only: the llm cache stores risk, not acts.
    c, lp.COMBINE = c.removesuffix("+c"), c.endswith("+c")
    assert not (lp.COMBINE and c.startswith("llm")), "+c needs per-act output; llm cache has risk"
    if c.startswith("llm"):
        vm.ACTS_FROM, vm.LLM_HYBRID = "llm", c == "llm"
        tag = "hybrid" if vm.LLM_HYBRID else "two-valued"
    else:
        vm.ACTS_FROM = "distilled"
        act_model.OUT = ROOT / "models" / c
        act_model.load.cache_clear()
        w = next(iter(sorted(act_model.OUT.glob("model*"))), None)
        tag = hashlib.md5(w.read_bytes()).hexdigest()[:8] if w else "MISSING"
    calls.update(n=0, s=0.0)
    hp, ha = peaks(hvb_b)
    sp, _ = peaks(sb_test)
    ip, _ = peaks(inj) if inj else (np.zeros(1), None)
    cp, _ = peaks([cached[i] for i in cids])
    ap, _ = peaks(auth.text)
    json.dump(cache, open(CACHE, "w"))
    spc = f"{calls['s'] / calls['n']:.1f}" if calls["n"] else "-"
    tag += "+c" if lp.COMBINE else ""
    print(f"{c + ' ' + tag:<32} {(hp >= BAND).mean():>10.1%} {ha.mean():>7.1%} | "
          f"{(sp >= BAND).mean():>10.1%} {(sp >= HARD).mean():>6.1%} | "
          f"{(ip >= BAND).mean() if inj else float('nan'):>9.1%} "
          f"{rate(sb_new):>6} {rate(kor):>6} | "
          f"{roc_auc_score(cy, cp):>6.3f} | "
          f"{roc_auc_score(ay[ad], ap[ad]):>8.3f} {roc_auc_score(ay[~ad], ap[~ad]):>6.3f} | {spc}",
          flush=True)
