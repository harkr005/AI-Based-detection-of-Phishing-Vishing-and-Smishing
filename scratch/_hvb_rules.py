"""Which keyword rule causes the 12.0% HarperValley false alarms (step 3 of the
voice plan, 2026-10-10).

Rules only - the act model is not loaded - because `_hvb_eval.py` showed the
false-alarm rate is identical with and without it, so every HVB false alarm is
a rule. HVB is split into seeded halves: DIAGNOSE on half A (phrases printed -
HVB is a public acted corpus), and any fix is judged on half B only.

For each rule: share of HVB false alarms it takes part in, how often it fires on
legitimate calls, and how often on scam calls (scambait, 27 real calls' scam
half). A rule that fires as much on scams as on bank calls is noise; one that
fires mostly on scams is worth keeping even if it costs some false alarms.

    python -m scratch._hvb_rules [--phrases]
"""
import json
import re
import sys
from collections import Counter

import data                       # before torch
import numpy as np
import voice_model as vm

BAND = 0.35


def per_call(text):
    """-> (rules-only peak risk, set of rule names firing in any window)."""
    texts = [t for _, t in vm.windows(vm.as_words(text))]
    if not texts:
        return 0.0, set()
    peak = float(vm.risk_curve(vm.window_risk(texts)).max())
    fired = {n for n, (_, rx) in zip(vm._ACT_NAMES, vm.RULES)
             if any(rx.search(t) for t in texts)}
    return peak, fired


hvb = data.load_harpervalley()
half = np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2
sb = data.load_scambait()
meta = {c["id"]: c for c in json.load(open(data.DATA / "vishing" / "calls" / "index.json"))}
cached = json.load(open(data.DATA / "vishing" / "calls" / "transcripts.json"))
real = [(cached[i], meta[i]["label"]) for i in cached if len(cached[i]) > 200]

sets = {"HVB A (legit)": [t for t, h in zip(hvb.transcript, half) if h],
        "HVB B (legit)": [t for t, h in zip(hvb.transcript, half) if not h],
        "scambait (scam)": list(sb.transcript),
        "27 calls scam": [t for t, y in real if y == 1],
        "27 calls legit": [t for t, y in real if y == 0]}
scored = {k: [per_call(t) for t in v] for k, v in sets.items()}

print(f"{'':<16}" + "".join(f"{k:>17}" for k in sets))
print(f"{'n':<16}" + "".join(f"{len(v):>17}" for v in sets.values()))
print(f"{'flagged >=.35':<16}" + "".join(
    f"{np.mean([p >= BAND for p, _ in s]):>16.1%} " for s in scored.values()))
print("\ncalls where the rule fires at least once:")
for n in vm._ACT_NAMES:
    print(f"  {n:<14}" + "".join(
        f"{np.mean([n in f for _, f in s]):>16.1%} " for s in scored.values()))

fa = [f for p, f in scored["HVB A (legit)"] if p >= BAND]
print(f"\nHVB A false alarms: {len(fa)}. Rule combinations behind them:")
for combo, c in Counter(tuple(sorted(f)) for f in fa).most_common(8):
    print(f"  {c:>4}  {' + '.join(combo) or '(none)'}")

if "--phrases" in sys.argv:
    # What the rule actually matched, on HVB half A only.
    for n in vm._ACT_NAMES:
        rx = vm.RULES[vm._ACT_NAMES.index(n)][1]
        hits = Counter(re.sub(r"\s+", " ", m.group(0).lower())
                       for t in sets["HVB A (legit)"] for m in rx.finditer(t))
        if hits:
            print(f"\n{n}: " + ", ".join(f"{p!r} {c}" for p, c in hits.most_common(12)))

# RESULT 2026-10-10: MONEY was 94 of 96 half-A false alarms, all on "transfer"
# (own-account moves). Dropping that verb: HVB B 10.8% -> 0.3%, scambait catch
# 75.1% -> 74.7%. Numbers and the rejected alternative are at voice_model.MONEY.
