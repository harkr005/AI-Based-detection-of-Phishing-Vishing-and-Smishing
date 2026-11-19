"""HarperValleyBank -> transcripts, through OUR Whisper. The missing negatives.

1,446 real recorded human-human banking calls (23 h, 8 kHz telephony, public
domain, github.com/cricketclub/gridspace-stanford-harper-valley). The corpus
ships human transcripts and we deliberately DO NOT USE THEM: every failure on
this channel came from positives being ASR output and negatives being written
text, so the negatives have to go through the same faster-whisper pass as the
scam calls or the model can just detect transcription provenance again.

The two speaker channels are mixed to one mono track, matching the 27 YouTube
calls' format. Standing caveat that must travel with any number: these are crowd
workers performing banking tasks, so they are ACTED, the same weakness as our 15
legitimate calls - but there are 96x more of them and the topics (balance, card
replacement, transfers) are exactly the hard negatives the content layer fails on.

Resumable: writes after every call, so a killed session loses one file.
"""
import json
import sys
import time

import data                       # before torch
import numpy as np
import soundfile as sf
import voice_model as vm

ROOT = data.DATA / "vishing" / "harpervalley"
OUT = ROOT / "transcripts.json"
SR = 16000
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6


def mixed(sid):
    """Both speakers, one mono track at 16 kHz."""
    parts = []
    for role in ("agent", "caller"):
        x, sr = sf.read(ROOT / "data" / "audio" / role / f"{sid}.wav", dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        parts.append(x)
    n = max(len(p) for p in parts)
    return sum(np.pad(p, (0, n - len(p))) for p in parts) / len(parts)


sids = sorted(p.stem for p in (ROOT / "data" / "audio" / "agent").glob("*.wav"))
done = json.load(open(OUT)) if OUT.exists() else {}
todo = [s for s in sids if s not in done][:LIMIT]
print(f"{len(sids)} calls, {len(done)} already done, transcribing {len(todo)}", flush=True)

t0 = time.time()
for k, sid in enumerate(todo, 1):
    words = vm.transcribe(mixed(sid))
    done[sid] = " ".join(w for _, _, w in words)
    json.dump(done, open(OUT, "w"))
    if k % 25 == 0 or k == len(todo):
        el = (time.time() - t0) / 60
        print(f"  {k}/{len(todo)}  {el:.1f} min  eta {el / k * (len(todo) - k):.0f} min",
              flush=True)

n = [len(t.split()) for t in done.values()]
print(f"{len(done)} transcripts, median {int(np.median(n))} words, "
      f"{sum(1 for x in n if x < 20)} under 20 words")
