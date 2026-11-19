"""KorCCVi vishing calls, Korean -> English, as a third held-out catch-rate test.

706 REAL voice-phishing calls the Korean Financial Supervisory Service published
from victims' recordings (github.com/selfcontrol7/Korean_Voice_Phishing_Detection,
DATA.md). Only the vishing half is used: every legit call comes from a different
source (AI Hub), so the corpus's own authors warn that source == label - its
legit half cannot give an honest false-alarm number. So, like scambait and FTC,
this yields a CATCH RATE and never an accuracy.

Terms: the transcripts are FSS-derived and not ours to redistribute; they stay
under Datasets/ (gitignored). Never print transcript text - rates only.

Translation is Helsinki-NLP/opus-mt-ko-en (Marian, Apache-2.0) per CLOVA segment,
on CPU so the GPU stays free. CAVEAT that travels with the number: machine
translation, not our Whisper - a missed act may be a translation loss, and the
Korean scam scripts (prosecutor impersonation, "safe account" transfers, loan
fraud) are not the US scripts the act model learned.

    python -m scratch._korccvi --translate     # resumable, ~30-60 min CPU
"""
import glob
import json
import sys
import time
from pathlib import Path

import data

SRC = data.DATA / "vishing" / "korccvi_repo" / "Multimodal" / "data" / "transcripts" / "vishing"
OUT = data.DATA / "vishing" / "korccvi" / "translated.json"


def save(done):
    """Temp file then os.replace: a hard power-off mid-write must not truncate
    the hours of translation already done (this laptop's GPU has crashed it)."""
    import os
    tmp = OUT.with_suffix(".tmp")
    json.dump(done, open(tmp, "w"))
    os.replace(tmp, OUT)


def translate(bs=32):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(4)                 # leave cores for the GPU job's data loading
    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    mdl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.load(open(OUT)) if OUT.exists() else {}
    files = sorted(glob.glob(str(SRC / "*.json")))
    t0 = time.time()
    for k, f in enumerate(files, 1):
        cid = Path(f).stem
        if cid in done:
            continue
        segs = [s["text"].strip() for s in
                sorted(json.load(open(f, encoding="utf-8"))["segments"], key=lambda s: s["start"])]
        segs = [s for s in segs if s]
        en = []
        for i in range(0, len(segs), bs):
            enc = tok(segs[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            with torch.no_grad():
                gen = mdl.generate(**enc, max_new_tokens=256, num_beams=1)
            en += tok.batch_decode(gen, skip_special_tokens=True)
        done[cid] = " ".join(en)
        if k % 20 == 0 or k == len(files):
            save(done)
            el = (time.time() - t0) / 60
            print(f"  {k}/{len(files)}  {el:.1f} min", flush=True)
    save(done)
    n = sorted(len(t.split()) for t in done.values())
    print(f"{len(done)} calls translated, median {n[len(n) // 2]} English words")


if __name__ == "__main__":
    if "--translate" in sys.argv:
        translate()
    else:
        print(__doc__)
