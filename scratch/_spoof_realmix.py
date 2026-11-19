"""Real voices from MANY recording sources: does that stop the spoof head learning
"which recordings are real"?

_spoof_mlaad.py (2026-10-27) trained on 143 modern TTS generators and caught ~90%
of unseen ones, but flagged real people from any recording source it had not
trained on (--conv: unseen scambait channels 0 -> 54%, ITW real 4 -> 19%). Its
real class came from 2-4 sources. This run gives it five more, all free:

  TRAIN real   Common Voice 21 en (volunteers' own phones/laptops, CC0), People's
               Speech (internet audio, CC-BY), VoxPopuli en (parliament, CC0),
               AMI headset + distant mic (meetings, CC-BY), Svarah (Indian English,
               117 speakers, CC-BY-4.0; gated - skipped if not accessible)
               + phone twins of half, + everything the plain MLAAD run used
               (ASV19 + ASV5 rows, MLAAD train families, LibriSpeech dev).
               NOT HVB, NOT scambait - those stay test-only this time.
  TEST real    LEAVE-SOURCE-OUT, never trained on: Earnings-22 (earnings phone calls,
               global accents, CC-BY-SA), FLEURS en_us (Google, CC-BY), ITW real,
               HVB (all), scambait (all channels), LibriSpeech test.
  TEST fake    as _spoof_mlaad: 41 held-out MLAAD generators + Edge, our 304 calls.

RULE, fixed 2026-10-27 before any score was seen: ADOPT only if unseen generators
(phone) and our Edge calls (phone) each gain >= 10 pts caught at 0.5, EVERY real
test source rises <= 2 pts flagged at 0.5, and ASV21 pstn / ASV5 eval / ITW phone
EER rise <= 1 pt. Writes Datasets/vishing/mlaad/head_realmix.pkl, never models/.
Rates and counts only.

    python -m scratch._spoof_realmix --download   # ~3.6 GB, resumable
    python -m scratch._spoof_realmix --extract    # decode to 16 kHz wav clips
    python -m scratch._spoof_realmix              # embed + train + test
"""
import io
import json
import os
import sys
import tarfile

import data   # before torch (Windows libiomp)
import numpy as np
import soundfile as sf

import spoof_model as sm
from scratch import _spoof_mlaad as mm

ROOT = data.DATA / "vishing" / "realmix"
RAW = ROOT / "raw"
CLIP = 8.0          # seconds kept per clip (the head reads the first 4)
N_TRAIN = 2000      # clips per training source
N_TEST = 600
FILES = {   # source -> (repo, path in repo)
    "cv":      ("fsicoli/common_voice_21_0", "audio/en/test/en_test_0.tar"),
    "ps":      ("MLCommons/peoples_speech", "test/test-00000-of-00008.parquet"),
    "vp":      ("facebook/voxpopuli", "en/test-00000-of-00001.parquet"),
    "ami_ihm": ("edinburghcstr/ami", "ihm/test-00000-of-00004.parquet"),
    "ami_sdm": ("edinburghcstr/ami", "sdm/test-00000-of-00004.parquet"),
    "svarah":  ("ai4bharat/Svarah", "data/test-00000-of-00003.parquet"),
    "earn":    ("distil-whisper/earnings22", "full/test-00000-of-00004-4541e1972fe3ed48.parquet"),
    "fleurs":  ("google/fleurs", "data/en_us/audio/test.tar.gz"),
}
TRAIN_SRC = ["cv", "ps", "vp", "ami_ihm", "ami_sdm", "svarah"]
TEST_SRC = ["earn", "fleurs"]


def download():
    import requests
    from huggingface_hub import get_token
    RAW.mkdir(parents=True, exist_ok=True)
    auth = {"Authorization": f"Bearer {get_token()}"}
    for src, (repo, path) in FILES.items():
        out = RAW / f"{src}{os.path.splitext(path)[1] if not path.endswith('.tar.gz') else '.tar.gz'}"
        if out.exists():
            continue
        r = requests.get(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}",
                         headers=auth, stream=True, timeout=60)
        if r.status_code in (401, 403):
            print(f"  {src}: no access ({r.status_code}) - accept the terms on huggingface.co/datasets/{repo}", flush=True)
            continue
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
        os.replace(tmp, out)
        print(f"  {src}: {out.stat().st_size / 1e6:.0f} MB", flush=True)
    print("download done", flush=True)


def _write(x, sr, out):
    if x.ndim > 1:
        x = x.mean(1)
    if sr != sm.SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), sm.SR)
        x = resample_poly(x, sm.SR // g, int(sr) // g)
    sf.write(out, x[:int(CLIP * sm.SR)].astype(np.float32), sm.SR)


def _decode(b):
    try:
        return sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        return None


def extract():
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    for src in FILES:
        raw = next(RAW.glob(f"{src}.*"), None)
        out = ROOT / src
        if raw is None or raw.suffix == ".part" or (out.exists() and any(out.iterdir())):
            continue
        out.mkdir(parents=True, exist_ok=True)
        n = N_TEST if src in TEST_SRC else N_TRAIN
        k = 0
        if raw.name.endswith((".tar", ".tar.gz")):
            with tarfile.open(raw) as t:
                mem = [m for m in t.getmembers() if m.isfile() and m.name.endswith((".mp3", ".wav", ".flac"))]
                for m in [mem[i] for i in sorted(rng.permutation(len(mem))[:n * 2])]:
                    a = _decode(t.extractfile(m).read())
                    if a is not None and len(a[0]) >= 2 * a[1]:      # >= 2 s of audio
                        _write(*a, out / f"{k}.wav")
                        k += 1
                    if k >= n:
                        break
        else:
            col = "audio_filepath" if src == "svarah" else "audio"   # Svarah names its audio column so
            tab = pq.read_table(raw, columns=[col]).column(col).to_pylist()
            if src == "earn":   # whole earnings calls: random 8 s slices across each call
                per = -(-n // len(tab))
                for row in tab:
                    a = _decode(row["bytes"])
                    if a is None:
                        continue
                    x, sr = a
                    for s in sorted(rng.integers(0, max(1, len(x) - int(CLIP * sr)), per)):
                        _write(x[s:s + int(CLIP * sr)], sr, out / f"{k}.wav")
                        k += 1
            else:
                for i in sorted(rng.permutation(len(tab))[:n * 2]):
                    a = _decode(tab[i]["bytes"])
                    if a is not None and len(a[0]) >= 2 * a[1]:
                        _write(*a, out / f"{k}.wav")
                        k += 1
                    if k >= n:
                        break
        print(f"  {src}: {k} clips", flush=True)
    print("extract done", flush=True)


def clips(src):
    return sorted((str(p) for p in (ROOT / src).glob("*.wav")), key=lambda p: int(os.path.basename(p)[:-4]))


def main():
    import joblib
    from scratch._spoof_wholecall import calib_sets, test_sets
    E = lambda name, paths, phone=False: mm.cached(f"realmix_{name}", paths, phone=phone)

    # everything the plain MLAAD run trained on (all cached)
    files = [f for f in json.load(open(mm.ROOT / "picked.json")) if (mm.ROOT / f).exists()]
    gen_of = lambda f: f.split("/")[2]
    tr_g, te_g = mm.split(sorted({gen_of(f) for f in files}))
    tr_f = [str(mm.ROOT / f) for f in files if gen_of(f) in tr_g]
    te_f = [str(mm.ROOT / f) for f in files if gen_of(f) in te_g and mm.family(gen_of(f)) != "edge"]
    lib_tr = sorted(str(p) for p in (mm.LIBRI / "dev-clean").rglob("*.flac"))
    lib_te = sorted(str(p) for p in (mm.LIBRI / "test-clean").rglob("*.flac"))
    M, Mp = mm.cached("train", tr_f), mm.cached("train_phonehalf", mm.half(tr_f), phone=True)
    Lr, Lrp = mm.cached("libri_dev", lib_tr), mm.cached("libri_dev_phonehalf", mm.half(lib_tr), phone=True)
    T = {ph: mm.cached(f"heldout_{ph}", te_f, phone=ph == "phone") for ph in ("clean", "phone")}
    LT = {ph: mm.cached(f"libri_test_{ph}", lib_te, phone=ph == "phone") for ph in ("clean", "phone")}
    sd = data.DATA / "vishing" / "scam_dialogue"
    calls = [str(sd / d / f"{i}.wav") for d in ("tts", "tts_calib") for i in json.load(open(sd / d / "picked.json"))]
    OC = {ph: mm.cached(f"ourcalls_{ph}", calls, phone=ph == "phone") for ph in ("clean", "phone")}
    (_, cal_h), (_, test_h) = calib_sets(), test_sets()
    HU2, HUc = mm.cached("humans_2x2", test_h), mm.cached("humans_calib", cal_h)

    # new real sources
    have = [s for s in TRAIN_SRC if clips(s)]
    print(f"train real sources: {[(s, len(clips(s))) for s in have]}  (missing: {sorted(set(TRAIN_SRC) - set(have))})")
    print(f"test real sources:  {[(s, len(clips(s))) for s in TEST_SRC]}", flush=True)
    real = [Lr, Lrp]
    for s in have:
        real += [E(s, clips(s)), E(f"{s}_phonehalf", mm.half(clips(s)), phone=True)]
    RT = {s: {ph: E(f"{s}_{ph}", clips(s), phone=ph == "phone") for ph in ("clean", "phone")} for s in TEST_SRC}

    Xa, ya = mm.asv_rows()
    R = np.vstack(real)
    print(f"\ntraining rows: ASV {len(ya):,} + MLAAD {len(M) + len(Mp):,} fake + real {len(R):,} "
          f"(LibriSpeech + {len(have)} new sources, with phone twins)", flush=True)
    new = sm.fit_head(np.vstack([Xa, M, Mp, R]), np.r_[ya, np.ones(len(M) + len(Mp)), np.zeros(len(R))])
    joblib.dump(new, mm.ROOT / "head_realmix.pkl")
    heads = {"shipped": joblib.load(sm.OUT.with_suffix(".pkl")), "+realmix": new}

    d21, A21, e5, E5, yi, I = mm.gates()
    pst = (d21.codec == "pstn").values
    P = lambda h, X: h.predict_proba(X)[:, 1]
    fl = lambda h, X: float(np.mean(P(h, X) >= .5))
    rows = [
        ("A unseen gens caught, phone", lambda h: fl(h, T["phone"])),
        ("B our Edge calls caught, studio", lambda h: fl(h, OC["clean"])),
        ("B our Edge calls caught, phone", lambda h: fl(h, OC["phone"])),
        ("C Earnings-22 calls flagged", lambda h: fl(h, RT["earn"]["clean"])),
        ("C Earnings-22 flagged, phone", lambda h: fl(h, RT["earn"]["phone"])),
        ("C FLEURS flagged", lambda h: fl(h, RT["fleurs"]["clean"])),
        ("C FLEURS flagged, phone", lambda h: fl(h, RT["fleurs"]["phone"])),
        ("C scambait test channels flagged", lambda h: fl(h, HU2[:100])),
        ("C HVB B bank calls flagged", lambda h: fl(h, HU2[100:])),
        ("C calib humans (HVB A + scambait) flagged", lambda h: fl(h, HUc)),
        ("C LibriSpeech test flagged, phone", lambda h: fl(h, LT["phone"])),
        ("C ITW real flagged, phone", lambda h: fl(h, I["phone"][yi == 0])),
        ("D ASV21 EER pstn", lambda h: sm.eer(d21.label[pst], P(h, A21)[pst])),
        ("D ASV5 eval EER", lambda h: sm.eer(e5.label, P(h, E5))),
        ("D ITW EER phone", lambda h: sm.eer(yi, P(h, I["phone"]))),
    ]
    print(f"\n{'':44} {'shipped':>9} {'+realmix':>9}")
    res = {}
    for name, f in rows:
        res[name] = [f(h) for h in heads.values()]
        print(f"  {name:42} {res[name][0]:>9.1%} {res[name][1]:>9.1%}", flush=True)
    d = {k: v[1] - v[0] for k, v in res.items()}
    ok = (d["A unseen gens caught, phone"] >= .10 and d["B our Edge calls caught, phone"] >= .10
          and all(d[k] <= .02 for k in res if k.startswith("C "))
          and all(d[k] <= .01 for k in res if k.startswith("D ")))
    print(f"\n-> {'ADOPT' if ok else 'DO NOT ADOPT'} by the rule fixed in the docstring")
    allreal = np.vstack([RT["earn"]["phone"], RT["fleurs"]["phone"], HU2, HUc, I["phone"][yi == 0]])
    for k, h in heads.items():   # same false-alarm rate: 5% of the calibration humans
        t = np.quantile(P(h, HUc), 0.95)
        print(f"  equal-FA {k:9} our calls {np.mean(P(h, OC['phone']) >= t):.1%}  unseen gens "
              f"{np.mean(P(h, T['phone']) >= t):.1%}  all unseen real {np.mean(P(h, allreal) >= t):.1%}", flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    download() if "--download" in sys.argv else extract() if "--extract" in sys.argv else main()

# RESULT 2026-10-27 (scratch/_spoof_realmix.log). Real class = LibriSpeech + CV, People's
# Speech, VoxPopuli, AMI ihm/sdm, Svarah (18,793 rows with phone twins); no HVB/scambait.
# At 0.5 (the pre-fixed rule): DO NOT ADOPT - unseen gens 19.8 -> 89.2%, our calls 20.4 ->
# 95.4%, but scambait test channels 0 -> 25%, calib humans 5 -> 16%, Earnings 7.2 -> 15.8%,
# ASV5 eval EER 9.5 -> 14.1%, ASV21 pstn 6.5 -> 7.8%. ITW real 3.9 -> 3.0%, ITW EER 17.9 -> 16.3%.
# POST-HOC (found after the rule failed - needs confirming on fresh data): cut-off set so 5% of
# the 100 calibration humans flag (shipped 0.177, realmix 0.925), read on everything else:
#                               shipped  realmix
#   our AI calls (phone)          27.0%    79.6%
#   unseen AI tools (phone)       23.2%    79.0%
#   Earnings-22 / phone        9.5/12.0  6.7/5.6
#   FLEURS phone                   3.8%     3.3%
#   scambait new channels          2.0%    14.0%   <- the one real source that got worse
#   HVB B bank calls              14.0%     5.0%
#   LibriSpeech test phone         0.6%     2.2%
#   ITW real phone                 4.9%     1.4%
#   ITW fake phone (caught)       46.4%    33.0%   <- older real-world deepfakes caught less
# Unlike the MLAAD-only runs it RANKS better (unseen real false alarms mostly down at equal
# FA), i.e. the diverse real class removed most of the source shortcut. Not promoted.
