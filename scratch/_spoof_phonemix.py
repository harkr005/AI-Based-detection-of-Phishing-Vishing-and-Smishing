"""Real PHONE-CALL voices in the real class: does that stop the realmix head
calling real scam-call recordings synthetic?

_spoof_confirm.py (2026-10-27) failed one condition: on 105 real scam-call
recordings from 7 new YouTube channels the realmix head flagged 10.5% at its
0.925 cut-off (deployed 1.9%), confidently (0.93-1.00). The real class had no
real telephone calls and little YouTube audio. This run adds:

  CallHome (BrunoHays/callhome-telephony, CC-BY-NC-SA-4.0): real international
      phone calls between family/friends, 8 kHz, one speaker per channel.
      TRAIN en/es/de/ja; zh is HELD OUT as an unseen-language telephone test.
  Open Yap 1K shard 0 (CC-BY-4.0): casual calls between friends/family made
      on an app instead of the phone, one speaker per track, 48 kHz.
  VoxCeleb1 test (YouTube interviews, 40 speakers).
  + everything _spoof_realmix trained on (ASV19+ASV5, MLAAD train families,
    LibriSpeech, CV, People's Speech, VoxPopuli, AMI, Svarah).
Long calls are cut into random CLIP-s slices from one channel, silent slices dropped.

FIXED 2026-10-27 BEFORE ANY SCORE WAS SEEN:
  cut-off  CUT = 95th percentile of the new head on the 100 calibration humans
           (the same procedure that gave realmix 0.925), computed before any test
  PASS iff (1) fresh unseen-tool fakes (phone) caught >= deployed@0.5 + 20 pts
           (2) fresh Edge-TTS (phone) caught >= deployed + 20 pts
           (3) EVERY deciding real set flagged <= deployed + 2 pts:
               EdAcc, EdAcc phone, scambait NEW channels, CallHome zh
  The fresh fakes / EdAcc / scambait-new sets are _spoof_confirm's. NOTE: this is
  the SECOND attempt against the scambait-new set, so a pass is weaker evidence
  than a first-time pass; CallHome zh is the only never-scored deciding set.
Writes Datasets/vishing/mlaad/head_phonemix.pkl, never models/. Rates only.

    python -m scratch._spoof_phonemix --extract
    python -m scratch._spoof_phonemix
"""
import io
import os
import sys
import tarfile
import zipfile

import data   # before torch (Windows libiomp)
import numpy as np
import soundfile as sf

import spoof_model as sm
from scratch import _spoof_mlaad as mm
from scratch import _spoof_realmix as rm

ROOT = data.DATA / "vishing" / "phonemix"
RAW = ROOT / "raw"
CLIP = 8.0
N = {"ch_en": 1500, "ch_es": 1500, "ch_de": 1500, "ch_ja": 1500, "ch_zh": 600, "yap": 2000, "vox": 2000}
TRAIN = ["ch_en", "ch_es", "ch_de", "ch_ja", "yap", "vox"]


def _slices(x, sr, n, rng, out, k):
    """n random CLIP-s slices of one channel, skipping near-silent ones."""
    L = int(CLIP * sr)
    if len(x) <= L:
        return k
    st = rng.integers(0, len(x) - L, n * 3)
    rms = np.array([np.sqrt(np.mean(x[s:s + L] ** 2)) for s in st])
    keep = [s for s, r in zip(st, rms) if r >= 0.25 * np.median(rms)][:n]
    for s in keep:
        rm._write(x[s:s + L], sr, out / f"{k}.wav")
        k += 1
    return k


def extract():
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    for lang in ("en", "es", "de", "ja", "zh"):
        src, out = f"ch_{lang}", ROOT / f"ch_{lang}"
        if out.exists() and any(out.iterdir()):
            continue
        out.mkdir(parents=True, exist_ok=True)
        tab = pq.read_table(RAW / f"callhome_{lang}.parquet", columns=["audio"]).column("audio").to_pylist()
        per, k = -(-N[src] // (2 * len(tab))), 0
        for row in tab:
            a = rm._decode(row["bytes"])
            if a is None:
                continue
            x, sr = a
            for ch in range(x.shape[1] if x.ndim > 1 else 1):
                k = _slices(x[:, ch] if x.ndim > 1 else x, sr, per, rng, out, k)
        print(f"  {src}: {k} clips from {len(tab)} calls", flush=True)

    out = ROOT / "yap"
    if not (out.exists() and any(out.iterdir())):
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(RAW / "openyap_0.tar") as t:
            tracks = [m for m in t.getmembers() if m.name.endswith((".a.flac", ".b.flac"))]
            per, k = -(-N["yap"] // len(tracks)), 0
            for m in tracks:
                a = rm._decode(t.extractfile(m).read())
                if a is not None:
                    k = _slices(a[0] if a[0].ndim == 1 else a[0].mean(1), a[1], per, rng, out, k)
        print(f"  yap: {k} clips from {len(tracks)} tracks ({len(tracks) // 2} calls)", flush=True)

    out = ROOT / "vox"
    if not (out.exists() and any(out.iterdir())):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW / "voxceleb1_test.zip") as z:
            w = sorted(n for n in z.namelist() if n.endswith(".wav"))
            k = 0
            for i in sorted(rng.permutation(len(w))[:N["vox"] * 2]):
                a = rm._decode(z.read(w[i]))
                if a is not None and len(a[0]) >= 2 * a[1]:
                    rm._write(*a, out / f"{k}.wav")
                    k += 1
                if k >= N["vox"]:
                    break
        print(f"  vox: {k} clips from {len({n.split('/')[-3] for n in w})} speakers", flush=True)
    print("extract done", flush=True)


def clips(src):
    return sorted((str(p) for p in (ROOT / src).glob("*.wav")), key=lambda p: int(os.path.basename(p)[:-4]))


def main():
    import json
    import joblib
    from scratch._spoof_wholecall import calib_sets
    E = lambda name, paths, phone=False: mm.cached(f"phonemix_{name}", paths, phone=phone)
    R = lambda name, paths, phone=False: mm.cached(f"realmix_{name}", paths, phone=phone)

    # the realmix training set, all cached
    files = [f for f in json.load(open(mm.ROOT / "picked.json")) if (mm.ROOT / f).exists()]
    tr_g, _ = mm.split(sorted({f.split("/")[2] for f in files}))
    tr_f = [str(mm.ROOT / f) for f in files if f.split("/")[2] in tr_g]
    lib_tr = sorted(str(p) for p in (mm.LIBRI / "dev-clean").rglob("*.flac"))
    M, Mp = mm.cached("train", tr_f), mm.cached("train_phonehalf", mm.half(tr_f), phone=True)
    real = [mm.cached("libri_dev", lib_tr), mm.cached("libri_dev_phonehalf", mm.half(lib_tr), phone=True)]
    for s in rm.TRAIN_SRC:
        real += [R(s, rm.clips(s)), R(f"{s}_phonehalf", mm.half(rm.clips(s)), phone=True)]
    # + phone calls: CallHome is already telephone audio (no twins); Open Yap / VoxCeleb get twins
    for s in TRAIN:
        real.append(E(s, clips(s)))
        if s in ("yap", "vox"):
            real.append(E(f"{s}_phonehalf", mm.half(clips(s)), phone=True))
    print("train phone-call sources: " + ", ".join(f"{s} {len(clips(s))}" for s in TRAIN), flush=True)

    Xa, ya = mm.asv_rows()
    Rx = np.vstack(real)
    print(f"training rows: ASV {len(ya):,} + MLAAD {len(M) + len(Mp):,} fake + real {len(Rx):,}", flush=True)
    new = sm.fit_head(np.vstack([Xa, M, Mp, Rx]), np.r_[ya, np.ones(len(M) + len(Mp)), np.zeros(len(Rx))])
    joblib.dump(new, mm.ROOT / "head_phonemix.pkl")
    dep = joblib.load(sm.OUT.with_suffix(".pkl"))
    P = lambda h, X: h.predict_proba(X)[:, 1]

    # cut-off FIRST, from the calibration humans only
    _, cal_h = calib_sets()
    cut = float(np.quantile(P(new, mm.cached("humans_calib", cal_h)), 0.95))
    print(f"cut-off (95th pct of calibration humans): {cut:.3f}", flush=True)

    C = lambda n: mm.cached(f"confirm_{n}", [])   # _spoof_confirm caches, already on disk
    zh = clips("ch_zh")
    S = {"fresh unseen tools, phone": C("tools_phone"), "fresh Edge-TTS, phone": C("edge_phone"),
         "EdAcc (as recorded)": C("edacc"), "EdAcc, phone": C("edacc_phone"),
         "scambait new channels": C("sb_new"), "CallHome zh (unseen language)": E("ch_zh", zh)}
    r = {k: (np.mean(P(dep, X) >= .5), np.mean(P(new, X) >= cut)) for k, X in S.items()}
    print(f"\n{'':32} {'deployed@0.5':>13} {'new@cut':>9}")
    for k, (a, b) in r.items():
        print(f"  {k:30} {a:>13.1%} {b:>9.1%}")
    d = {k: v[1] - v[0] for k, v in r.items()}
    ok = (d["fresh unseen tools, phone"] >= .20 and d["fresh Edge-TTS, phone"] >= .20
          and all(d[k] <= .02 for k in list(S)[2:]))
    print(f"\n-> {'PASS' if ok else 'FAIL'} by the rule fixed in the docstring")
    p = P(new, S["scambait new channels"])
    print(f"  scambait new channels, new-head scores: median {np.median(p):.3f}, 75th {np.percentile(p, 75):.3f}, "
          f">= 0.5: {int((p >= .5).sum())}/105", flush=True)

    # context only (not part of the rule)
    d21, A21, e5, E5, yi, I = mm.gates()
    pst = (d21.codec == "pstn").values
    ctx = {"ITW real flagged (phone)": np.mean(P(new, I["phone"][yi == 0]) >= cut),
           "ITW fakes caught (phone)": np.mean(P(new, I["phone"][yi == 1]) >= cut),
           "HVB B + scambait test flagged": np.mean(P(new, mm.cached("humans_2x2", [])) >= cut),
           "Earnings-22 flagged": np.mean(P(new, R("earn_clean", [])) >= cut),
           "ASV21 pstn EER": sm.eer(d21.label[pst], P(new, A21)[pst]),
           "ASV5 eval EER": sm.eer(e5.label, P(new, E5))}
    print("\ncontext (not in the rule): " + " | ".join(f"{k} {v:.1%}" for k, v in ctx.items()))


def promote():
    """Write head_phonemix_cal.pkl: the phonemix head with the cut-off baked into its
    intercept, so P >= 0.5 (what fusion.SPOOF_COUNTS and the UI test) is exactly
    old P >= cut. No code elsewhere changes. The user copies it into models/."""
    import copy
    import joblib
    from scratch._spoof_wholecall import calib_sets
    head = joblib.load(mm.ROOT / "head_phonemix.pkl")
    _, cal_h = calib_sets()
    cut = float(np.quantile(head.predict_proba(mm.cached("humans_calib", cal_h))[:, 1], 0.95))
    cal = copy.deepcopy(head)
    cal[-1].intercept_ = cal[-1].intercept_ - np.log(cut / (1 - cut))
    for name in ("confirm_sb_new", "confirm_tools_phone", "confirm_edacc", "humans_calib"):
        X = mm.cached(name, [])
        assert ((cal.predict_proba(X)[:, 1] >= .5) == (head.predict_proba(X)[:, 1] >= cut)).all(), name
    joblib.dump(cal, mm.ROOT / "head_phonemix_cal.pkl")
    print(f"cut {cut:.4f} baked in -> {mm.ROOT / 'head_phonemix_cal.pkl'}")


if __name__ == "__main__":
    extract() if "--extract" in sys.argv else promote() if "--promote" in sys.argv else main()

# RESULT 2026-10-27 (scratch/_spoof_phonemix.log). Real class + CallHome en/es/de/ja (3,456
# telephone clips), Open Yap (2,000 clips, only 8 calls / 16 people), VoxCeleb1 test (2,000,
# 40 speakers) + phone twins = 28,249 real rows. Cut-off (calib humans 95th pct) = 0.710.
#                                   deployed@0.5   new@0.710
#   fresh unseen tools, phone             20.0%      85.8%
#   fresh Edge-TTS, phone                 20.0%      96.7%
#   EdAcc / EdAcc phone              1.9 / 2.9%  0.8 / 0.6%
#   scambait new channels                  1.9%       6.7%   <- FAILS (+4.8 pts, 7 vs 2 of 105)
#   CallHome zh (unseen language)         10.9%       1.7%
# FAIL on (3) by one set (second attempt on it). Context: ITW real 2.3%, ITW fakes caught
# 43.2%, HVB B + scambait test 9.5% (deployed 6.0%), Earnings 3.6%, ASV21 pstn EER 8.1%
# (6.5), ASV5 eval EER 14.2% (9.5). Each real-data addition cut the scambait-new false
# alarms (realmix 10.5% -> 6.7%) while keeping the catch. Not promoted.
