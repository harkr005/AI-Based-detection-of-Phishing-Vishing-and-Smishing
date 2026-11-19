"""Does the shipped spoof head survive deepfakes from the wild?

In-the-Wild (Mueller et al. 2022, arXiv 2203.16263): 31,779 clips of 54
celebrities and politicians from online video, 17.2 h fake / 20.7 h real,
CC-BY-SA 4.0 (Hugging Face mueller91/In-The-Wild). The fakes are whatever was
circulating up to 2022 - newer than ASVspoof 2019's 2018 vocoders, but not 2026
voice cloning. Real speech here is ALSO in-the-wild (studio, stage, phone), so
it tests "real vs fake" in conditions ASVspoof never had.

    python -m scratch._itw_eval          # embed (cached) + score the shipped head
    python -m scratch._itw_eval --fit    # + a head that has seen TRAIN speakers

Features are the shipped LAYER-20 mean+std embeddings, cached under the dataset
folder for clean and phone (spoof_model.telephony) audio, so a retrain does not
re-embed. Speaker split FIXED before any result was seen: seed-0 permutation of
the 54 speakers in meta.csv, 27 train / 27 test; every number marked "test" is on speakers
the head never heard. EER only, plus mean score on real voices (the
bonafide-inversion check from 2026-08-19). No audio or text is printed.
"""
import sys

import data  # before torch
import numpy as np
import pandas as pd
import spoof_model as sm

ROOT = data.DATA / "vishing" / "in_the_wild" / "release_in_the_wild"
meta = pd.read_csv(ROOT / "meta.csv")
meta["y"] = (meta.label == "spoof").astype(int)
spk = np.random.default_rng(0).permutation(sorted(meta.speaker.unique()))
meta["test"] = meta.speaker.isin(spk[len(spk) // 2:])
print(f"{len(meta):,} clips  {meta.y.mean():.1%} fake  {meta.speaker.nunique()} speakers  "
      f"test speakers {meta[meta.test].speaker.nunique()} ({meta.test.mean():.1%} of clips)")


def feats(phone):
    path = ROOT.parent / f"emb_l{sm.LAYER}_{'phone' if phone else 'clean'}.npy"
    if not path.exists():
        X = sm.embed([str(ROOT / f) for f in meta.file], phone=phone)[:, 0]
        np.save(path, X.astype(np.float16))
    return np.load(path).astype(np.float32)


def report(tag, p, saw_train=False):
    parts = [("test", meta.test.values)]
    if not saw_train:                    # "all" is fair only for a head that never saw ITW
        parts.insert(0, ("all", np.ones(len(meta), bool)))
    for part, m in parts:
        y = meta.y.values[m]
        print(f"  {tag:<28s} {part:<4s} EER {sm.eer(y, p[m]):6.2%}   "
              f"mean on real voices {p[m][y == 0].mean():.3f}   on fakes {p[m][y == 1].mean():.3f}")


import joblib  # noqa: E402
shipped = joblib.load(sm.OUT.with_suffix(".pkl"))
X = {ph: feats(ph) for ph in (False, True)}
for ph in (False, True):
    report(f"shipped head, {'phone' if ph else 'clean'}", shipped.predict_proba(X[ph])[:, 1])

if "--fit" in sys.argv:
    # Train speakers only, clean + phone stacked (the same augmentation recipe
    # the shipped head uses). ITW alone: does the in-the-wild signal transfer
    # across speakers at all? ASVspoof + ITW is the shippable recipe and needs
    # spoof_model's own training embeddings, so it lives there, not here.
    tr = ~meta.test.values
    head = sm.fit_head(np.vstack([X[False][tr], X[True][tr]]),
                       np.concatenate([meta.y.values[tr]] * 2))
    for ph in (False, True):
        report(f"ITW-train head, {'phone' if ph else 'clean'}", head.predict_proba(X[ph])[:, 1],
               saw_train=True)
