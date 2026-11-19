"""Is In-the-Wild teaching real-vs-fake, or how ITW was collected?

_itw_eval.py found (2026-10-13, 27 unseen test speakers): shipped head EER
31.7% clean / 35.1% phone, and it scores REAL voices at mean 0.596 - it fails
on wild audio. A head fit on the 27 train speakers: 1.57% / 2.84%. That second
number is the kind this project has been fooled by five times (PhiUSIIL HTTPS,
CEAS has-URL, ...): ITW's real clips and fake clips come from different videos,
so a head can learn the source instead of the voice. Three checks:

  1. duration alone - how well does clip length separate real from fake?
  2. transfer - the ITW-only head on ASVspoof 2021 LA (labelled, real codecs,
     same 400-per-codec-x-label draw as _paper_spoof.py). A head that learned
     "fake" should carry over; one that learned "YouTube deepfake upload" won't.
  3. the shippable recipe - ASVspoof 2019 train (+ half phone twins, exactly
     spoof_model's step 2) PLUS ITW train speakers (clean + phone), scored on
     ITW test speakers AND ASVspoof 2021. Ship only if it keeps 2021.

Embeddings are cached next to each corpus. Writes no model. Rates only.

    python -m scratch._spoof_itw_combined
"""
import data  # before torch
import numpy as np
import pandas as pd
import soundfile as sf
import spoof_model as sm
from sklearn.metrics import roc_auc_score

ITW = data.DATA / "vishing" / "in_the_wild"
meta = pd.read_csv(ITW / "release_in_the_wild" / "meta.csv")
meta["y"] = (meta.label == "spoof").astype(int)
spk = np.random.default_rng(0).permutation(sorted(meta.speaker.unique()))   # = _itw_eval
meta["test"] = meta.speaker.isin(spk[len(spk) // 2:])
te, tr = meta.test.values, ~meta.test.values
I = {ph: np.load(ITW / f"emb_l{sm.LAYER}_{'phone' if ph else 'clean'}.npy").astype(np.float32)
     for ph in (False, True)}


def cached(name, paths, phone=False):
    path = data.DATA / "vishing" / "LA" / f"emb_l{sm.LAYER}_{name}.npy"
    if not path.exists():
        np.save(path, sm.embed(list(paths), phone=phone)[:, 0].astype(np.float16))
    return np.load(path).astype(np.float32)


# 1. duration
dur = np.array([sf.info(str(ITW / "release_in_the_wild" / f)).duration for f in meta.file])
auc = roc_auc_score(meta.y, dur)
print(f"1. clip duration alone separates ITW real/fake at AUC {max(auc, 1 - auc):.3f} "
      f"(median real {np.median(dur[meta.y == 0]):.1f}s, fake {np.median(dur[meta.y == 1]):.1f}s)")

# ASVspoof: 2019 train (+ phone twins of half, spoof_model step 2), 2021 draw
a19 = data.load_spoof("train")
half = np.random.default_rng(0).permutation(len(a19))[:len(a19) // 2]
A19 = cached("a19_train_clean", a19.path)
A19p = cached("a19_train_phonehalf", a19.path.iloc[half], phone=True)
s21 = data.load_spoof21()
rng = np.random.default_rng(0)
d21 = pd.concat([g.iloc[rng.permutation(len(g))[:400]]
                 for _, g in s21.groupby(["codec", "label"])])          # = _paper_spoof
A21 = cached("a21_draw5600", d21.path)


def row(tag, head):
    p_itw = {ph: head.predict_proba(I[ph][te])[:, 1] for ph in (False, True)}
    p21 = head.predict_proba(A21)[:, 1]
    y = meta.y.values[te]
    print(f"  {tag:<30s} ITW test EER clean {sm.eer(y, p_itw[False]):6.2%}  "
          f"phone {sm.eer(y, p_itw[True]):6.2%}  real-voice mean {p_itw[True][y == 0].mean():.3f}"
          f"  |  ASVspoof21 EER {sm.eer(d21.label, p21):6.2%}  "
          f"pstn {sm.eer(d21.label[d21.codec == 'pstn'], p21[(d21.codec == 'pstn').values]):6.2%}")


import joblib  # noqa: E402
y19 = np.r_[a19.label, a19.label.iloc[half]]
yitw = np.concatenate([meta.y.values[tr]] * 2)
print("\n2./3. heads (ITW test = 27 speakers no head trained on)")
row("shipped (ASV19, from disk)", joblib.load(sm.OUT.with_suffix(".pkl")))
row("ASV19 refit here", sm.fit_head(np.vstack([A19, A19p]), y19))
row("ITW train only", sm.fit_head(np.vstack([I[False][tr], I[True][tr]]), yitw))
row("ASV19 + ITW train", sm.fit_head(np.vstack([A19, A19p, I[False][tr], I[True][tr]]),
                                     np.r_[y19, yitw]))

# RESULT 2026-10-24: duration alone AUC 0.619 (median real 3.0s, fake 4.0s).
#                     ITW test EER clean / phone, real-voice mean | ASV21 EER / pstn
#   shipped (ASV19)   31.73% / 35.13%  0.466                      |  3.25% /  3.50%
#   ITW train only     1.57% /  2.84%  0.052                      | 20.59% / 26.75%
#   ASV19 + ITW        2.77% /  3.52%  0.055                      |  7.54% /  9.00%
# The ITW-only head does not carry over to ASVspoof 2021 (20.6%), so part of its
# 1.6% is how ITW was collected, not "fake". The combined head keeps most of the
# ITW gain but more than doubles the 2021 EER (pstn 3.5 -> 9.0). Not promoted.
