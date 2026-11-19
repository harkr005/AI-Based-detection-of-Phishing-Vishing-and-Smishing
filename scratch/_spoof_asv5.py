"""Does ASVspoof 5 (2024) fix the spoof head's failure on real-world deepfakes
without losing the phone line?

_spoof_itw_combined.py (2026-10-24) found the shipped head (ASVspoof 2019 only)
at 31.7% EER on In-the-Wild, and that training on ITW itself trades the phone
line away (ASVspoof 2021 pstn 3.5 -> 9.0%). ASVspoof 5 is the other route:
newer attacks, ~2,000 crowdsourced speakers in ordinary rooms, and nothing
from ITW in training - so ITW becomes a fully external test.

Recipe FIXED before any result was seen:
  train  ASVspoof 2019 LA train (+ phone twins of half, spoof_model step 2)
         + ASVspoof 5 train: all 18,797 bonafide + 3,000 per attack A01-A08,
           phone twins of half (default_rng(0)), same as 2019.
  test   ASVspoof 2021 LA (the 5,600 draw of _paper_spoof.py: real phone
           lines, attacks A07-A19 unseen by either corpus's train)
         ASVspoof 5 eval slice (flac_E_aa only): attacks A17-A32, never in
           train, 400 bonafide + 400 spoof per codec condition, seed 0
         In-the-Wild, ALL 31,779 clips, clean and phone (no head here sees it)
  ship   only if 2021 pstn stays near 3.5% AND ITW / ASV5-eval improve.

Embeddings cached next to each corpus (float16). Writes no model. Rates only.

    python -m scratch._spoof_asv5
"""
import data  # before torch
import numpy as np
import pandas as pd
import spoof_model as sm

ASV5 = data.ASV5
ITW = data.DATA / "vishing" / "in_the_wild"
LA = data.DATA / "vishing" / "LA"


def cached(path, paths, phone=False, chunk=4000):
    """Saved in 4,000-clip parts so a GPU crash or a low-memory kill resumes
    instead of losing hours of embedding."""
    if not path.exists():
        paths = list(paths)
        parts = []
        for i in range(0, len(paths), chunk):
            part = path.with_name(f"{path.stem}.part{i // chunk:03d}.npy")
            if not part.exists():
                np.save(part, sm.embed(paths[i:i + chunk], phone=phone)[:, 0].astype(np.float16))
                print(f"  {path.name}: {min(i + chunk, len(paths)):,}/{len(paths):,}", flush=True)
            parts.append(part)
        np.save(path, np.concatenate([np.load(p) for p in parts]))
        for p in parts:
            p.unlink()
    return np.load(path).astype(np.float32)


# --- train: 2019 (same cache as _spoof_itw_combined) ---
a19 = data.load_spoof("train")
h19 = np.random.default_rng(0).permutation(len(a19))[:len(a19) // 2]
A19 = cached(LA / f"emb_l{sm.LAYER}_a19_train_clean.npy", a19.path)
A19p = cached(LA / f"emb_l{sm.LAYER}_a19_train_phonehalf.npy", a19.path.iloc[h19], phone=True)
y19 = np.r_[a19.label, a19.label.iloc[h19]]

# --- train: ASVspoof 5 sample ---
t5 = data.load_spoof5("train")
assert len(t5) == 182_357, f"only {len(t5):,} train flacs on disk - extraction incomplete"
rng = np.random.default_rng(0)
t5 = pd.concat([t5[t5.label == 0]] +
               [g.iloc[rng.permutation(len(g))[:3000]] for a, g in t5[t5.label == 1].groupby("attack")]
               ).reset_index(drop=True)
h5 = np.random.default_rng(0).permutation(len(t5))[:len(t5) // 2]
print(f"ASV5 train sample {len(t5):,} ({(t5.label == 0).sum():,} bonafide), phone twins {len(h5):,}")
A5 = cached(ASV5 / f"emb_l{sm.LAYER}_t5_clean.npy", t5.path)
A5p = cached(ASV5 / f"emb_l{sm.LAYER}_t5_phonehalf.npy", t5.path.iloc[h5], phone=True)
y5 = np.r_[t5.label, t5.label.iloc[h5]]

# --- tests ---
s21 = data.load_spoof21()
r = np.random.default_rng(0)
d21 = pd.concat([g.iloc[r.permutation(len(g))[:400]]
                 for _, g in s21.groupby(["codec", "label"])])          # = _paper_spoof
A21 = cached(LA / f"emb_l{sm.LAYER}_a21_draw5600.npy", d21.path)

e5 = data.load_spoof5("eval")
r = np.random.default_rng(0)
e5 = pd.concat([g.iloc[r.permutation(len(g))[:400]]
                for _, g in e5.groupby(["codec", "label"])]).reset_index(drop=True)
print(f"ASV5 eval slice {len(e5):,} clips, {e5.codec.nunique()} codec conditions, "
      f"attacks {e5[e5.label == 1].attack.nunique()}")
E5 = cached(ASV5 / f"emb_l{sm.LAYER}_e5_draw.npy", e5.path)

meta = pd.read_csv(ITW / "release_in_the_wild" / "meta.csv")
yitw = (meta.label == "spoof").astype(int).values
I = {ph: np.load(ITW / f"emb_l{sm.LAYER}_{'phone' if ph else 'clean'}.npy").astype(np.float32)
     for ph in (False, True)}


def row(tag, head):
    p21 = head.predict_proba(A21)[:, 1]
    pst = (d21.codec == "pstn").values
    p5 = head.predict_proba(E5)[:, 1]
    pi = {ph: head.predict_proba(I[ph])[:, 1] for ph in (False, True)}
    print(f"  {tag:<22} ASV21 EER {sm.eer(d21.label, p21):6.2%}  pstn {sm.eer(d21.label[pst], p21[pst]):6.2%}"
          f"  |  ASV5 eval {sm.eer(e5.label, p5):6.2%}"
          f"  |  ITW clean {sm.eer(yitw, pi[False]):6.2%}  phone {sm.eer(yitw, pi[True]):6.2%}"
          f"  real-voice mean {pi[True][yitw == 0].mean():.3f}", flush=True)
    return p5


import joblib  # noqa: E402
print("\nheads (no head trains on ASV21, ASV5 eval attacks, or ITW)")
row("shipped (ASV19)", joblib.load(sm.OUT.with_suffix(".pkl")))
row("ASV5 only", sm.fit_head(np.vstack([A5, A5p]), y5))
p5 = row("ASV19 + ASV5", sm.fit_head(np.vstack([A19, A19p, A5, A5p]), np.r_[y19, y5]))
print("\nASV19 + ASV5 on the ASV5 eval slice, per codec condition ('-' = none)")
for c, m in e5.groupby("codec").groups.items():
    m = np.asarray(m)
    print(f"  {c:<4} EER {sm.eer(e5.label[m], p5[m]):6.2%}  (n={len(m)})")

# RESULT 2026-10-25 (EER; ITW = all 31,779 clips, ASV21 = the 5,600 draw, pstn n=800)
#   head            ASV21   pstn  | ASV5 eval |  ITW clean  phone | ITW real-voice mean
#   shipped ASV19    3.25%  3.50% |   23.52%  |   38.84%  42.93%  |  0.496
#   ASV5 only       31.52% 39.25% |   12.93%  |   11.21%  15.77%  |  0.000
#   ASV19 + ASV5     5.36%  6.50% |    9.48%  |   13.55%  17.94%  |  0.040
# ASV19+ASV5 per codec on ASV5 eval: none 1.9%, worst C07 20.5%, C04 18.8%, C08 13.3%.
# ITW and ASV5 eval improve a lot; pstn 3.5 -> 6.5% is NOT "near 3.5%", so by the rule
# fixed above it does not ship on its own. Best trade measured (ASV19+ITW was pstn 9.0%).
# Not promoted; promotion is the user's call via /promote-model.
