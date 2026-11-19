"""Fresh spoof EER for the paper: the SHIPPED head (models/spoof_head.pkl) next to
the previous one (models/spoof_head_asv5.pkl), from cached embeddings. Retrains
nothing. EER needs no threshold, so the phonemix head's baked-in cut-off
(scratch/_spoof_phonemix.py --promote) does not move it.

  In-the-Wild   real-world celebrity deepfakes, clean and phone-channel
  ASVspoof 21   LA eval, real transmission channels, labelled (pooled + pstn);
                same seeded draw as spoof_model.__main__ step 6
  ASVspoof 5    eval, 400 per codec x label

None of these is in either head's training data.
    python -m scratch._paper_spoof
"""
import data                       # before torch
import joblib

import spoof_model as sm
from scratch import _spoof_mlaad as mm

d21, A21, e5, E5, yi, I = mm.gates()
pst = (d21.codec == "pstn").values
heads = {"shipped (phonemix)": joblib.load(sm.OUT.with_suffix(".pkl"))}
old = sm.OUT.parent / "spoof_head_asv5.pkl"
if old.exists():
    heads["previous (ASV19+5)"] = joblib.load(old)
print(f"{'':22} {'ITW clean':>10} {'ITW phone':>10} {'ASV21 all':>10} {'ASV21 pstn':>11} {'ASV5 eval':>10}")
for name, h in heads.items():
    P = lambda X: h.predict_proba(X)[:, 1]
    p21 = P(A21)
    print(f"{name:22} {sm.eer(yi, P(I['clean'])):>10.2%} {sm.eer(yi, P(I['phone'])):>10.2%} "
          f"{sm.eer(d21.label, p21):>10.2%} {sm.eer(d21.label[pst], p21[pst]):>11.2%} "
          f"{sm.eer(e5.label, P(E5)):>10.2%}")
