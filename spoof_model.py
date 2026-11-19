"""Synthetic-voice detection (spec §4): wav2vec2-XLS-R front-end + a spoof head.

The other half of the voice channel. `voice_model.py` asks what the caller is
*saying*; this asks whether a human is saying it at all. §8 gives it 0.20 of the
voice verdict, and until now that slot has been passed None - no evidence, rather
than a guess.

Trained on ASVspoof 2019 LA + an ASVspoof 5 (2024) sample, half of each pushed
through `telephony()` first: both corpora are clean audio and no call ever is.
The 2019 split is the one this project uses everywhere: train and dev share
attacks A01-A06, eval uses A07-A19, algorithms the model has never heard. ASV5
train is attacks A01-A08; its eval attacks A17-A32 stay out of training.

Why ASV5 was added (2026-10-25, scratch/_spoof_asv5.py, EER):
    head            ASV21  pstn | ASV5 eval | In-the-Wild clean  phone | ITW real-voice mean
    ASV19 only       3.25  3.50 |   23.52   |   38.84   42.93          | 0.496
    ASV19 + ASV5     5.36  6.50 |    9.48   |   13.55   17.94          | 0.040
ASV19-only scored real people in real-world audio at 0.50 - it would accuse
ordinary callers. The cost is 3 points on ASVspoof 2021's PSTN line, a
2019-era lab attack over a landline; quote it next to the gain.

ponytail: the front-end is FROZEN and the head is small. Unfreezing XLS-R's top
12 layers was measured (scratch/_spoof_finetune.py, one epoch): ASV5 eval 9.48
-> 8.58 but In-the-Wild clean 13.55 -> 21.65, pstn unchanged. It fits the lab
corpora and loses the real-world deepfakes, so the frozen head stays.
"""
import numpy as np
import pandas as pd

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data
import soundfile as sf
import torch

XLSR = "facebook/wav2vec2-xls-r-300m"
SR = 16_000
CROP = 4.0                      # seconds; ASVspoof utterances run 1-10s
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = data.Path(__file__).parent / "models" / "spoof_head.npz"
# The shipped .pkl (2026-10-27) is NOT what __main__ below trains: it is the phonemix
# head (ASV19+5, MLAAD, many real voices incl. phone calls) from
# `python -m scratch._spoof_phonemix` then `--promote`, which bakes its 0.710 cut-off
# into the intercept so 0.5 still means "synthetic". Running __main__ overwrites it.

# Which transformer layer to read. Measured by the probe in __main__, and the
# result inverts the obvious guess. Judged on attacks it was trained on, the
# LOW layers win - layer 4 hits 0.00% EER, because the vocoder's own artefacts
# live down there and are trivially memorable. Judged on attacks held out of
# training, that ranking flips:
#
#     layer  4   seen 0.00%   unseen 16.46%
#     layer 20   seen 0.35%   unseen  8.88%
#
# The low layers memorise the six training vocoders; the high layers carry
# something about speech that transfers to vocoders they have never heard. An
# in-condition metric would have picked layer 4 and shipped the worse model.
LAYER = 20


def telephony(x, sr=SR):
    """Push studio audio through a phone channel: 300-3400 Hz band, 8 kHz
    sampling, 8-bit mu-law, back up to 16 kHz.

    ASVspoof 2019 LA is clean studio audio. Every call this project would ever
    see has been through a codec, and the vocoder artefacts a spoof detector
    keys on live largely in the band a phone throws away. The 2021 LA eval set
    exists for exactly this and is on disk - but it ships only a trial list, and
    the ground-truth keys are a separate download we do not have, so it cannot
    be scored. This simulates the same degradation on labelled 2019 audio.
    ponytail: a real GSM/AMR codec would be closer; this needs no new dependency
    and already answers the question.
    """
    from scipy.signal import butter, resample_poly, sosfilt
    x = sosfilt(butter(6, [300, 3400], btype="band", fs=sr, output="sos"), x)
    x = resample_poly(x, 1, 2)                       # 16k -> 8k
    mu, peak = 255.0, max(np.abs(x).max(), 1e-9)
    y = np.sign(x) * np.log1p(mu * np.abs(x / peak)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1      # 8-bit quantise
    x = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu * peak
    return resample_poly(x, 2, 1).astype(np.float32)  # 8k -> 16k


def _read(path):
    """-> (samples, sr). soundfile first, ffmpeg for what it cannot decode.

    ponytail: libsndfile 1.2.2 fails on ~40% of ASVspoof 2021's FLACs with
    "decoder lost sync". The files are fine - ffmpeg reads them without
    complaint - so this is a decoder limitation, not corruption, and the
    fallback keeps 70,000 usable files usable without re-encoding any of them.
    Spawning ffmpeg per file is slow; it only happens for the files that need
    it, and re-encoding the corpus up front would cost more than it saves.
    """
    try:
        return sf.read(str(path), dtype="float32")
    except Exception:
        import subprocess
        import imageio_ffmpeg
        r = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
             "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True)
        if r.returncode or not r.stdout:
            raise
        return np.frombuffer(r.stdout, dtype=np.float32).copy(), SR


def load_audio(path, seconds=CROP, phone=False):
    """-> fixed-length float32 at 16 kHz. Short clips repeat rather than
    zero-pad: silence is itself a cue, and padding teaches the head to read
    utterance length instead of the voice."""
    x, sr = _read(path)
    if x.ndim > 1:
        # ponytail: mix down rather than crash. A caller who wants one party
        # scored should hand in a channel from voice_model.split_stereo().
        x = x.mean(1)
    if sr != SR:
        # Resample, do not reject. Real telephony arrives at 8 kHz and §10's
        # gateway will never hand this model a 16 kHz studio file; an assert
        # here would make the model unusable on exactly the audio it is for.
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        sr = SR
    if phone:
        x = telephony(x, sr)
    n = int(seconds * SR)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    return x[:n]


def _frontend():
    from functools import lru_cache
    return _cached_frontend()


def _cached_frontend(_cache={}):
    if "m" not in _cache:
        from transformers import Wav2Vec2Model
        m = Wav2Vec2Model.from_pretrained(XLSR).to(DEV).eval().half()
        m.requires_grad_(False)
        _cache["m"] = m
    return _cache["m"]


def embed(paths, layers=(LAYER,), bs=8, phone=False):
    """-> (n, len(layers), 2*hidden) mean+std pooled features, float32.

    Mean and std, not mean alone: a vocoder's tell is often the *stability* of
    the frame sequence, which a mean throws away.
    """
    m = _frontend()
    out = []
    for i in range(0, len(paths), bs):
        wav = np.stack([load_audio(p, phone=phone) for p in paths[i:i + bs]])
        with torch.no_grad():
            h = m(torch.from_numpy(wav).to(DEV).half(),
                  output_hidden_states=True).hidden_states
        feats = [torch.cat([h[l].mean(1), h[l].std(1)], -1) for l in layers]
        out.append(torch.stack(feats, 1).float().cpu().numpy())
    return np.concatenate(out)


def eer(y, score):
    """Equal error rate - the metric ASVspoof is reported in, so the number can
    sit next to published work without a conversion."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, score)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def fit_head(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=C,
                                            class_weight="balanced")).fit(X, y)


def score(paths, head=None):
    """-> spoof probability per file, for §8's 0.20 voice slot."""
    import joblib
    head = head or joblib.load(OUT.with_suffix(".pkl"))
    return head.predict_proba(embed(list(paths))[:, 0])[:, 1]


def sample(d, n, seed=0):
    """Balanced subsample. ASVspoof is 90% spoof, so a random slice of 2,000 has
    ~200 bonafide and any EER read off it is mostly noise."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.flatnonzero(d.label == k))[:n // 2]
                          for k in (0, 1)])
    return d.iloc[np.sort(idx)].reset_index(drop=True)


def _selfcheck():
    """load_audio must hand back CROP seconds at SR whatever it was given."""
    import tempfile
    t = np.linspace(0, 3, 3 * 8000, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * t)
    with tempfile.TemporaryDirectory() as td:
        f = data.Path(td) / "t.wav"
        sf.write(f, np.stack([tone, tone], 1), 8000)      # 8 kHz STEREO, 3 s
        x = load_audio(f)
        assert x.shape == (int(CROP * SR),), x.shape      # resampled and tiled
        assert x.dtype == np.float32, x.dtype
        sf.write(f, tone, 8000)                           # 8 kHz mono
        assert load_audio(f).shape == (int(CROP * SR),)
    print("load_audio self-check ok (8 kHz stereo and mono -> 16 kHz mono)")


if __name__ == "__main__":
    import sys
    import time

    import joblib

    _selfcheck()

    tr, dv = data.load_spoof("train"), data.load_spoof("dev")
    print(f"train {len(tr):,} ({tr.label.mean():.0%} spoof, attacks "
          f"{sorted(set(tr.attack) - {'-'})})")
    print(f"dev   {len(dv):,} ({dv.label.mean():.0%} spoof)")

    # 1. Which layer? Probe on a small balanced subset before spending an hour
    #    embedding 50,000 files from the wrong one.
    #
    #    Selected on HELD-OUT ATTACKS, not on dev. Dev shares A01-A06 with train,
    #    so every layer scores under 1% there and the ranking is two files wide -
    #    picking on that is picking on noise, and worse, it says nothing about
    #    the only thing that matters at eval time: attacks never heard before.
    #    So the probe trains on A01-A04 and selects on A05-A06, which is the same
    #    shape as the real test without touching it.
    ALL = tuple(range(0, 25, 2))
    p_tr, p_dv = sample(tr, 2000), sample(dv, 2000)
    t = time.time()
    Xtr, Xdv = embed(p_tr.path.tolist(), ALL), embed(p_dv.path.tolist(), ALL)
    print(f"\nprobe embedded {len(p_tr) + len(p_dv):,} files in {time.time() - t:.0f}s")

    SEEN = ["A01", "A02", "A03", "A04"]
    fit_m = (p_tr.attack.isin(SEEN) | (p_tr.label == 0)).to_numpy()
    sel_m = (~p_dv.attack.isin(SEEN)).to_numpy()      # A05, A06 + all bonafide
    print(f"probe fits on {fit_m.sum()} (bonafide + A01-A04), "
          f"selects on {sel_m.sum()} (bonafide + A05-A06)")
    rows = {}
    for j, l in enumerate(ALL):
        h = fit_head(Xtr[fit_m, j], p_tr.label[fit_m])
        held = eer(p_dv.label[sel_m], h.predict_proba(Xdv[sel_m, j])[:, 1])
        full = eer(p_dv.label, fit_head(Xtr[:, j], p_tr.label)
                   .predict_proba(Xdv[:, j])[:, 1])
        rows[l] = held
        print(f"  layer {l:>2}   held-out-attack EER {held:>7.2%}   "
              f"all-dev EER {full:>7.2%}")
    best = min(rows, key=rows.get)
    print(f"-> layer {best} on unseen attacks; LAYER in the file is {LAYER}")
    if "--probe-only" in sys.argv:
        sys.exit()

    # 2. Full train on the chosen layer.
    #    Half the training set is pushed through telephony() first - §9.9's
    #    augmentation rule, the same one the text models use. Measured before
    #    it was adopted, on 6,000 paired eval files:
    #
    #                        studio EER   phone EER   bonafide-on-phone mean
    #      clean-trained        0.87%       5.30%          0.610
    #      phone-augmented      1.02%       1.63%          0.016
    #
    #    The last column is the reason to accept it. A clean-trained head does
    #    not merely lose accuracy through a codec, it INVERTS: the mean score
    #    on real human voices goes to 0.610, because the head partly learned
    #    "clean = human" and a phone line is not clean. Deployed on calls it
    #    would accuse every caller. Augmentation buys that back for 0.15
    #    points of studio EER - on the full eval set that is 0.78% -> 1.07%
    #    (dev 0.11% -> 0.32%), and the worst unseen attacks move A10 1.72 ->
    #    2.17, A19 1.10 -> 1.82. Quote 1.07% as this head's studio number; the
    #    clean-trained head is kept at models/spoof_head_clean.pkl so the
    #    ablation can be re-run without reverting this file.
    #    2b. ASVspoof 5 train joins (see the docstring): every bonafide clip +
    #    3,000 per attack A01-A08, phone twins of half, seed 0 - the exact rows
    #    scratch/_spoof_asv5.py measured. Its float16 embedding caches are read
    #    when present, so this rebuilds the measured head without re-embedding.
    def cached(path, paths, phone=False):
        if path.exists():
            return np.load(path).astype(np.float32)
        return embed(list(paths), (LAYER,), phone=phone)[:, 0].astype(np.float16).astype(np.float32)

    t = time.time()
    half = np.random.default_rng(0).permutation(len(tr))[:len(tr) // 2]
    Xtr = cached(data.DATA / "vishing" / "LA" / f"emb_l{LAYER}_a19_train_clean.npy", tr.path)
    Xph = cached(data.DATA / "vishing" / "LA" / f"emb_l{LAYER}_a19_train_phonehalf.npy",
                 tr.path.iloc[half], phone=True)
    t5 = data.load_spoof5("train")
    assert len(t5) == 182_357, f"only {len(t5):,} ASVspoof 5 train flacs on disk"
    rng5 = np.random.default_rng(0)
    t5 = pd.concat([t5[t5.label == 0]] +
                   [g.iloc[rng5.permutation(len(g))[:3000]] for _, g in t5[t5.label == 1].groupby("attack")]
                   ).reset_index(drop=True)
    h5 = np.random.default_rng(0).permutation(len(t5))[:len(t5) // 2]
    X5 = cached(data.ASV5 / f"emb_l{LAYER}_t5_clean.npy", t5.path)
    X5p = cached(data.ASV5 / f"emb_l{LAYER}_t5_phonehalf.npy", t5.path.iloc[h5], phone=True)
    Xdv = embed(dv.path.tolist(), (LAYER,))[:, 0]
    print(f"\ntrain ASV19 {len(tr):,} + {len(half):,} phone twins, ASV5 {len(t5):,} + {len(h5):,} phone twins; "
          f"{(time.time() - t) / 60:.1f} min")
    head = fit_head(np.vstack([Xtr, Xph, X5, X5p]),
                    np.r_[tr.label, tr.label.iloc[half], t5.label, t5.label.iloc[h5]])
    s_dv = head.predict_proba(Xdv)[:, 1]
    print(f"dev  EER {eer(dv.label, s_dv):.2%}   (attacks A01-A06, seen in training)")

    # 3. Eval: attacks A07-A19, none of them in training. The real number.
    ev = data.load_spoof("eval")
    Xev = embed(ev.path.tolist(), (LAYER,))[:, 0]
    s_ev = head.predict_proba(Xev)[:, 1]
    print(f"eval EER {eer(ev.label, s_ev):.2%}   (attacks A07-A19, all unseen)")
    per = (ev.assign(s=s_ev).query("attack != '-'")
             .groupby("attack").apply(lambda g: eer(
                 np.r_[np.zeros((ev.label == 0).sum()), np.ones(len(g))],
                 np.r_[s_ev[(ev.label == 0).to_numpy()], g.s])))
    print("worst attacks:", per.nlargest(3).round(4).to_dict())
    print("best attacks: ", per.nsmallest(3).round(4).to_dict())

    # 4. Does it survive a phone line? The whole channel exists for calls.
    sub = sample(ev, 6000)
    clean = head.predict_proba(embed(sub.path.tolist(), (LAYER,))[:, 0])[:, 1]
    phoned = head.predict_proba(embed(sub.path.tolist(), (LAYER,), phone=True)[:, 0])[:, 1]
    print()
    print(f"same {len(sub):,} eval files, paired:")
    print(f"  studio audio      EER {eer(sub.label, clean):.2%}")
    print(f"  through a phone   EER {eer(sub.label, phoned):.2%}")

    # 5. Is telephony() a fair simulation? Step 4 degrades studio audio with our
    #    own filter, so it can only ever confirm our own model of a phone line.
    #    These 501 Zenodo robocalls are natively 8 kHz - audio that crossed real
    #    phone networks. ASVspoof 2021 LA exists for this and ships no keys, so
    #    this is the closest available check.
    #
    #    NO LABELS HERE, so it is a distribution check and never an EER. Read it
    #    only for saturation: a head that scores real telephony at median 1.000
    #    is reacting to the channel, not to the voice. Measured when adopted:
    #      clean-trained     mean 0.722   median 1.000   >=0.5  72.5%
    #      phone-augmented   mean 0.530   median 0.716   >=0.5  52.7%
    #    Caveat that must travel with the number: robocalls really are often TTS,
    #    so a high score can be correct. What this shows is that the clean head
    #    SATURATES where the augmented head spreads - the same pathology
    #    telephony() predicted, now on audio it never touched.
    import glob
    real = sorted(glob.glob(str(data.DATA / "vishing" / "intl_robocall" /
                                "audio" / "english*" / "*.wav")))
    if real:
        s_real = head.predict_proba(embed(real, (LAYER,))[:, 0])[:, 1]
        print()
        print(f"{len(real)} real 8 kHz robocalls (no labels, distribution only):")
        print(f"  mean {s_real.mean():.3f}   median {np.median(s_real):.3f}   "
              f">=0.5 {(s_real >= 0.5).mean():.1%}")

    # 6. THE AUTHORITATIVE PHONE NUMBER. ASVspoof 2021 LA eval: the same attack
    #    families as step 3, but pushed through real transmission channels, with
    #    labels and a per-trial codec column. Steps 4 and 5 are our own
    #    simulation and an unlabelled distribution; this is ground truth.
    #
    #      codec   clean EER   aug EER   clean bonafide mean   aug
    #      none       0.87%     1.18%          0.005          0.007
    #      g722       1.63%     1.84%          0.011          0.012
    #      opus       1.95%     1.84%          0.048          0.025
    #      alaw       2.37%     2.11%          0.096          0.013
    #      ulaw       2.66%     2.50%          0.091          0.013
    #      gsm        5.00%     3.89%          0.508          0.140
    #      pstn       7.21%     3.21%          0.841          0.130
    #      ALL        8.49%     3.42%          0.229          0.049
    #
    #    Three things this settles:
    #    a) Augmentation is worth it: pooled EER 8.49% -> 3.42%. The only
    #       condition where the clean head wins is `none`, the studio case.
    #    b) The bonafide inversion is REAL, not a simulation artefact. On pstn
    #       the clean head scores genuine human voices at mean 0.841 - deployed
    #       on a phone line it accuses nearly every real caller. 0.130 augmented.
    #    c) telephony() is directionally right and optimistic on magnitude: it
    #       predicted 5.30% -> 1.63% where the truth on pstn is 7.21% -> 3.21%.
    #       Trust it to rank two heads; do not quote its numbers.
    #    Note the pooled EER beats every per-codec EER for the augmented head but
    #    not the clean one: augmentation also made scores COMPARABLE across
    #    channels, which is what a single deployed threshold actually needs.
    if data.ASV21_KEYS.exists() and data.ASV21.exists():
        s21 = data.load_spoof21()
        rng21 = np.random.default_rng(0)
        take = pd.concat([g.iloc[rng21.permutation(len(g))[:400]]
                          for _, g in s21.groupby(["codec", "label"])])
        p21 = head.predict_proba(embed(take.path.tolist(), (LAYER,))[:, 0])[:, 1]
        take = take.assign(p=p21)
        print()
        print("ASVspoof 2021 LA, real transmission channels (labelled):")
        for c, g in take.groupby("codec"):
            print(f"  {c:<6} EER {eer(g.label, g.p):>6.2%}   "
                  f"bonafide mean {g[g.label == 0].p.mean():.3f}")
        print(f"  {'ALL':<6} EER {eer(take.label, take.p):>6.2%}")

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(head, OUT.with_suffix(".pkl"))
    print(f"\nsaved {OUT.with_suffix('.pkl')}")
