"""Does training on MODERN fake voices (MLAAD, 143 English generators, 2023-26)
fix the spoof head's blindness to commercial neural TTS?

The shipped head (ASVspoof 2019 + 2024) flags 19% of Microsoft Edge TTS calls at
0.5 (_deepfake_intent_eval.py); lowering the cut-off or reading the whole call
both just flag more real people (_spoof_wholecall.py). The remaining lever is
the training data. MLAAD (huggingface.co/datasets/mueller91/MLAAD, CC-BY-NC-4.0,
gated - needs `hf auth login`) has ElevenLabs, OpenAI, Gemini, Cartesia, MiniMax,
XTTS, F5, VibeVoice and ~130 more, all reading M-AILABS audiobook text.

RECIPE AND RULE, fixed 2026-10-27 before any score was seen:
  split   generators grouped into FAMILIES (ElevenLabs v2/v3/Turbo = one family,
          see family()); families shuffled with default_rng(0), 30% held out.
          Edge-TTS is ALWAYS held out: our own test calls are Edge voices.
  train   the shipped rows (ASV19 + ASV5, phone twins of half, = new_head())
          + MLAAD train families, PER_GEN clips each, phone twins of half
          + LibriSpeech dev-clean as REAL (real people reading audiobooks), phone
            twins of half - without it the head can learn "audiobook = fake".
  test    A. unseen generators: MLAAD held-out families (not Edge) vs
             LibriSpeech test-clean (unseen speakers) - caught / flagged at 0.5,
             clean and phone, plus EER
          B. Edge-TTS: MLAAD's Edge clips + our 304 Edge-voiced calls
             (tts + tts_calib), first 4 s as deployed, studio and phone
          C. real people: the 2x2 humans (HVB B + scambait test channels) and
             the calibration humans (HVB A + scambait train) - flagged at 0.5
          D. the existing gates: ASVspoof 2021 (pooled, pstn), ASVspoof 5 eval,
             In-the-Wild (clean, phone) EER, and ITW real voices flagged at 0.5
  ADOPT only if: A phone caught >= +10 pts AND B (our calls, phone) >= +10 pts
          AND every human-flagged rate (C, ITW real, LibriSpeech test real)
          <= +2 pts AND ASV21 pstn / ASV5 eval / ITW phone EER <= +1 pt.
Writes the head to Datasets/vishing/mlaad/head_mlaad.pkl, never to models/;
promotion is the user's /promote-model. Rates and counts only.

    python -m scratch._spoof_mlaad --download    # resumable, ~5 GB
    python -m scratch._spoof_mlaad               # embed (cached) + train + test
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import data   # before torch (Windows libiomp)
import numpy as np
import pandas as pd

import spoof_model as sm

REPO = "mueller91/MLAAD"
ROOT = data.DATA / "vishing" / "mlaad"
LIBRI = data.DATA / "vishing" / "librispeech" / "LibriSpeech"
PER_GEN = 120
HOLD = 0.30
ALIAS = {"openaudio": "fish", "e": "f", "pocket": "kyutai"}   # f5 -> "f" after the digit strip


def family(g):
    """Generator folder name -> product family, so versions of one model never
    straddle the train/test split."""
    g = g.lower()
    g = re.sub(r"^(tts_models_(en|multilingual)_[a-z0-9-]+_|microsoft[ _]|facebook_|suno_|nari |hume |ringg )", "", g)
    k = re.split(r"[-_ .(]", g)[0].rstrip("0123456789") or g
    k = re.sub(r"tts$", "", k) or k
    return ALIAS.get(k, k)


def generators():
    from huggingface_hub import HfApi
    return sorted(x.path.split("/")[-1] for x in
                  HfApi().list_repo_tree(REPO, path_in_repo="fake/en", repo_type="dataset"))


def split(gens):
    fams = sorted({family(g) for g in gens} - {"edge"})
    held = set(np.random.default_rng(0).permutation(fams)[:round(HOLD * len(fams))]) | {"edge"}
    return [g for g in gens if family(g) not in held], [g for g in gens if family(g) in held]


def download():
    from huggingface_hub import HfApi
    api = HfApi()
    files = json.load(open(ROOT / "picked.json")) if (ROOT / "picked.json").exists() else []
    for g in [] if files else generators():
        f = sorted(x.path for x in api.list_repo_tree(REPO, path_in_repo=f"fake/en/{g}", repo_type="dataset")
                   if x.path.endswith(".wav"))
        files += [f[i] for i in sorted(np.random.default_rng(0).permutation(len(f))[:PER_GEN])]
    todo = [f for f in files if not (ROOT / f).exists()]
    print(f"{len(files):,} clips, {len(todo):,} to fetch", flush=True)
    json.dump(files, open(ROOT / "picked.json", "w"))

    # Plain HTTPS GETs, not hf_hub_download: MLAAD is Xet-backed and the Xet
    # client crawls on tiny files (~4 clips/s). Each GET costs one "resolvers"
    # request, capped at 5,000 per 5 min (~16/s), so 24 threads at ~2 s a file
    # sit just under the cap; a 429 backs off.
    import os
    import threading
    import time
    import urllib.parse
    import requests
    from huggingface_hub import get_token
    auth, local = {"Authorization": f"Bearer {get_token()}"}, threading.local()

    def get(f):
        s = local.__dict__.setdefault("s", requests.Session())
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{urllib.parse.quote(f)}"
        out = ROOT / f
        out.parent.mkdir(parents=True, exist_ok=True)
        for k in range(6):
            try:
                r = s.get(url, headers=auth, timeout=60)
                if r.status_code == 429:
                    time.sleep(30)
                    continue
                r.raise_for_status()
                tmp = out.with_suffix(".part")
                tmp.write_bytes(r.content)
                os.replace(tmp, out)
                return
            except Exception as e:
                err = e
                time.sleep(2 * (k + 1))
        print(f"  FAILED {f}: {err}", flush=True)

    with ThreadPoolExecutor(24) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 500 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    print("download done", flush=True)


def cached(name, paths, phone=False, chunk=2000):
    """sm.embed in saved parts, so a crash resumes (= _spoof_asv5.cached)."""
    path = ROOT / f"emb_l{sm.LAYER}_{name}.npy"
    if not path.exists():
        paths, parts = list(paths), []
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


def half(paths, seed=0):
    return [paths[i] for i in sorted(np.random.default_rng(seed).permutation(len(paths))[:len(paths) // 2])]


def asv_rows():
    """The shipped training rows (= _deepfake_intent_eval.new_head)."""
    tr = data.load_spoof("train")
    h = np.random.default_rng(0).permutation(len(tr))[:len(tr) // 2]
    t5 = data.load_spoof5("train")
    r = np.random.default_rng(0)
    t5 = pd.concat([t5[t5.label == 0]] + [g.iloc[r.permutation(len(g))[:3000]]
                                          for _, g in t5[t5.label == 1].groupby("attack")]).reset_index(drop=True)
    h5 = np.random.default_rng(0).permutation(len(t5))[:len(t5) // 2]
    L, A = data.DATA / "vishing" / "LA", lambda p: np.load(p).astype(np.float32)
    X = np.vstack([A(L / f"emb_l{sm.LAYER}_a19_train_clean.npy"), A(L / f"emb_l{sm.LAYER}_a19_train_phonehalf.npy"),
                   A(data.ASV5 / f"emb_l{sm.LAYER}_t5_clean.npy"), A(data.ASV5 / f"emb_l{sm.LAYER}_t5_phonehalf.npy")])
    return X, np.r_[tr.label, tr.label.iloc[h], t5.label, t5.label.iloc[h5]]


def gates():
    """ASV21 draw, ASV5 eval slice, ITW - cached embeddings + labels (= _spoof_asv5)."""
    s21 = data.load_spoof21()
    r = np.random.default_rng(0)
    d21 = pd.concat([g.iloc[r.permutation(len(g))[:400]] for _, g in s21.groupby(["codec", "label"])])
    A21 = np.load(data.DATA / "vishing" / "LA" / f"emb_l{sm.LAYER}_a21_draw5600.npy").astype(np.float32)
    e5 = data.load_spoof5("eval")
    r = np.random.default_rng(0)
    e5 = pd.concat([g.iloc[r.permutation(len(g))[:400]] for _, g in e5.groupby(["codec", "label"])]).reset_index(drop=True)
    E5 = np.load(data.ASV5 / f"emb_l{sm.LAYER}_e5_draw.npy").astype(np.float32)
    itw = data.DATA / "vishing" / "in_the_wild"
    yi = (pd.read_csv(itw / "release_in_the_wild" / "meta.csv").label == "spoof").astype(int).values
    I = {ph: np.load(itw / f"emb_l{sm.LAYER}_{ph}.npy").astype(np.float32) for ph in ("clean", "phone")}
    return d21, A21, e5, E5, yi, I


def main():
    import joblib
    from scratch._spoof_wholecall import calib_sets, test_sets

    files = json.load(open(ROOT / "picked.json"))
    files = [f for f in files if (ROOT / f).exists()]
    gen_of = lambda f: f.split("/")[2]
    tr_g, te_g = split(sorted({gen_of(f) for f in files}))
    tr_f = [str(ROOT / f) for f in files if gen_of(f) in tr_g]
    te_f = [str(ROOT / f) for f in files if gen_of(f) in te_g and family(gen_of(f)) != "edge"]
    ed_f = [str(ROOT / f) for f in files if family(gen_of(f)) == "edge"]
    print(f"MLAAD: {len(tr_g)} train generators ({len(tr_f):,} clips), {len(te_g) - 1} held-out "
          f"({len(te_f):,} clips) + Edge-TTS ({len(ed_f)} clips)")
    print(f"  held-out families: {sorted({family(g) for g in te_g})}", flush=True)
    lib_tr = sorted(str(p) for p in (LIBRI / "dev-clean").rglob("*.flac"))
    lib_te = sorted(str(p) for p in (LIBRI / "test-clean").rglob("*.flac"))

    # --- embeddings (first CROP s, as deployed) ---
    M = cached("train", tr_f)
    Mp = cached("train_phonehalf", half(tr_f), phone=True)
    Lr = cached("libri_dev", lib_tr)
    Lrp = cached("libri_dev_phonehalf", half(lib_tr), phone=True)
    T = {ph: cached(f"heldout_{ph}", te_f, phone=ph == "phone") for ph in ("clean", "phone")}
    LT = {ph: cached(f"libri_test_{ph}", lib_te, phone=ph == "phone") for ph in ("clean", "phone")}
    ED = {ph: cached(f"edge_{ph}", ed_f, phone=ph == "phone") for ph in ("clean", "phone")}
    sd = data.DATA / "vishing" / "scam_dialogue"
    calls = [str(sd / d / f"{i}.wav") for d in ("tts", "tts_calib") for i in json.load(open(sd / d / "picked.json"))]
    OC = {ph: cached(f"ourcalls_{ph}", calls, phone=ph == "phone") for ph in ("clean", "phone")}
    (_, cal_h), (_, test_h) = calib_sets(), test_sets()
    HU = {"2x2 humans": cached("humans_2x2", test_h), "calib humans": cached("humans_calib", cal_h)}

    # --- heads ---
    Xa, ya = asv_rows()
    print(f"\ntraining rows: ASV {len(ya):,} ({int(ya.sum()):,} fake) + MLAAD {len(M) + len(Mp):,} fake "
          f"+ LibriSpeech {len(Lr) + len(Lrp):,} real", flush=True)
    real = [Lr, Lrp]
    if CONV:
        # --conv (2026-10-27, after the plain run failed on real phone speech): REAL
        # conversational phone audio in the real class - HVB half A calls, agent AND
        # caller side, and scambait train-channel calls (+ phone twins of half),
        # minus the 100 calibration humans. Same rule as the docstring; HVB B and
        # scambait test channels become same-source, so ITW real is the independent check.
        import act_model
        cal = set(cal_h)
        hvb = data.load_harpervalley()
        hA = hvb[np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2]
        au = data.DATA / "vishing" / "harpervalley" / "data" / "audio"
        hv = [str(au / side / f"{i}.wav") for i in hA.id for side in ("agent", "caller")
              if str(au / "agent" / f"{i}.wav") not in cal]
        sb = data.load_scambait()
        sb = sb[~sb.channel.isin(act_model.SCAMBAIT_TEST_CHANNELS | act_model.SCAMBAIT_NEW_CHANNELS)]
        sbp = [str(data.DATA / "vishing" / "scambait" / f"{i}.wav") for i in sb.id]
        sbp = [x for x in sbp if data.Path(x).exists() and x not in cal]
        real += [cached("conv_hvb", hv), cached("conv_sb", sbp), cached("conv_sb_phonehalf", half(sbp), phone=True)]
        print(f"  + conversational real: {len(hv):,} HVB clips, {len(sbp)} scambait (+{len(sbp) // 2} phone)", flush=True)
    R = np.vstack(real)
    new = sm.fit_head(np.vstack([Xa, M, Mp, R]), np.r_[ya, np.ones(len(M) + len(Mp)), np.zeros(len(R))])
    tag = "+MLAAD+conv" if CONV else "+MLAAD"
    joblib.dump(new, ROOT / ("head_mlaad_conv.pkl" if CONV else "head_mlaad.pkl"))
    heads = {"shipped": joblib.load(sm.OUT.with_suffix(".pkl")), tag: new}

    d21, A21, e5, E5, yi, I = gates()
    pst = (d21.codec == "pstn").values
    P = lambda h, X: h.predict_proba(X)[:, 1]
    fl = lambda h, X: float(np.mean(P(h, X) >= .5))
    rows = [  # (label, fn(head), "up" = higher is better)
        ("A unseen gens caught, clean", lambda h: fl(h, T["clean"]), "up"),
        ("A unseen gens caught, phone", lambda h: fl(h, T["phone"]), "up"),
        ("A unseen gens EER vs Libri, phone", lambda h: sm.eer(np.r_[np.ones(len(T["phone"])), np.zeros(len(LT["phone"]))],
                                                               np.r_[P(h, T["phone"]), P(h, LT["phone"])]), "down"),
        ("B MLAAD Edge caught, clean", lambda h: fl(h, ED["clean"]), "up"),
        ("B MLAAD Edge caught, phone", lambda h: fl(h, ED["phone"]), "up"),
        ("B our Edge calls caught, studio", lambda h: fl(h, OC["clean"]), "up"),
        ("B our Edge calls caught, phone", lambda h: fl(h, OC["phone"]), "up"),
        ("C 2x2 humans flagged", lambda h: fl(h, HU["2x2 humans"]), "down"),
        ("C calib humans flagged", lambda h: fl(h, HU["calib humans"]), "down"),
        ("C LibriSpeech test real flagged, phone", lambda h: fl(h, LT["phone"]), "down"),
        ("C ITW real flagged, phone", lambda h: fl(h, I["phone"][yi == 0]), "down"),
        ("D ASV21 EER pooled", lambda h: sm.eer(d21.label, P(h, A21)), "down"),
        ("D ASV21 EER pstn", lambda h: sm.eer(d21.label[pst], P(h, A21)[pst]), "down"),
        ("D ASV5 eval EER", lambda h: sm.eer(e5.label, P(h, E5)), "down"),
        ("D ITW EER clean", lambda h: sm.eer(yi, P(h, I["clean"])), "down"),
        ("D ITW EER phone", lambda h: sm.eer(yi, P(h, I["phone"])), "down"),
    ]
    print(f"\n{'':40} {'shipped':>9} {tag:>12}")
    res = {}
    for name, f, _ in rows:
        res[name] = [f(h) for h in heads.values()]
        print(f"  {name:38} {res[name][0]:>9.1%} {res[name][1]:>9.1%}", flush=True)

    d = {k: v[1] - v[0] for k, v in res.items()}
    ok = (d["A unseen gens caught, phone"] >= .10 and d["B our Edge calls caught, phone"] >= .10
          and all(d[k] <= .02 for k in res if k.startswith("C "))
          and all(d[k] <= .01 for k in ("D ASV21 EER pstn", "D ASV5 eval EER", "D ITW EER phone")))
    print(f"\n-> {'ADOPT' if ok else 'DO NOT ADOPT'} by the rule fixed in the docstring")
    # same false-alarm rate: cut-off where 5% of the calibration humans flag
    for k, h in heads.items():
        t = np.quantile(P(h, HU["calib humans"]), 0.95)
        print(f"  equal-FA {k:12} our calls {np.mean(P(h, OC['phone']) >= t):.1%}  unseen gens "
              f"{np.mean(P(h, T['phone']) >= t):.1%}  2x2 humans {np.mean(P(h, HU['2x2 humans']) >= t):.1%}  "
              f"ITW real {np.mean(P(h, I['phone'][yi == 0]) >= t):.1%}", flush=True)


CONV = "--conv" in sys.argv

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    download() if "--download" in sys.argv else main()

# RESULT 2026-10-27 (scratch/_spoof_mlaad.log). Held-out: 41 generators in 29 families
# (ElevenLabs, OpenAI, Gemini, Resemble, Tortoise, RVC, ...) + Edge; flag = score >= 0.5.
#                                         shipped   +MLAAD
#   A unseen generators caught, phone       19.8%    89.3%
#   B our Edge calls caught, phone          20.4%    92.8%
#   C 2x2 humans flagged                     6.0%    52.0%
#   C calib humans flagged                   5.0%    32.0%
#   C ITW real flagged, phone                3.9%    19.8%
#   C LibriSpeech test real, phone           0.5%     6.8%
#   D ASV21 pstn / ASV5 eval / ITW phone    6.5/9.5/17.9   7.3/11.7/23.0  (EER %)
# DO NOT ADOPT. At an EQUAL false-alarm rate (cut-off set so 5% of calib humans flag):
#   shipped  our calls 27.0%  unseen gens 23.2%  2x2 humans 8.0%  | AUC calls vs humans .828
#   +MLAAD   our calls  6.2%  unseen gens 25.5%  2x2 humans 11.5% | AUC .702
# The catch is bought entirely with false alarms. Reading: every REAL training clip is
# clean read speech (ASV19, ASV5, LibriSpeech) while MLAAD's fakes span many styles, so
# the head learned "clean read speech = real, anything else = fake" - real
# conversational phone audio lands on the fake side. The missing ingredient is REAL
# conversational/phone speech in training; untested (would make HVB/scambait in-domain).

# RESULT --conv 2026-10-27 (scratch/_spoof_mlaad_conv.log): + 1,346 HVB A clips + 269 scambait
# train-channel calls (+134 phone) as REAL.               shipped   +MLAAD+conv
#   A unseen generators caught, phone                       19.8%     89.1%
#   B our Edge calls caught, phone                          20.4%     89.1%
#   C calib humans (same sources as training)                5.0%      6.0%
#   C 2x2 humans                                             6.0%     27.5%
#       scambait TEST channels (unseen YouTube channels)     0%       54%
#       HVB B (same corpus as training, other calls)        12%        1%
#   C ITW real flagged, phone                                3.9%     19.2%
#   D ASV21 pstn / ASV5 eval / ITW phone EER          6.5/9.5/17.9  7.1/12.9/23.4
# DO NOT ADOPT. It learned WHICH RECORDINGS are real, not what a human voice is: real
# speech from a source it trained on passes (HVB 1%), real speech from any new source is
# called fake (unseen scambait channels 54%, ITW 19%). With a handful of real sources the
# real class cannot generalise; that needs many diverse real recording sources, not more
# fakes. Shipped head stays.
