"""Deepfake x intent: does the synthetic-voice slot help the voice verdict, or
only add false alarms? No public English corpus pairs synthetic voices with
scam labels, so this builds a 2 x 2 from what is on disk plus neural TTS:

                    human voice                      AI voice (Edge neural TTS)
    scam     held-out scambait calls (real)        Scam Dialogue scam half, voiced
    legit    HarperValley B bank calls (real)      Scam Dialogue legit half, voiced

100 per cell, seed 0. Fixed before any result was seen. Each call goes through
voice_model.analyse's own fusion (fusion.fuse_voice, no caller number):
semantic = score_transcript on the transcript we hold (the dialogue text for
the TTS calls - clean TTS through Whisper is near-verbatim, and it keeps a GPU
transcription job off this machine); spoof = spoof_model's first-4-s score,
exactly as analyse() reads an upload. AI calls are also scored through
telephony(), since a call is never studio audio.

Heads: the shipped ASV19 head, and ASV19 + ASV5 (rebuilt from the cached
embeddings exactly as spoof_model.__main__ now trains it).

CAVEATS that travel with the numbers:
  - stock Microsoft voices, not a clone of a specific person;
  - the human scam row is YouTube scam-baiting audio, the human legit row is a
    recorded bank-call corpus: different sources, so read the human row for
    false alarms and catch, not as a same-source accuracy;
  - first 4 s of each file, as deployed: a greeting, an intro, a ringtone.
Scam Dialogue: huggingface.co/datasets/BothBosu/scam-dialogue, Apache-2.0.
Dialogue text goes to Microsoft's TTS service; it is a public dataset.
Rates and counts only.

    python -m scratch._deepfake_intent_eval --tts     # resumable, ~30-45 min, network
    python -m scratch._deepfake_intent_eval           # score the 2 x 2
"""
import asyncio
import json
import os
import re
import sys
import tempfile

import data                       # before torch
import numpy as np
import pandas as pd
import soundfile as sf
import spoof_model as sm

SD = data.DATA / "vishing" / "scam_dialogue"
TTS = SD / "tts"
PICK = TTS / "picked.json"
TTS_CAL = SD / "tts_calib"          # calibration calls: never in the 2 x 2
N = 25                            # per Scam Dialogue type -> 100 per class


def picked():
    d = pd.read_csv(SD / "scam-dialogue_all.csv").drop_duplicates("dialogue")
    r = np.random.default_rng(0)
    idx = [int(i) for _, g in d.groupby("type") for i in g.index[r.permutation(len(g))[:N]]]
    return d.loc[idx]


async def voice(rows, voices, out_dir=TTS, conc=4):
    import edge_tts
    from scipy.signal import resample_poly
    sem = asyncio.Semaphore(conc)
    rng = np.random.default_rng(0)
    pairs = {i: rng.choice(voices, 2, replace=False) for i in rows.index}

    async def turn(text, v, tmp):
        for k in range(4):
            try:
                await edge_tts.Communicate(text, v, connect_timeout=30, receive_timeout=60).save(tmp)
                x, sr = sm._read(tmp)
                return resample_poly(x, sm.SR, sr).astype(np.float32) if sr != sm.SR else x
            except Exception:
                await asyncio.sleep(5 * (k + 1))
        raise RuntimeError("TTS failed 4 times")

    async def one(i, text):
        out = out_dir / f"{i}.wav"
        if out.exists():
            return
        async with sem:
            bits = re.split(r"\b(caller|receiver):", text)[1:]
            gap = np.zeros(int(0.3 * sm.SR), np.float32)
            audio = []
            with tempfile.TemporaryDirectory() as td:
                for j in range(0, len(bits) - 1, 2):
                    t = bits[j + 1].strip()
                    if t:
                        v = pairs[i][0] if bits[j] == "caller" else pairs[i][1]
                        audio += [await turn(t, v, os.path.join(td, f"{j}.mp3")), gap]
            tmp = out.with_suffix(".tmp.wav")
            sf.write(tmp, np.concatenate(audio), sm.SR)
            os.replace(tmp, out)
            print(f"  voiced {len(list(out_dir.glob('*[0-9].wav')))}/{len(rows)}", flush=True)

    await asyncio.gather(*(one(i, t) for i, t in rows.dialogue.items()))


def tts(calib=False):
    """calib: 13 per type from the dialogues the 2 x 2 did NOT pick, seed 1."""
    import edge_tts
    out = TTS_CAL if calib else TTS
    out.mkdir(parents=True, exist_ok=True)
    if calib:
        d = pd.read_csv(SD / "scam-dialogue_all.csv").drop_duplicates("dialogue")
        d = d.drop(json.load(open(PICK)))
        r = np.random.default_rng(1)
        rows = d.loc[[int(i) for _, g in d.groupby("type") for i in g.index[r.permutation(len(g))[:13]]]]
        json.dump([int(i) for i in rows.index], open(out / "picked.json", "w"))
    else:
        rows = picked()
        json.dump([int(i) for i in rows.index], open(PICK, "w"))
    voices = sorted(v["ShortName"] for v in asyncio.run(edge_tts.list_voices())
                    if v["Locale"] in ("en-US", "en-GB", "en-IN", "en-AU"))
    print(f"{len(rows)} dialogues, {len(voices)} voices", flush=True)
    asyncio.run(voice(rows, voices, out))
    print("done")


def new_head():
    """ASV19 + ASV5, the rows spoof_model.__main__ trains on, from the caches."""
    tr = data.load_spoof("train")
    half = np.random.default_rng(0).permutation(len(tr))[:len(tr) // 2]
    t5 = data.load_spoof5("train")
    r = np.random.default_rng(0)
    t5 = pd.concat([t5[t5.label == 0]] + [g.iloc[r.permutation(len(g))[:3000]]
                                          for _, g in t5[t5.label == 1].groupby("attack")]).reset_index(drop=True)
    h5 = np.random.default_rng(0).permutation(len(t5))[:len(t5) // 2]
    L, A = data.DATA / "vishing" / "LA", lambda p: np.load(p).astype(np.float32)
    X = np.vstack([A(L / f"emb_l{sm.LAYER}_a19_train_clean.npy"), A(L / f"emb_l{sm.LAYER}_a19_train_phonehalf.npy"),
                   A(data.ASV5 / f"emb_l{sm.LAYER}_t5_clean.npy"), A(data.ASV5 / f"emb_l{sm.LAYER}_t5_phonehalf.npy")])
    return sm.fit_head(X, np.r_[tr.label, tr.label.iloc[half], t5.label, t5.label.iloc[h5]])


def test_cells():
    """The 2 x 2 test: cells, intent scores, first-4-s embeddings (+ AI through telephony())."""
    import act_model
    import voice_model as vm
    d = pd.read_csv(SD / "scam-dialogue_all.csv").loc[json.load(open(PICK))]
    d["path"] = [str(TTS / f"{i}.wav") for i in d.index]
    assert all(os.path.exists(p) for p in d.path), "run --tts first"
    r = np.random.default_rng(0)
    sb = data.load_scambait()
    sb = sb[sb.channel.isin(act_model.SCAMBAIT_TEST_CHANNELS)]
    sb = sb.iloc[r.permutation(len(sb))[:100]]
    hvb = data.load_harpervalley()
    hvb_b = hvb[~(np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2)]   # = _voice_llm_eval
    hvb_b = hvb_b.iloc[np.random.default_rng(0).permutation(len(hvb_b))[:100]]
    cells = {
        "AI scam":     (d[d.label == 1].path.tolist(), d[d.label == 1].dialogue.tolist(), 1),
        "AI legit":    (d[d.label == 0].path.tolist(), d[d.label == 0].dialogue.tolist(), 0),
        "human scam":  ([str(data.DATA / "vishing" / "scambait" / f"{i}.wav") for i in sb.id], sb.transcript.tolist(), 1),
        "human legit": ([str(data.DATA / "vishing" / "harpervalley" / "data" / "audio" / "agent" / f"{i}.wav")
                         for i in hvb_b.id], hvb_b.transcript.tolist(), 0),
    }
    sem = {c: np.array([vm.score_transcript(t)["peak_risk"] for t in v[1]]) for c, v in cells.items()}
    emb = {c: sm.embed(v[0])[:, 0] for c, v in cells.items()}
    emb.update({c + " (phone)": sm.embed(cells[c][0], phone=True)[:, 0] for c in ("AI scam", "AI legit")})
    return cells, sem, emb


def evaluate():
    import joblib
    import fusion
    cells, sem, emb = test_cells()
    heads = {"shipped ASV19": joblib.load(sm.OUT.with_suffix(".pkl")), "ASV19 + ASV5": new_head()}

    def flag(c, sp, one_sided=False):
        # one_sided: POST-HOC variant added after the first run showed a low
        # spoof score dragging real (human) scams under the band. The spoof slot
        # only counts when it says "synthetic" (>= 0.5), else it abstains.
        # NOTE: fusion.fuse_voice itself is one-sided since 03a908c, so a rerun
        # now prints the same number for both; the two-sided ones are in RESULT.
        s = sem[c.replace(" (phone)", "")]
        return np.array([fusion.fuse_voice(a, spoof=b if not one_sided or b >= 0.5 else None)["risk"]
                         for a, b in zip(s, sp)]) >= 0.35

    from sklearn.metrics import roc_auc_score
    for tag, ai in (("studio", ("AI scam", "AI legit")), ("phone", ("AI scam (phone)", "AI legit (phone)"))):
        X = np.vstack([emb[c] for c in (*ai, "human scam", "human legit")])
        yy = np.r_[np.ones(200), np.zeros(200)]
        print(f"fake-voice AUC, AI vs human ({tag}): " + "   ".join(
            f"{n} {roc_auc_score(yy, h.predict_proba(X)[:, 1]):.3f}" for n, h in heads.items()))

    print(f"\n{'cell':<20} {'intent only':>11} | " + " | ".join(f"{h:^30}" for h in heads))
    print(f"{'':<20} {'flagged':>11} | " + " | ".join(f"{'fake>=.5':>9} {'mean':>6} {'flagged':>11}" for _ in heads))
    for c in emb:
        y = cells[c.replace(" (phone)", "")][2]
        out = [f"{c:<20} {(sem[c.replace(' (phone)', '')] >= 0.35).mean():>11.0%}"]
        for h in heads.values():
            sp = h.predict_proba(emb[c])[:, 1]
            out.append(f"{(sp >= 0.5).mean():>9.0%} {sp.mean():>6.2f} {flag(c, sp).mean():>11.0%}")
        print(" | ".join(out) + ("   <- want HIGH" if y else "   <- want LOW"), flush=True)

    for tag, ai in (("studio", ("AI scam", "AI legit")), ("phone", ("AI scam (phone)", "AI legit (phone)"))):
        cs = [ai[0], "human scam", ai[1], "human legit"]
        y = np.r_[np.ones(200), np.zeros(200)]
        acc = [f"intent only {np.mean(np.concatenate([sem[c.replace(' (phone)', '')] >= 0.35 for c in cs]) == y):.1%}"]
        for n, h in heads.items():
            for one in (False, True):
                f = np.concatenate([flag(c, h.predict_proba(emb[c])[:, 1], one) for c in cs])
                acc.append(f"+ {n}{' one-sided*' if one else ''} {np.mean(f == y):.1%}")
        print(f"\naccuracy, all 400 calls, AI calls {tag}:  " + "   ".join(acc))


def recalibrated(head, t):
    """Same head, offset shifted so its 0.5 sits where the raw score was t.
    Monotone: every EER and AUC is unchanged; only what 0.5 means moves."""
    import copy
    h = copy.deepcopy(head)
    h.steps[-1][1].intercept_ = h.steps[-1][1].intercept_ - np.log(t / (1 - t))
    return h


def calibrate():
    """Gap 2a. Rule fixed before any calibration score was seen: on the
    calibration set (104 fresh TTS dialogues, studio + telephony(); 50 HVB A
    bank-agent files + 50 scambait TRAIN-channel calls), pick the cut-off t that
    maximises (AI flagged - human flagged) for the ASV19+ASV5 head, shift the
    head so 0.5 = t, then compare against the shipped head on held-out data."""
    import joblib
    import act_model
    import fusion
    from sklearn.metrics import roc_curve
    cal = pd.read_csv(SD / "scam-dialogue_all.csv").loc[json.load(open(TTS_CAL / "picked.json"))]
    ai = [str(TTS_CAL / f"{i}.wav") for i in cal.index]
    assert all(os.path.exists(p) for p in ai), "run --tts --calib first"
    hvb = data.load_harpervalley()
    hvb_a = hvb[np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2]
    hvb_a = hvb_a.iloc[np.random.default_rng(1).permutation(len(hvb_a))[:50]]
    sb = data.load_scambait()
    sb = sb[~sb.channel.isin(act_model.SCAMBAIT_TEST_CHANNELS | act_model.SCAMBAIT_NEW_CHANNELS)]
    sb = sb[[(data.DATA / "vishing" / "scambait" / f"{i}.wav").exists() for i in sb.id]]
    sb = sb.iloc[np.random.default_rng(1).permutation(len(sb))[:50]]
    human = ([str(data.DATA / "vishing" / "harpervalley" / "data" / "audio" / "agent" / f"{i}.wav") for i in hvb_a.id]
             + [str(data.DATA / "vishing" / "scambait" / f"{i}.wav") for i in sb.id])
    new = new_head()
    Xc = np.vstack([sm.embed(ai)[:, 0], sm.embed(ai, phone=True)[:, 0], sm.embed(human)[:, 0]])
    yc = np.r_[np.ones(2 * len(ai)), np.zeros(len(human))]
    fpr, tpr, thr = roc_curve(yc, new.predict_proba(Xc)[:, 1])
    k = int(np.argmax(tpr - fpr))
    t = float(np.clip(thr[k], 1e-4, 1 - 1e-4))
    print(f"calibration: {2 * len(ai)} AI clips ({len(ai)} studio + phone), {len(human)} human; "
          f"cut-off {t:.3f} -> AI flagged {tpr[k]:.0%}, human flagged {fpr[k]:.0%}", flush=True)
    heads = {"shipped ASV19": joblib.load(sm.OUT.with_suffix(".pkl")), "ASV19+ASV5": new,
             "ASV19+ASV5 recal": recalibrated(new, t)}

    cells, sem, emb = test_cells()
    s21 = data.load_spoof21()
    r = np.random.default_rng(0)
    d21 = pd.concat([g.iloc[r.permutation(len(g))[:400]] for _, g in s21.groupby(["codec", "label"])])
    A21 = np.load(data.DATA / "vishing" / "LA" / f"emb_l{sm.LAYER}_a21_draw5600.npy").astype(np.float32)
    itw = data.DATA / "vishing" / "in_the_wild"
    yi = (pd.read_csv(itw / "release_in_the_wild" / "meta.csv").label == "spoof").values
    Iph = np.load(itw / f"emb_l{sm.LAYER}_phone.npy").astype(np.float32)
    ai_c = ["AI scam", "AI legit", "AI scam (phone)", "AI legit (phone)"]
    hu_c = ["human scam", "human legit"]
    print()
    print(f"held-out, flag = fake score >= 0.5     {'shipped ASV19':>14} {'ASV19+ASV5':>11} {'recal':>8}   want")
    rows = [
        ("2x2 AI voices flagged", lambda h: np.mean(np.concatenate([h.predict_proba(emb[c])[:, 1] for c in ai_c]) >= .5), "high"),
        ("2x2 real humans flagged", lambda h: np.mean(np.concatenate([h.predict_proba(emb[c])[:, 1] for c in hu_c]) >= .5), "low"),
        ("In-the-Wild fakes flagged (phone)", lambda h: np.mean(h.predict_proba(Iph[yi])[:, 1] >= .5), "high"),
        ("In-the-Wild real flagged (phone)", lambda h: np.mean(h.predict_proba(Iph[~yi])[:, 1] >= .5), "low"),
        ("ASVspoof 2021 fakes flagged", lambda h: np.mean(h.predict_proba(A21[d21.label.values == 1])[:, 1] >= .5), "high"),
        ("ASVspoof 2021 real flagged", lambda h: np.mean(h.predict_proba(A21[d21.label.values == 0])[:, 1] >= .5), "low"),
    ]
    for name, f, want in rows:
        print(f"  {name:<36} " + " ".join(f"{f(h):>{w}.1%}" for h, w in zip(heads.values(), (14, 11, 8))) + f"   {want}")
    for tag, ai2 in (("studio", ("AI scam", "AI legit")), ("phone", ("AI scam (phone)", "AI legit (phone)"))):
        cs, y = [ai2[0], "human scam", ai2[1], "human legit"], np.r_[np.ones(200), np.zeros(200)]
        acc = []
        for n, h in heads.items():
            fl = np.concatenate([np.array([fusion.fuse_voice(a, spoof=b)["risk"] for a, b in
                                           zip(sem[c.replace(" (phone)", "")], h.predict_proba(emb[c])[:, 1])]) >= .35
                                 for c in cs])
            acc.append(f"{n} {np.mean(fl == y):.1%}")
        print(f"  2x2 voice-verdict accuracy, AI {tag} (intent + spoof, one-sided): " + "   ".join(acc))
    print()
    print(f"EERs and AUCs are unchanged by the shift; the offset is log(t/(1-t)) = {np.log(t / (1 - t)):.3f}")


if __name__ == "__main__":
    if "--tts" in sys.argv:
        tts(calib="--calib" in sys.argv)
    elif "--calibrate" in sys.argv:
        calibrate()
    else:
        evaluate()

# RESULT 2026-10-25 (100 calls per cell; flagged = fused risk >= 0.35, no caller number)
#                      intent | shipped ASV19: fake>=.5 flagged | ASV19+ASV5: fake>=.5 flagged
#   AI scam             73%   |               69%      83%     |             15%      67%
#   AI legit             6%   |               74%      10%     |             21%       5%
#   human scam          85%   |               30%      81%     |              0%      77%
#   human legit          0%   |               43%       4%     |             12%       2%
#   (AI through telephony(): fake>=.5 shipped 79/84%, ASV19+ASV5 16/25%)
# Fake-voice AUC, AI vs human: shipped .737 studio / .787 phone; ASV19+ASV5 .809 / .825.
# Accuracy on all 400: intent only 88.0%; + shipped 87.5%; + ASV19+ASV5 84.2% (phone 86.8 / 84.8).
# POST-HOC one-sided spoof slot (counts only when >= 0.5): 88.5% / 87.5% - back to intent-only, no gain.
# Reading: (1) neither head reliably flags modern commercial neural TTS at 0.5; ASV19+ASV5 ranks it
# better (AUC) but scores it low. (2) In the weighted sum a LOW spoof score reads as "human, so safer"
# and pulls real human scams under the band (85 -> 77%). Most scams are human voices, so the slot
# should escalate, not acquit. The one-sided fix was found on this data: confirm on fresh data first.

# CALIBRATION RESULT 2026-10-26 (--calibrate). The fixed rule (max AI-flagged minus human-flagged on
# the calibration set) picked a cut-off at the floor (~0.0001, offset -9.21): AI 89% / human 35% there.
# Held-out, flag = score >= 0.5:      shipped ASV19   ASV19+ASV5   ASV19+ASV5 recal
#   2x2 AI voices flagged                 76.5%         19.2%          60.0%
#   2x2 real humans flagged               36.5%          6.0%          15.5%
#   In-the-Wild fakes flagged (phone)     63.5%         40.3%          75.0%
#   In-the-Wild real flagged (phone)      49.4%          3.9%          13.2%
#   ASVspoof 2021 fakes flagged           97.8%         95.2%          99.4%
#   ASVspoof 2021 real flagged             4.3%          6.4%          28.7%
#   2x2 verdict accuracy studio/phone   88.5/87.8     87.5/87.8      86.5/87.0
# Not adopted: the shift buys AI catch with 13-29% of real voices flagged. No single cut-off beats the
# shipped head on every row - the shipped head's catch comes from flagging ~half of real voices. In the
# one-sided fusion the slot should fire only when sure, so the unshifted ASV19+ASV5 head is the pick.
