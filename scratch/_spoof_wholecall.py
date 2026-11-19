"""Whole-call spoof scoring: does reading the whole call beat the first 4 s?

The shipped detector (spoof_model.score, as voice_model.analyse calls it) reads
only the first CROP = 4 s of an upload - a greeting or a ringtone. This scores
every 4 s slice across the call (non-overlapping, at most MAX_CROPS evenly
spaced, near-silent slices dropped) with the SAME shipped head and combines them.

RULE, fixed before any score was seen (2026-10-26):
  - the combination (mean / median / max / share of slices >= 0.5) is CHOSEN on
    the calibration set only: the 104 tts_calib AI calls (studio + telephony())
    vs the same 100 calibration humans as _deepfake_intent_eval.calibrate()
    (50 HVB A agents + 50 scambait train-channel calls), by the largest
    (AI flagged - human flagged) at a cut-off of 0.5. The cut-off never moves.
  - it is then READ ONCE on the held-out 2 x 2 voices (200 Scam Dialogue TTS
    calls, studio + phone; 100 HVB B + 100 scambait test-channel humans).
  - ADOPT only if, held-out, AI flagged rises >= 10 points over first-4-s AND
    human flagged rises <= 2 points.
Caveats: stock Microsoft Edge voices only; the human rows are two corpora
(bank agents, YouTube scam-baiting). In-the-Wild / ASVspoof clips run 1-10 s,
i.e. 1-2 slices, so those published numbers are essentially unaffected.
Rates and counts only.

    python -m scratch._spoof_wholecall
"""
import json
import os

import data   # before torch (Windows libiomp)
import numpy as np
import pandas as pd
import torch

import act_model
import spoof_model as sm

MAX_CROPS = 16          # bounds the cost on 10-minute scambait calls
SILENT = 0.1            # a slice under 10% of the loudest slice's RMS is dropped
V = data.DATA / "vishing"
SD = V / "scam_dialogue"


def wav(path, phone=False):
    """Whole file -> float32 16 kHz mono (spoof_model.load_audio without the crop)."""
    x, sr = sm._read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != sm.SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), sm.SR)
        x = resample_poly(x, sm.SR // g, int(sr) // g).astype(np.float32)
    return sm.telephony(x) if phone else x


def slices(x):
    n = int(sm.CROP * sm.SR)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    c = np.stack([x[i:i + n] for i in range(0, len(x) - n + 1, n)])
    if len(c) > MAX_CROPS:
        c = c[np.linspace(0, len(c) - 1, MAX_CROPS).round().astype(int)]
    rms = np.sqrt((c ** 2).mean(1))
    return c[rms >= SILENT * rms.max()]


def slice_scores(paths, head, phone=False, bs=8):
    """-> list of per-slice spoof probabilities, one array per file."""
    m = sm._frontend()
    out = []
    for p in paths:
        c = slices(wav(p, phone))
        f = []
        for i in range(0, len(c), bs):
            with torch.no_grad():
                h = m(torch.from_numpy(c[i:i + bs]).to(sm.DEV).half(),
                      output_hidden_states=True).hidden_states[sm.LAYER]
            f.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy())
        out.append(head.predict_proba(np.concatenate(f))[:, 1])
    return out


COMBINE = {"mean": np.mean, "median": np.median, "max": np.max,
           "share>=.5": lambda s: float(np.mean(s >= 0.5))}


def flagged(per_file, how):
    return float(np.mean([COMBINE[how](s) >= 0.5 for s in per_file]))


def calib_sets():
    cal = pd.read_csv(SD / "scam-dialogue_all.csv").loc[json.load(open(SD / "tts_calib" / "picked.json"))]
    ai = [str(SD / "tts_calib" / f"{i}.wav") for i in cal.index]
    hvb = data.load_harpervalley()
    hvb_a = hvb[np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2]
    hvb_a = hvb_a.iloc[np.random.default_rng(1).permutation(len(hvb_a))[:50]]
    sb = data.load_scambait()
    sb = sb[~sb.channel.isin(act_model.SCAMBAIT_TEST_CHANNELS | act_model.SCAMBAIT_NEW_CHANNELS)]
    sb = sb[[(V / "scambait" / f"{i}.wav").exists() for i in sb.id]]
    sb = sb.iloc[np.random.default_rng(1).permutation(len(sb))[:50]]
    human = ([str(V / "harpervalley" / "data" / "audio" / "agent" / f"{i}.wav") for i in hvb_a.id]
             + [str(V / "scambait" / f"{i}.wav") for i in sb.id])
    return ai, human


def test_sets():
    """The held-out 2 x 2 voices, selected exactly as _deepfake_intent_eval.test_cells()."""
    d = pd.read_csv(SD / "scam-dialogue_all.csv").loc[json.load(open(SD / "tts" / "picked.json"))]
    ai = [str(SD / "tts" / f"{i}.wav") for i in d.index]
    r = np.random.default_rng(0)
    sb = data.load_scambait()
    sb = sb[sb.channel.isin(act_model.SCAMBAIT_TEST_CHANNELS)]
    sb = sb.iloc[r.permutation(len(sb))[:100]]
    hvb = data.load_harpervalley()
    hvb_b = hvb[~(np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2)]
    hvb_b = hvb_b.iloc[np.random.default_rng(0).permutation(len(hvb_b))[:100]]
    human = ([str(V / "scambait" / f"{i}.wav") for i in sb.id]
             + [str(V / "harpervalley" / "data" / "audio" / "agent" / f"{i}.wav") for i in hvb_b.id])
    return ai, human


def run(ai, human, head):
    s = {"AI": slice_scores(ai, head) + slice_scores(ai, head, phone=True),
         "human": slice_scores(human, head)}
    # first-4-s = what ships: spoof_model.score on the file (no silence gate)
    first = {"AI": np.r_[sm.score(ai, head), head.predict_proba(sm.embed(ai, phone=True)[:, 0])[:, 1]],
             "human": sm.score(human, head)}
    return s, first


if __name__ == "__main__":
    import joblib
    from sklearn.metrics import roc_auc_score
    head = joblib.load(sm.OUT.with_suffix(".pkl"))

    def table(s, first, tag):
        print(f"\n{tag}: {len(s['AI'])} AI clips (studio + phone), {len(s['human'])} human; "
              f"median slices/file AI {int(np.median([len(x) for x in s['AI']]))}, "
              f"human {int(np.median([len(x) for x in s['human']]))}")
        y = np.r_[np.ones(len(s["AI"])), np.zeros(len(s["human"]))]
        print(f"  {'method':12} {'AI flagged':>10} {'human flagged':>14} {'AUC':>6}")
        rows = {"first 4 s": (np.mean(first["AI"] >= .5), np.mean(first["human"] >= .5),
                              roc_auc_score(y, np.r_[first["AI"], first["human"]]))}
        for how, f in COMBINE.items():
            rows[how] = (flagged(s["AI"], how), flagged(s["human"], how),
                         roc_auc_score(y, [f(x) for x in s["AI"] + s["human"]]))
        for k, (a, h, auc) in rows.items():
            print(f"  {k:12} {a:>10.1%} {h:>14.1%} {auc:>6.3f}")
        return rows

    cal = table(*run(*calib_sets(), head), "CALIBRATION (choose here)")
    pick = max(COMBINE, key=lambda k: cal[k][0] - cal[k][1])
    print(f"\n-> chosen on calibration: {pick}")
    test = table(*run(*test_sets(), head), "HELD-OUT 2 x 2 (read once)")
    da = test[pick][0] - test["first 4 s"][0]
    dh = test[pick][1] - test["first 4 s"][1]
    print(f"\n{pick} vs first 4 s, held-out: AI flagged {da:+.1%}, human flagged {dh:+.1%} "
          f"-> {'ADOPT' if da >= .10 and dh <= .02 else 'DO NOT ADOPT'} (rule: AI >= +10 pts, human <= +2 pts)")

# RESULT 2026-10-26 (scratch/_spoof_wholecall.log), shipped ASV19+ASV5 head, cut-off 0.5:
#                       calibration (208 AI / 100 human)   held-out (400 AI / 200 human)
#   method              AI flag  human flag   AUC          AI flag  human flag   AUC
#   first 4 s (ships)    17.8%      5.0%     .826           19.2%      6.0%     .817
#   mean                  2.9%      1.0%     .707            3.0%      0.0%     .753
#   median                3.4%      1.0%     .836            2.8%      0.0%     .839
#   max (chosen)         63.5%     38.0%     .685           67.5%     32.5%     .745
#   share >= .5           3.8%      1.0%     .649            3.2%      0.0%     .701
# DO NOT ADOPT: max catches +48 pts of AI calls but flags +26.5 pts of real people.
# Reading: the rest of an Edge TTS call scores as HUMAN - averaging pulls AI calls
# down to ~3%. Whatever the head reacts to is in the opening seconds, not spread
# through the voice; reading more of the call does not help. First 4 s stays.
