"""Confirmation run for the realmix spoof head - ON FRESH DATA, rule fixed first.

_spoof_realmix.py failed its pre-fixed 0.5 rule, and only AFTER that did a cut-off
of 0.925 (5% of the 100 calibration humans flagged) look good. A cut-off found
after a failure is exactly how this project has fooled itself before, so it is
tested once here, frozen, on data no spoof experiment has touched.

FIXED 2026-10-27 BEFORE ANY SCORE WAS SEEN:
  new head   Datasets/vishing/mlaad/head_realmix.pkl at cut-off CUT = 0.925
  deployed   models/spoof_head.pkl at 0.5 (fusion.fuse_voice counts the slot at >= 0.5)
  fresh fake FRESH_PER clips per held-out MLAAD generator (the 41 + Edge-TTS of
             _spoof_mlaad.split), drawn with default_rng(1) from files NOT in
             picked.json - same generators, never-seen clips; first 4 s, studio + phone
  fresh real EdAcc (Edinburgh International Accents of English: conversational
             video-call English, many accents incl. Indian; CC-BY-SA-4.0) - one
             test shard, N_EDACC clips >= 2 s, as recorded + phone; and the 105
             scambait calls from SCAMBAIT_NEW_CHANNELS (7 channels never used in any
             spoof run)
  PASS iff   (1) fresh unseen-tool fakes caught (phone) >= deployed + 20 pts
             (2) fresh Edge-TTS caught (phone) >= deployed + 20 pts
             (3) EVERY fresh real set (EdAcc, EdAcc phone, scambait new channels)
                 flagged <= deployed + 2 pts
Read once. Rates and counts only.

    python -m scratch._spoof_confirm --fetch
    python -m scratch._spoof_confirm
"""
import json
import os
import sys

import data   # before torch (Windows libiomp)
import numpy as np

import spoof_model as sm
from scratch import _spoof_mlaad as mm
from scratch import _spoof_realmix as rm

CUT = 0.925
FRESH_PER = 60
N_EDACC = 600
EDACC = ("edinburghcstr/edacc", "data/test-00004-of-00010-806407c9bc68112a.parquet")
FRESH = mm.ROOT / "confirm_fresh.json"


def fetch():
    import requests
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, get_token
    api, auth = HfApi(), {"Authorization": f"Bearer {get_token()}"}
    if not FRESH.exists():
        seen = set(json.load(open(mm.ROOT / "picked.json")))
        _, held = mm.split(mm.generators())
        fresh = []
        for g in held:
            f = sorted(x.path for x in api.list_repo_tree(mm.REPO, path_in_repo=f"fake/en/{g}", repo_type="dataset")
                       if x.path.endswith(".wav") and x.path not in seen)
            fresh += [f[i] for i in sorted(np.random.default_rng(1).permutation(len(f))[:FRESH_PER])]
        json.dump(fresh, open(FRESH, "w"))
    fresh = [f for f in json.load(open(FRESH)) if not (mm.ROOT / f).exists()]
    print(f"fresh MLAAD clips to fetch: {len(fresh)}", flush=True)

    def get(f):
        out = mm.ROOT / f
        out.parent.mkdir(parents=True, exist_ok=True)
        for k in range(6):
            r = requests.get(f"https://huggingface.co/datasets/{mm.REPO}/resolve/main/{urllib.parse.quote(f)}",
                             headers=auth, timeout=60)
            if r.ok:
                tmp = out.with_suffix(".part")
                tmp.write_bytes(r.content)
                os.replace(tmp, out)
                return
            import time
            time.sleep(30 if r.status_code == 429 else 2)
        print(f"  FAILED {f}", flush=True)

    with ThreadPoolExecutor(24) as ex:
        list(ex.map(get, fresh))

    raw = rm.RAW / "edacc.parquet"
    if not raw.exists():
        r = requests.get(f"https://huggingface.co/datasets/{EDACC[0]}/resolve/main/{EDACC[1]}", headers=auth, timeout=120)
        r.raise_for_status()
        raw.write_bytes(r.content)
    out = rm.ROOT / "edacc"
    if not (out.exists() and any(out.iterdir())):
        import pyarrow.parquet as pq
        out.mkdir(parents=True, exist_ok=True)
        tab = pq.read_table(raw, columns=["audio"]).column("audio").to_pylist()
        k = 0
        for i in sorted(np.random.default_rng(0).permutation(len(tab))):
            a = rm._decode(tab[i]["bytes"])
            if a is not None and len(a[0]) >= 2 * a[1]:
                rm._write(*a, out / f"{k}.wav")
                k += 1
            if k >= N_EDACC:
                break
        print(f"  edacc: {k} clips from {len(tab)} segments", flush=True)
    print("fetch done", flush=True)


def main():
    import joblib
    import act_model
    E = lambda name, paths, phone=False: mm.cached(f"confirm_{name}", paths, phone=phone)
    fresh = [f for f in json.load(open(FRESH)) if (mm.ROOT / f).exists()]
    edge = [str(mm.ROOT / f) for f in fresh if mm.family(f.split("/")[2]) == "edge"]
    tools = [str(mm.ROOT / f) for f in fresh if mm.family(f.split("/")[2]) != "edge"]
    sb = data.load_scambait()
    sbn = [str(data.DATA / "vishing" / "scambait" / f"{i}.wav")
           for i in sb[sb.channel.isin(act_model.SCAMBAIT_NEW_CHANNELS)].id]
    ed = rm.clips("edacc")
    print(f"fresh: {len(tools)} unseen-tool clips ({len({f.split('/')[2] for f in fresh}) - 1} tools) + "
          f"{len(edge)} Edge | real: {len(ed)} EdAcc, {len(sbn)} scambait new-channel calls", flush=True)
    S = {"fresh unseen tools, studio": E("tools_clean", tools),
         "fresh unseen tools, phone": E("tools_phone", tools, phone=True),
         "fresh Edge-TTS, studio": E("edge_clean", edge),
         "fresh Edge-TTS, phone": E("edge_phone", edge, phone=True),
         "EdAcc (as recorded)": E("edacc", ed),
         "EdAcc, phone": E("edacc_phone", ed, phone=True),
         "scambait new channels": E("sb_new", sbn)}
    dep = joblib.load(sm.OUT.with_suffix(".pkl"))
    new = joblib.load(mm.ROOT / "head_realmix.pkl")
    r = {k: (np.mean(dep.predict_proba(X)[:, 1] >= .5), np.mean(new.predict_proba(X)[:, 1] >= CUT),
             np.mean(new.predict_proba(X)[:, 1] >= .5)) for k, X in S.items()}
    print(f"\n{'':30} {'deployed@0.5':>13} {'new@0.925':>10} {'(new@0.5)':>10}")
    for k, (a, b, c) in r.items():
        print(f"  {k:28} {a:>13.1%} {b:>10.1%} {c:>10.1%}")
    d = {k: v[1] - v[0] for k, v in r.items()}
    ok = (d["fresh unseen tools, phone"] >= .20 and d["fresh Edge-TTS, phone"] >= .20
          and all(d[k] <= .02 for k in ("EdAcc (as recorded)", "EdAcc, phone", "scambait new channels")))
    print(f"\n-> {'PASS' if ok else 'FAIL'} by the rule fixed in the docstring")


if __name__ == "__main__":
    fetch() if "--fetch" in sys.argv else main()

# RESULT 2026-10-27 (scratch/_spoof_confirm.log), read once:
#                                deployed@0.5  new@0.925  (new@0.5)
#   fresh unseen tools, studio        19.7%      86.0%      92.9%
#   fresh unseen tools, phone         20.0%      78.9%      89.3%
#   fresh Edge-TTS, studio            25.0%      95.0%      96.7%
#   fresh Edge-TTS, phone             20.0%      93.3%      96.7%
#   EdAcc (as recorded)                1.9%       1.0%       4.0%
#   EdAcc, phone                       2.9%       2.1%       6.5%
#   scambait new channels              1.9%      10.5%      24.8%
# FAIL on (3): scambait new channels +8.6 pts (11 of 105 calls). Concentrated: Scam Sandwich
# 6/17, ScammerRevolts 3/17, three channels 1 each, three 0. Fakes confirmed (+59/+73 pts on
# fresh clips) and conversational EdAcc false alarms went DOWN - but real YouTube scam-call
# audio from some channels reads as synthetic, and that is the audio this app is for.
# Not promoted. Deployed head stays.
