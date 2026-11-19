"""Re-label FTC with the 7B judge + criteria prompt. v1 used 3B + bare names and
distilled to 0.772; the 7B judge itself reads 0.842, so this is the better
teacher. Same within-corpus trick: act-present and act-absent are both FTC."""
import json, time, numpy as np, data, llm_primitives as lp

KEYS = ["credential", "payment", "remote", "threat", "secrecy", "urgency", "prize"]
d = data.load_robocalls()
segs = [lp.chunks(t)[0] if lp.chunks(t) else "" for t in d.transcript]
# ponytail: checkpoint every block. The first run of this died with the session
# and lost 100% of its work - same lesson as _aug.py. Resumes where it stopped.
CKPT = "_label7b_partial.npy"
t0 = time.time(); out = np.zeros((len(segs), len(KEYS)), bool)
start = 0
if data.Path(CKPT).exists():
    prev = np.load(CKPT)
    done = int(prev[-1, 0])            # last row stores how many are finished
    out[:done] = prev[:-1][:done].astype(bool)
    start = done
    print(f"resuming from {done}/{len(segs)}", flush=True)
B = 120
for i in range(start, len(segs), B):
    out[i:i+B] = lp.detect(segs[i:i+B], bs=8, keys=KEYS, max_new=72, criteria=True)
    n = min(i+B, len(segs))
    stamp = np.zeros((1, len(KEYS)), np.int16); stamp[0, 0] = n
    np.save(CKPT, np.vstack([out.astype(np.int16), stamp]))
    print(f"  {n}/{len(segs)}  {(time.time()-t0)/60:.1f} min  "
          f"{dict(zip(KEYS, out[:n].mean(0).round(3)))}", flush=True)
json.dump({"id": d.id.tolist(), "keys": KEYS, "labels": out.astype(int).tolist()},
          open("Datasets/vishing/robocall/llm_acts.json", "w"))
print(f"\nlabelled in {(time.time()-t0)/60:.1f} min")
for j, k in enumerate(KEYS):
    print(f"  {k:<11} {out[:,j].sum():>5}  ({out[:,j].mean():.1%})")
