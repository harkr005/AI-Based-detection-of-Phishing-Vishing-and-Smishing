import json, time, numpy as np, llm_primitives as lp, voice_model as vm
from sklearn.metrics import roc_auc_score
calls = json.load(open("Datasets/vishing/calls/transcripts.json"))
idx = {c["id"]: c for c in json.load(open("Datasets/vishing/calls/index.json"))}
ids = sorted(i for i in calls if len(calls[i]) > 200)
y = np.array([idx[i]["label"] for i in ids])

t0 = time.time(); rows = []
for n, i in enumerate(ids, 1):
    r, f = lp.call_risk(calls[i], criteria=True)
    rows.append((r, f))
    print(f"[{n}/{len(ids)}] y={idx[i]['label']} {r:.2f} {','.join(f) or '-':<32} "
          f"{idx[i]['title'][:38]}", flush=True)
cri = np.array([r for r, _ in rows])
print(f"\nscored in {(time.time()-t0)/60:.1f} min")

def rule_peak(t, n=40, st=20):
    w = t.split()
    ws = [" ".join(w[k:k+n]) for k in range(0, max(len(w)-n, 0)+1, st)] or [t]
    return max(vm.rule_score(x) for x in ws)
R = np.array([rule_peak(calls[i]) for i in ids])

dev = np.zeros(len(ids), bool); r2 = np.random.default_rng(7)
for lab in (0, 1):
    w = np.flatnonzero(y == lab); dev[r2.permutation(w)[:len(w)//2]] = True

def show(nm, v):
    print(f"  {nm:<34} dev {roc_auc_score(y[dev],v[dev]):.3f}  "
          f"test {roc_auc_score(y[~dev],v[~dev]):.3f}  all {roc_auc_score(y,v):.3f}")
print()
show("3B criteria acts alone", cri)
show("rules + 3B criteria acts", np.clip(R + cri, 0, 1))
print("  --- baselines on the same 27 calls ---")
print("  3B bare-name acts alone            dev 0.595  test 0.823  all 0.686")
print("  rules + 3B bare-name (clean acts)   dev 0.774  test 0.719  all 0.747")
print("  rules only                          dev 0.702  test 0.708  all 0.703")
print("  rules + DISTILLED acts (SHIPPED)    dev 0.786  test 0.760  all 0.772")
acts = [f for _, f in rows]
print("\n  per-act fire rate (criteria prompt):")
for k in lp.ORDER:
    s = np.mean([k in f for f, l in zip(acts, y) if l == 1])
    g = np.mean([k in f for f, l in zip(acts, y) if l == 0])
    print(f"    {k:<12} scam {s:>5.0%}   legit {g:>5.0%}")
json.dump({i: {"risk": r, "acts": f} for i, (r, f) in zip(ids, rows)},
          open("_cri_scores.json", "w"), indent=1)
