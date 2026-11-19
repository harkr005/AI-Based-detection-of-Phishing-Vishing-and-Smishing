"""The measurements the guide's gap analysis asked for, on the SHIPPED models'
own splits. Retrains no shipped model; baselines are fitted fresh on the same
training rows the shipped model saw.

  --sms    char-LR (shipped) vs word-ngram LR, char LinearSVC, word NB, on
           sms_model.__main__'s split (GroupShuffleSplit rs=0 + Dataset_5971 held out)
  --url    shipped XGBoost on the OpenPhish+Tranco holdout: PR-AUC, FPR at 0.5,
           recall at 1% FPR, the all-benign baseline, and a logistic regression
           on the same 15 features trained on the same PhiUSIIL rows
  --email  v6 (shipped) vs TF-IDF LR / LinearSVC / NB trained on v6's exact rows
           (needs `python -m scratch._train_unified --daily --enron 1500
           --social 1500 --out email_distilbert_v6 --dump Datasets/paper/v6`)
  --ci     bootstrap 95% CIs (1,000 resamples, seed 0) from the saved predictions,
           Wilson intervals for the voice/spoof proportions, bootstrap EER CIs

Per-row predictions go to Datasets/paper/preds/ (private-derived: never commit).
Rates and counts only.
    python -m scratch._paper_extra --sms --url --email --ci
"""
import sys
import time

import data   # before torch (Windows libiomp)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

PAPER = data.DATA / "paper"
PRED = PAPER / "preds"
PRED.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


def baselines():
    return {
        "word 1-2gram TF-IDF + LR": make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, C=10)),
        "char 3-5gram TF-IDF + linear SVM": make_pipeline(
            TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3), LinearSVC(C=1)),
        "word TF-IDF + naive Bayes": make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2), MultinomialNB(alpha=0.1)),
    }


def sms():
    import adversarial
    import sms_model
    s = data.load_sms()
    held, pool = s[s.source == "Dataset_5971"], s[s.source != "Dataset_5971"]
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0).split(pool, groups=pool.group))
    tr, te = pool.iloc[i_tr], pool.iloc[i_te]
    tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)
    L, k = sms_model.LABELS, sms_model.LABELS.index("smishing")
    clean = lambda d: sms_model._clean(d.text.fillna(""))
    y_te, y_cs = te.label_3.map(L.index).values, held.label_3.map(L.index).values
    models = {"char 3-5gram TF-IDF + LR (shipped)": sms_model.load(), **baselines()}
    print(f"\n=== SMS  train {len(tr):,}  in-source {len(te):,}  cross-source {len(held):,}")
    print(f"{'model':36} {'in F1':>6} {'cross F1':>8} {'cross smish P':>13} {'cross smish R':>13}")
    for name, m in models.items():
        if "shipped" not in name:
            m.fit(clean(tr), tr.label_3.map(L.index).values)
        p_in, p_cs = m.predict(clean(te)), m.predict(clean(held))
        print(f"{name:36} {f1_score(y_te, p_in, average='macro'):6.3f} {f1_score(y_cs, p_cs, average='macro'):8.3f} "
              f"{precision_score(y_cs == k, p_cs == k):13.3f} {recall_score(y_cs == k, p_cs == k):13.3f}", flush=True)
        if "shipped" in name:
            np.savez(PRED / "sms.npz", y_in=y_te, p_in=p_in, y_cs=y_cs, p_cs=p_cs,
                     prob_cs=m.predict_proba(clean(held))[:, k])


def url():
    import url_model
    feed = data.load_url_feed()
    y, X = feed.label.values, url_model.featurize(feed.url)
    p = url_model.load().predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    d = data.load_urls()
    lr = make_pipeline(__import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
                       LogisticRegression(max_iter=2000)).fit(url_model.featurize(d.url), d.label.values)
    q = lr.predict_proba(X)[:, 1]
    print(f"\n=== URL  holdout n={len(y):,} ({y.sum()} phishing, base rate {y.mean():.2%}); train n={len(d):,}")
    for name, s in [("XGBoost (shipped)", p), ("logistic regression, same features", q)]:
        f, t, _ = roc_curve(y, s)
        print(f"  {name:36} ROC-AUC {roc_auc_score(y, s):.3f}  PR-AUC {average_precision_score(y, s):.3f}  "
              f"recall@0.5 {np.mean(s[y == 1] >= .5):.3f}  FPR@0.5 {np.mean(s[y == 0] >= .5):.3f}  "
              f"recall@1%FPR {t[np.searchsorted(f, .01, side='right') - 1]:.3f}")
    print(f"  {'all-benign baseline':36} accuracy {1 - y.mean():.3f}, recall 0, PR-AUC = base rate {y.mean():.3f}")
    print(f"  shipped accuracy@0.5 {np.mean((p >= .5) == y):.3f}")
    np.savez(PRED / "url.npz", y=y, p=p, q=q)


def email():
    import email_model as em
    D = PAPER / "v6"
    if not (D / "train.parquet").exists():
        sys.exit("run the --dump step first (see docstring)")
    S = {n: pd.read_parquet(D / f"{n}.parquet") for n in ("train", "temporal", "cross_source", "modern")}
    L, ph = em.LABELS, em.LABELS.index("phishing")
    y = {n: S[n].label_3.map(L.index).values for n in S}
    print(f"\n=== EMAIL  train {len(S['train']):,} (v6 rows) | temporal {len(S['temporal']):,} | "
          f"cross-source {len(S['cross_source']):,} | modern {len(S['modern']):,}")
    print(f"{'model':36} {'temp F1':>7} {'temp phish P/R':>15} {'cross F1':>8} {'modern R':>8} {'modern FP':>9}")

    def row(name, pred):
        t, c, m = pred["temporal"], pred["cross_source"], pred["modern"]
        ham = y["modern"] != ph
        print(f"{name:36} {f1_score(y['temporal'], t, average='macro', labels=sorted(set(y['temporal']))):7.3f} "
              f"{precision_score(y['temporal'] == ph, t == ph):7.3f}/{recall_score(y['temporal'] == ph, t == ph):.3f} "
              f"{f1_score(y['cross_source'], c, average='macro', labels=sorted(set(y['cross_source']))):8.3f} "
              f"{np.mean(m[~ham] == ph):8.1%} {np.mean(m[ham] == ph):9.1%}", flush=True)

    t0 = time.time()
    P = {n: em.score(S[n].text.tolist())[0] for n in ("temporal", "cross_source", "modern")}
    print(f"  (DistilBERT scored {sum(len(S[n]) for n in P):,} mails in {time.time() - t0:.0f}s)")
    row("DistilBERT v6 (shipped)", {n: v.argmax(1) for n, v in P.items()})
    np.savez(PRED / "email.npz", **{f"y_{n}": y[n] for n in P}, **{f"p_{n}": P[n] for n in P})
    for name, m in baselines().items():
        m.fit(S["train"].text.values, y["train"])
        row(name, {n: m.predict(S[n].text.values) for n in P})


def boot(f, *arrays, n=1000):
    idx = np.arange(len(arrays[0]))
    vals = [f(*(a[i] for a in arrays)) for i in (RNG.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def ci():
    import joblib
    import spoof_model as sm
    from scratch import _spoof_mlaad as mm
    show = lambda name, v, lo_hi: print(f"  {name:52} {v:.3f}  [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]")
    print("\n=== 95% CIs (bootstrap 1,000x unless marked Wilson)")
    s = np.load(PRED / "sms.npz")
    k = 2
    show("SMS cross-source macro-F1", f1_score(s["y_cs"], s["p_cs"], average="macro"),
         boot(lambda a, b: f1_score(a, b, average="macro"), s["y_cs"], s["p_cs"]))
    show("SMS cross-source smishing recall", recall_score(s["y_cs"] == k, s["p_cs"] == k),
         boot(lambda a, b: recall_score(a == k, b == k), s["y_cs"], s["p_cs"]))
    u = np.load(PRED / "url.npz")
    show("URL ROC-AUC", roc_auc_score(u["y"], u["p"]), boot(roc_auc_score, u["y"], u["p"]))
    show("URL PR-AUC", average_precision_score(u["y"], u["p"]), boot(average_precision_score, u["y"], u["p"]))
    if (PRED / "email.npz").exists():
        e = np.load(PRED / "email.npz")
        yt, pt = e["y_temporal"] == 2, e["p_temporal"].argmax(1) == 2
        show("email temporal phishing precision", precision_score(yt, pt), boot(precision_score, yt, pt))
        show("email temporal phishing recall", recall_score(yt, pt), boot(recall_score, yt, pt))
        ym, pm = e["y_modern"] == 2, e["p_modern"].argmax(1) == 2
        show("email modern 2025-26 phishing recall", pm[ym].mean(), boot(lambda a: a.mean(), pm[ym]))
        show("email modern 2025-26 ham false-alarm rate", pm[~ym].mean(), boot(lambda a: a.mean(), pm[~ym]))
    for name, k_, n in [("voice: 7 unseen scam-bait channels caught (Wilson)", 101, 105),
                        ("voice: real bank calls false alarms (Wilson)", 2, 723),
                        ("spoof: real scam-bait calls flagged as AI (Wilson)", 7, 105),
                        ("attachments: phishing not SAFE, 615 (Wilson)", 577, 615)]:
        show(name, k_ / n, wilson(k_, n))
    d21, A21, e5, E5, yi, I = mm.gates()
    h = joblib.load(sm.OUT.with_suffix(".pkl"))
    P = lambda X: h.predict_proba(X)[:, 1]
    pst = (d21.codec == "pstn").values
    for name, yy, pp in [("spoof EER, In-the-Wild phone", yi, P(I["phone"])),
                         ("spoof EER, ASV21 pstn", d21.label.values[pst], P(A21)[pst]),
                         ("spoof EER, ASVspoof 5 eval", e5.label.values, P(E5))]:
        show(name, sm.eer(yy, pp), boot(sm.eer, np.asarray(yy), pp, n=300))


if __name__ == "__main__":
    for flag, f in [("--sms", sms), ("--url", url), ("--email", email), ("--ci", ci)]:
        if flag in sys.argv:
            f()
