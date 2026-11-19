"""Distilled scam-ACT detector: DistilBERT, multi-label, one sigmoid per act.

The fast half of `llm_primitives`. The 3B judge scores a call in ~30 s, which
is useless against §6's 2-second cadence; this reproduces its labels in a
forward pass. It is the ACL-2026 recipe's shipping half - use the LLM to build
data, deploy something small.

WHY IT CANNOT REPEAT THE EARLIER FAILURES. Every previous attempt on this
channel learned provenance instead of intent, because the two classes came from
different corpora:
    sms_model      SMS spam  vs  SMS ham          -> banking words = alarm
    speech_model   FTC (ASR) vs  SGD (written)    -> "ASR transcript" = scam
    phrase mining  FTC       vs  SGD              -> "press to", "please press"
Here both classes are FTC robocalls. A transcript WITH a threat and one WITHOUT
are the same corpus, same recording conditions, same Whisper pass. There is no
provenance signal left to find, so the only thing separating them is the act.

`secrecy` is labelled but NOT modelled: it fires on 0% of robocalls, because a
recorded message never says "don't tell anyone" - that is a live-scammer
tactic. It has no positives here and stays a regex in voice_model, which is the
right tool for a fixed phrase anyway.
"""
from functools import lru_cache
from pathlib import Path

import data

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

CKPT = "distilbert-base-uncased"
OUT = Path(__file__).parent / "models" / "act_distilbert"
ACTS = ["threat", "prize"]   # v1 fallback; retrained models carry their list in config
MIN_POS = 20                 # an act with fewer positives than this cannot be distilled
MAX_LEN = 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LABELS_JSON = data.DATA / "vishing" / "robocall" / "llm_acts.json"
SPANS_JSON = data.DATA / "vishing" / "robocall" / "llm_spans.json"
SCAMBAIT_JSON = data.DATA / "vishing" / "scambait" / "llm_chunk_acts.json"
SCAMBAIT_SPANS = data.DATA / "vishing" / "scambait" / "llm_spans.json"   # _scambait_spans.py
# Whole channels held out of every --scambait training run, so the test is a
# different scam-baiter's calls, not other calls from the same videos. Fixed
# 2026-10-10 before any result was seen. Read by `_voice_llm_eval.py`.
SCAMBAIT_TEST_CHANNELS = {"Scammer Payback", "Scammer Jammer"}
# Fetched 2026-10-24 after every model here was trained; never judged, so
# scambait_rows() cannot reach them. A second, wider held-out catch-rate test.
SCAMBAIT_NEW_CHANNELS = {"Kitboga", "Jim Browning", "Scambaiter", "NanoBaiter",
                         "Scam Sandwich", "Deeveeaar", "ScammerRevolts"}


def cut(text, quotes):
    """Text with each quote removed. Case-insensitive, first occurrence only -
    a quote the locator returned twice is the same sentence found twice."""
    for q in quotes:
        i = text.lower().find(q.lower())
        if i >= 0:
            text = text[:i] + " " + text[i + len(q):]
    return " ".join(text.split())


def paste(text, quote, rng):
    """`quote` inserted into `text` at a random word boundary - cut()'s inverse."""
    w = text.split()
    i = int(rng.integers(len(w) + 1))
    return " ".join(w[:i] + [quote] + w[i:])


def injected(acts, test=False):
    """-> (texts, Y): one HarperValley bank call per located scambait span, with
    the span pasted in and labelled one-hot for its act.

    The mirror of background(). That removes the act from a scam call so "sounds
    like a scam call" stops being evidence; this puts a live scammer's act into a
    real bank call so "sounds like a bank call" stops being evidence of no act -
    the shortcut HVB-as-all-negatives invites, and exactly the call this project
    worries about most: a fake bank officer who sounds like a real one.

    test=False: train-channel spans into HVB half A (TRAIN only - half A is
    already the negatives). test=True: held-out-channel spans into HVB half B,
    for _voice_llm_eval's injection column. Neither side ever meets the other's
    calls or spans.
    """
    import json
    spans = json.load(open(SCAMBAIT_SPANS))
    d = data.load_scambait().set_index("id")
    hvb = data.load_harpervalley()
    half_a = np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2
    calls = hvb.transcript[~half_a if test else half_a].tolist()
    rng = np.random.default_rng(1 if test else 2)
    t, y = [], []
    for cid in sorted(spans):
        if (d.loc[cid].channel in SCAMBAIT_TEST_CHANNELS) != test:
            continue
        for per_act in spans[cid].values():
            for a, qs in per_act.items():
                if a not in acts:
                    continue
                for q in qs:
                    t.append(paste(calls[int(rng.integers(len(calls)))], q, rng))
                    row = np.zeros(len(acts), int)
                    row[acts.index(a)] = 1
                    y.append(row)
    return t, np.array(y).reshape(-1, len(acts))


def load_labelled(acts=None):
    """-> (texts, Y, acts). Acts default to those the judge actually fired on:
    a head for an act with no positives learns noise, and which acts survive
    depends on the judge, so it is read off the labels, not hardcoded."""
    import json
    j = json.load(open(LABELS_JSON))
    lab = np.array(j["labels"])
    acts = acts or [k for i, k in enumerate(j["keys"]) if lab[:, i].sum() >= MIN_POS]
    cols = [j["keys"].index(a) for a in acts]
    d = data.load_robocalls().set_index("id")
    return d.loc[j["id"]].transcript.tolist(), lab[:, cols], acts


def background(acts):
    """-> (texts, Y) for the other two ACL-2026 instance types, built from
    `_bg_spans.py`'s verbatim quotes. ADD THESE TO THE TRAINING HALF ONLY: every
    row here is derived from a specific labelled call, so mixing them into a test
    split scores the model on near-copies of what it trained on.

      type 2  BACKGROUND NEGATIVE - the transcript with its evidence CUT OUT,
              labelled all-zero. A scam call minus its scam act is a negative,
              and it is the same speaker, corpus and ASR pass as the positive it
              came from, so "sounds like a robocall" stops being evidence. This
              is the paper's explicit answer to a model fitting the background.
      type 3  LOCAL EVIDENCE - the quote alone, one-hot for its act. Teaches the
              act with none of the narrative around it.
    """
    import json
    d = data.load_robocalls().set_index("id")
    spans = json.load(open(SPANS_JSON))
    extra_t, extra_y = [], []
    for cid, per_act in spans.items():
        quotes = [q for a, qs in per_act.items() if a in acts for q in qs]
        if not quotes:
            continue
        back = cut(d.loc[cid].transcript, quotes)
        if len(back.split()) >= 10:      # a stub is not a background, it is noise
            extra_t.append(back)
            extra_y.append(np.zeros(len(acts), int))
        for a, qs in per_act.items():
            if a not in acts:
                continue
            for q in qs:
                y = np.zeros(len(acts), int)
                y[acts.index(a)] = 1
                extra_t.append(q)
                extra_y.append(y)
    return extra_t, np.array(extra_y)


def scambait_rows(acts):
    """-> (texts, Y): live scam calls, one row per judged chunk, plus HarperValley
    half A as all-zero rows. TRAINING ONLY.

    The robocalls stop before the ask, so credential / payment / remote barely
    exist in them; live calls are the ask. Chunk-level labels (not call-level)
    because a call runs ~10 chunks and an act sits in one or two of them - the
    others are the ACL background negatives, same speaker, same call.
    HVB half A gives the model real legitimate bank talk ("transfer", "card",
    "verify") labelled as no act; half B stays the false-alarm test.

    ponytail: a 250-word chunk is ~330 tokens and MAX_LEN is 256, so the tail of
    each chunk is cut and an act sitting there becomes a noisy label. Raising
    MAX_LEN changes how every saved model scores, so it waits for a measurement.
    """
    import json
    import llm_primitives as lp
    j = json.load(open(SCAMBAIT_JSON))
    cols = [j["keys"].index(a) for a in acts]
    d = data.load_scambait().set_index("id")
    t, y = [], []
    for cid, labs in j["calls"].items():
        if d.loc[cid].channel in SCAMBAIT_TEST_CHANNELS:
            continue
        segs = lp.chunks(d.loc[cid].transcript)
        assert len(segs) == len(labs), f"{cid}: transcript changed since it was judged"
        t += segs
        y += [np.array(row)[cols] for row in labs]
    hvb = data.load_harpervalley()
    half_a = np.random.default_rng(0).permutation(len(hvb)) < len(hvb) // 2
    for text in hvb.transcript[half_a]:
        for s in lp.chunks(text):
            t.append(s)
            y.append(np.zeros(len(acts), int))
    return t, np.array(y)


def encode(tok, texts, Y=None):
    e = tok(list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt")
    cols = [e["input_ids"], e["attention_mask"]]
    if Y is not None:
        cols.append(torch.tensor(np.asarray(Y), dtype=torch.float))
    return TensorDataset(*cols)


def train(texts, Y, acts, epochs=3, bs=16, lr=3e-5):
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, num_labels=len(acts), problem_type="multi_label_classification",
        id2label=dict(enumerate(acts)),
        label2id={a: i for i, a in enumerate(acts)}).to(DEV)
    loader = DataLoader(encode(tok, texts, Y), batch_size=bs, shuffle=True)
    # acts are rare (threat ~20%, prize ~6%) - weight the positives up or the
    # model learns to answer "no" and score a fine-looking loss
    pos = np.asarray(Y).mean(0).clip(1e-3, 1 - 1e-3)
    pw = torch.tensor((1 - pos) / pos, dtype=torch.float, device=DEV)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(.1 * steps), steps)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for ids, mask, yb in loader:
            opt.zero_grad()
            out = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
            loss = loss_fn(out.logits, yb.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep+1}  loss {tot/len(loader):.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    return tok, model


def predict(tok, model, texts, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for ids, mask in DataLoader(encode(tok, texts), batch_size=bs):
            logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


@lru_cache(maxsize=1)
def load():
    tok = AutoTokenizer.from_pretrained(OUT)
    return tok, AutoModelForSequenceClassification.from_pretrained(OUT).to(DEV).eval()


def acts_of(model):
    """The acts this saved model was trained on. v1 predates id2label and gets
    the fallback, so reverting to it stays a directory copy."""
    n = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return ACTS if n[0].startswith("LABEL_") else n


def score(texts):
    """-> (n, n_acts) act probabilities, columns in acts_of(model) order."""
    return predict(*load(), list(texts))


def fired_risk(acts, p):
    """One row of act probabilities -> (risk, acts fired), via llm_primitives'
    weights and combination rule - the same scoring as the judge it copies."""
    import llm_primitives as lp
    return lp.act_risk(a for a, v in zip(acts, p) if v >= 0.5)


def risk(text):
    """-> (risk, acts fired) for one text."""
    return fired_risk(acts_of(load()[1]), score([text])[0])


if __name__ == "__main__":
    import json
    import sys

    if {"--7b", "--union"} & set(sys.argv):
        # --union adds `_union_labels.py`: the same corpus with an act firing if
        # EITHER judge saw it. It exists because the 7B prompt collapses prize
        # from 45 calls to 2, and prize (0.35) is exactly SUSPICIOUS while
        # urgency (0.30) alone is below the band - so 7B-only labels trade
        # catches for a PHISHING band it can finally reach.
        tag = "union" if "--union" in sys.argv else "7b"
        # The shipped model distils the 3B judge's three acts (threat, secrecy,
        # prize). `_label7b.py` relabelled the same corpus with the 7B judge and
        # the criteria prompt, all seven acts, and that file has been sitting
        # unused: urgency 194 positives, and threat + urgency = 0.75 is the
        # first act pair that crosses §3's 0.70 PHISHING band. prize falls to 2
        # positives under the stricter prompt and drops out - that is the trade.
        LABELS_JSON = data.DATA / "vishing" / "robocall" / f"llm_acts_{tag}.json"
        SPANS_JSON = data.DATA / "vishing" / "robocall" / f"llm_spans_{tag}.json"
        OUT = OUT.with_name(f"act_distilbert_{tag}")

    if "--scambait" in sys.argv:
        # Every act the judge knows, filtered AFTER merging: credential, payment
        # and remote miss MIN_POS on robocalls alone, which is the whole point.
        OUT = OUT.with_name(OUT.name + "_scambait")
        texts, Y, acts = load_labelled(json.load(open(LABELS_JSON))["keys"])
        sb_t, sb_y = scambait_rows(acts)
        keep = [j for j in range(len(acts)) if Y[:, j].sum() + sb_y[:, j].sum() >= MIN_POS]
        acts, Y, sb_y = [acts[j] for j in keep], Y[:, keep], sb_y[:, keep]
        print(f"+ {len(sb_t):,} scambait train-channel chunks + HVB half A, TRAIN only   " +
              "  ".join(f"{a} {sb_y[:, j].sum()}" for j, a in enumerate(acts)))
    else:
        texts, Y, acts = load_labelled()
        sb_t, sb_y = [], np.zeros((0, len(acts)), int)
    print(f"{len(texts):,} labelled robocalls   " +
          "  ".join(f"{a} {Y[:,j].sum()} ({Y[:,j].mean():.1%})"
                    for j, a in enumerate(acts)))
    print(f"distilling {acts}  (acts with >= {MIN_POS} positives)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(texts))
    split = int(0.85 * len(texts))       # not `cut` - that is the span remover
    tr, te = perm[:split], perm[split:]
    xt, yt = [texts[i] for i in tr] + sb_t, np.vstack([Y[tr], sb_y])
    if "--bg" in sys.argv:
        OUT = OUT.with_name(OUT.name + "_bg")
        bt, by = background(acts)
        print(f"+ {len(bt)} ACL rows (background negatives + local evidence), "
              f"TRAIN half only -> {OUT.name}")
        xt, yt = xt + bt, np.vstack([yt, by])
    if "--inject" in sys.argv:
        OUT = OUT.with_name(OUT.name + "_inj")
        it, iy = injected(acts)
        print(f"+ {len(it)} bank calls with a train-channel scam span pasted in, TRAIN only   " +
              "  ".join(f"{a} {iy[:, j].sum()}" for j, a in enumerate(acts)))
        xt, yt = xt + it, np.vstack([yt, iy])
    tok, model = train(xt, yt, acts)

    p = predict(tok, model, [texts[i] for i in te])
    print("\ndoes it reproduce the 3B judge on held-out robocalls?")
    for j, a in enumerate(acts):
        n = int(Y[te, j].sum())
        auc = roc_auc_score(Y[te, j], p[:, j]) if 0 < n < len(te) else float("nan")
        print(f"  {a:<8} AUC {auc:.3f}   (n={len(te)}, {n} positives"
              + ("  - too few to trust the third decimal)" if n < 15 else ")"))

    # ---- End to end on the 27 real calls, never seen by anything here.
    idx = data.DATA / "vishing" / "calls" / "index.json"
    cache = data.DATA / "vishing" / "calls" / "transcripts.json"
    if cache.exists():
        import voice_model as vm
        meta = {c["id"]: c for c in json.load(open(idx))}
        txts = json.load(open(cache))
        ids = sorted(i for i in txts if len(txts[i]) > 200)
        y = np.array([meta[i]["label"] for i in ids])

        def chunks(t, n=250):
            w = t.split()
            return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [t]

        act = np.array([fired_risk(acts, predict(tok, model, chunks(txts[i])).max(0))[0]
                        for i in ids])

        def rule_peak(t, n=40, st=20):
            w = t.split()
            ws = [" ".join(w[k:k + n]) for k in range(0, max(len(w) - n, 0) + 1, st)] or [t]
            return max(vm.rule_score(x) for x in ws)
        rules = np.array([rule_peak(txts[i]) for i in ids])

        dev = np.zeros(len(ids), bool)
        r2 = np.random.default_rng(7)
        for lab in (0, 1):
            w = np.flatnonzero(y == lab)
            dev[r2.permutation(w)[:len(w) // 2]] = True

        print(f"\n27 real calls, held out from all training  (n={len(ids)})")
        for nm, v in [("distilled acts", act),
                      ("rules only", rules),
                      ("rules + distilled acts", np.clip(rules + act, 0, 1))]:
            print(f"  {nm:<24} dev {roc_auc_score(y[dev],v[dev]):.3f}  "
                  f"test {roc_auc_score(y[~dev],v[~dev]):.3f}  "
                  f"all {roc_auc_score(y,v):.3f}")
        print("  for reference: rules + distilled 3B acts (v1)  dev 0.786  test 0.760  all 0.772")
        print("                 rules + 7B judge, criteria       dev 0.810  test 0.865  all 0.842")
        print("                 sms_model, what this replaces    dev 0.262  test 0.438  all 0.344")
        print()
        print("  REVERT IF dev < 0.786: copy models/act_distilbert_v1_3b "
              "over models/act_distilbert")
