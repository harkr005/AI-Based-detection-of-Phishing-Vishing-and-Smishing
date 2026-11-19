"""Spoken-register scam head: DistilBERT, binary (legit / scam) on call speech.

WHY THIS EXISTS. The voice channel used to score speech with `sms_model`, the
SMS smishing model. On the 50 authored transcripts that looks fine (AUC 0.834).
On 27 real recorded calls it collapses to **AUC 0.433, worse than chance**: a
legitimate bank customer-service call scores 0.994 while a real scam call
scores 0.000.

The reason is in the training data, not the architecture. In SMS, an unsolicited
"verify your account" IS smishing, because banks do not hold SMS conversations -
so the SMS model learned that transactional banking vocabulary is itself the
alarm. On a phone call that vocabulary is what a LEGITIMATE agent says. The
model had never once seen a legitimate transactional conversation, so it had no
way to learn that one exists.

This head fixes the data gap: real scam call scripts as positives, legitimate
service and banking conversations as negatives.

THE SHORTCUT THIS DESIGN HAS TO AVOID. Positives are Whisper transcripts of real
audio (disfluent, unpunctuated, mis-heard); negatives are crowd-written text
(clean, punctuated). A model can separate those on STYLE and learn nothing about
scams - the same trap as the PhiUSIIL HTTPS artifact and the CEAS link flag.
Two things guard against it:
  1. `normalise()` strips the cheapest style cues (case, punctuation) from both
     sides before the tokenizer sees them.
  2. The honest test set cannot be gamed this way. All 27 real calls are Whisper
     output, both classes, so a style-shortcut model scores them identically and
     lands at AUC ~0.5. Separation there has to come from content.
Read the cross-source number in __main__, never the in-source one.
"""
from functools import lru_cache
from pathlib import Path

# data (pandas) before torch - see sms_model for the libiomp segfault.
import data

import adversarial
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

CKPT = "distilbert-base-uncased"
OUT = Path(__file__).parent / "models" / "speech_distilbert"
LABELS = ["legit", "scam"]
WINDOW_WORDS = 40      # ~15 s of phone speech at 2.5 wps, matching voice_model.CONTEXT
MAX_LEN = 96           # 40 words -> ~60 word-pieces; 96 covers the tail
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def normalise(t):
    """Lowercase and drop punctuation, so written negatives and ASR positives
    cannot be told apart by their commas. §9.9's unicode normalisation first."""
    import re
    t = adversarial.normalize(t).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t)).strip()


def windows(text, n=WINDOW_WORDS, stride=None):
    """-> list of n-word windows. Training uses stride n (no overlap, no near
    duplicates across the train/test line); serving overlaps, but that is
    voice_model's business, not this file's."""
    w = text.split()
    stride = stride or n
    if len(w) <= n:
        return [" ".join(w)] if w else []
    return [" ".join(w[i:i + n]) for i in range(0, len(w) - n + 1, stride)]


def build(neg_per_pos=2.0, seed=0):
    """-> DataFrame of windows: text, label, src.

    Positives are the 1,353 FTC robocalls - real scam scripts that actually
    defrauded people. Negatives are SGD, financial and service domains first
    because those are the ones whose words overlap a scam, then a sample of the
    rest for breadth.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    # Positives overlap (stride n/2) because there are only 1,353 of them and
    # most are one window long. Negatives do not - there is no shortage.
    pos = data.load_robocalls()
    pw = [{"text": w, "label": 1, "src": "ftc", "doc": f"ftc/{i}"}
          for i, t in enumerate(pos.transcript)
          for w in windows(t, stride=WINDOW_WORDS // 2)]

    sgd = data.load_sgd()
    fin = sgd.services.str.split(",").apply(
        lambda ss: any(x in data.SGD_FINANCIAL for x in ss))
    def wins(sub_df, src):
        return [{"text": w, "label": 0, "src": src, "doc": f"{src}/{i}"}
                for i, t in zip(sub_df.index, sub_df.text) for w in windows(t)]
    budget = int(len(pw) * neg_per_pos)
    nw, other = wins(sgd[fin], "sgd_fin"), wins(sgd[~fin], "sgd_other")
    # financial negatives first: they are the ones that share a scam's words
    nw = [nw[i] for i in rng.permutation(len(nw))[:budget]]
    other = [other[i] for i in rng.permutation(len(other))[:max(budget - len(nw), 0)]]
    d = pd.DataFrame(pw + nw + other)
    return d.iloc[rng.permutation(len(d))].reset_index(drop=True)


def encode(tok, texts, labels=None):
    e = tok([normalise(t) for t in texts], truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt")
    cols = [e["input_ids"], e["attention_mask"]]
    if labels is not None:
        cols.append(torch.tensor(list(labels)))
    return TensorDataset(*cols)


def train(df, epochs=2, bs=32, lr=3e-5):
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, num_labels=len(LABELS)).to(DEV)
    y = df.label.values
    loader = DataLoader(encode(tok, df.text, y), batch_size=bs, shuffle=True)

    counts = np.bincount(y, minlength=len(LABELS))
    w = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float, device=DEV)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(.1 * steps), steps)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for ids, mask, yb in loader:
            opt.zero_grad()
            out = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
            loss = loss_fn(out.logits, yb.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  epoch {ep + 1}  loss {total / len(loader):.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    return tok, model


def predict(tok, model, texts, bs=128):
    model.eval()
    loader = DataLoader(encode(tok, texts), batch_size=bs)
    out = []
    with torch.no_grad():
        for ids, mask in loader:
            logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.vstack(out)


@lru_cache(maxsize=1)
def load():
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT).to(DEV).eval()
    return tok, model


def score(texts):
    """-> P(scam) per text, for voice_model.window_risk."""
    return predict(*load(), list(texts))[:, LABELS.index("scam")]


def call_score(text):
    """-> P(scam) for a whole transcript: the loudest window wins. A scam call
    is a scam because of the thirty seconds where it asks, not the average."""
    w = windows(text, stride=WINDOW_WORDS // 2) or [text]
    return float(score(w).max())


if __name__ == "__main__":
    import json
    import pandas as pd

    d = build()
    print(f"windows: {len(d):,}  ({d.label.mean():.0%} scam)")
    print(d.src.value_counts().to_string())

    # Split by SOURCE DOCUMENT, not by window: two windows of one robocall share
    # phrasing, and splitting at window level leaks a call across the line.
    docs = d.doc.unique()
    perm = np.random.default_rng(0).permutation(len(docs))
    cut = int(0.85 * len(docs))
    tr = d[d.doc.isin(set(docs[perm[:cut]]))]
    te = d[d.doc.isin(set(docs[perm[cut:]]))]
    tok, model = train(tr)

    p = predict(tok, model, te.text)[:, 1]
    print(f"\nin-source held-out windows  AUC {roc_auc_score(te.label, p):.3f}  "
          f"(n={len(te):,})  <- the flattering number")

    # ---- The honest test. Never trained on, and both classes come through the
    #      same Whisper pipeline, so style cannot separate them.
    idx = data.DATA / "vishing" / "calls" / "index.json"
    cache = data.DATA / "vishing" / "calls" / "transcripts.json"
    if idx.exists():
        if not cache.exists():
            # ponytail: 16 min of Whisper. Cache it - this eval gets re-run.
            import voice_model as vm
            got = {c["id"]: " ".join(x[2] for x in vm.transcribe(c["path"]))
                   for c in json.load(open(idx))}
            json.dump(got, open(cache, "w"))
        txts = json.load(open(cache))
        rows = []
        for c in json.load(open(idx)):
            txt = txts.get(c["id"], "")
            if txt and sum(ord(ch) > 127 for ch in txt) / len(txt) < 0.02:
                rows.append({"label": c["label"], "new": call_score(txt),
                             "title": c["title"][:52]})
        g = pd.DataFrame(rows)
        print(f"\n27 real two-party calls, held out entirely  (n={len(g)})")
        print(f"  speech head   AUC {roc_auc_score(g.label, g.new):.3f}")
        print(f"  sms_model     AUC 0.433   <- what this head replaces")
        for lab, name in [(1, "scam"), (0, "legit")]:
            h = g[g.label == lab]
            print(f"    {name:<6} mean {h.new.mean():.3f}  (n={len(h)})")
        print(g.sort_values("new", ascending=False)
               .to_string(index=False, float_format="%.3f"))

    # ---- The authored set, also never trained on.
    a = data.load_transcripts()
    ap = np.array([call_score(t) for t in a.text])
    print(f"\n50 authored transcripts  AUC {roc_auc_score(a.label, ap):.3f}"
          f"   (sms_model reads 0.834)")
    print(f"  scam mean {ap[a.label == 1].mean():.3f}   "
          f"legit mean {ap[a.label == 0].mean():.3f}")
