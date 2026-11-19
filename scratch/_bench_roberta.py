"""RoBERTa-base on the email channel - the §4 benchmark the spec names.

Identical splits, identical training rows (adversarial twins included), identical
hyperparameters to the shipped DistilBERT. The only change is the encoder, so any
difference is the encoder.
"""
import pathlib
import re
import time

import data                      # before torch
import pandas as pd
import torch
import torch.nn as nn
import adversarial
import email_model as em
from transformers import AutoModel


class DualHead(nn.Module):
    """Same two heads; reads the width from whichever config the encoder ships.
    DistilBERT calls it `dim`, RoBERTa calls it `hidden_size`."""

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(em.CKPT)
        cfg = self.enc.config
        h = getattr(cfg, "dim", None) or cfg.hidden_size
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Linear(h, len(em.LABELS))
        self.persuade = nn.Linear(h, len(em.PRINCIPLES))

    def forward(self, ids, mask):
        cls = self.drop(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.cls(cls), self.persuade(cls)


em.DualHead = DualHead
em.CKPT = "roberta-base"
em.OUT = pathlib.Path(data.__file__).resolve().parent / "models" / "email_roberta"

e = data.load_email()
aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
ph_tr, ph_te = em.phishing_split(e)
tr = pd.concat([aug[aug.date.dt.day <= 7], ph_tr])
tr = pd.concat([tr, adversarial.augment(tr)], ignore_index=True)
te_time = pd.concat([aug[aug.date.dt.day == 8], ph_te])
te_src = e[e.corpus.isin(["Enron", "Ling", "SpamAssasin"])]

print(f"encoder {em.CKPT}  train {len(tr):,}  temporal {len(te_time):,}  "
      f"cross-source {len(te_src):,}", flush=True)

t0 = time.time()
tok, model = em.train(tr)
mins = (time.time() - t0) / 60
print(f"\ntrained in {mins:.1f} min", flush=True)

em.report("temporal (later mail, same corpora)", tok, model, te_time)
em.report("cross-source (unseen corpora, ham/spam)", tok, model, te_src)

for c in ["Nazario", "Nigerian_Fraud"]:
    held = ph_te[ph_te.corpus == c]
    if len(held):
        p, _ = em.predict(tok, model, held.text)
        print(f"held-out {c:15} n={len(held):5}  caught {100 * (p.argmax(1) == 2).mean():5.1f}%")

# the test that actually matters: mail from years the training set has never seen
pot = data.load_phishing_pot()
EN = set("the and you your account for that with this have from will not are was please click here".split())
enf = lambda t: (lambda w: sum(x in EN for x in w) / max(len(w), 1))(re.findall(r"[a-z]+", t.lower()[:2000]))
pot_en = pot[pot.text.map(enf) >= 0.04]
p, _ = em.predict(tok, model, pot_en.text)
print(f"\nphishing_pot ENGLISH n={len(pot_en):,}  caught {100 * (p.argmax(1) == 2).mean():.1f}%"
      f"  (DistilBERT 47.1%)")
by_year = pot_en.assign(pred=p.argmax(1))
for yr, g in by_year[by_year.date.dt.year.between(2022, 2026)].groupby(by_year.date.dt.year):
    print(f"  {yr}  n={len(g):<5} caught {100 * (g.pred == 2).mean():5.1f}%")
