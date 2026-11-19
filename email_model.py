"""Email channel: DistilBERT, three classes, with an auxiliary persuasion head.

Spec §4 (DistilBERT, max_len 256) and §6 (train on persuasion type - authority,
urgency, scarcity, fear, reward - not only the binary label).

WHY THREE CLASSES, not the binary label the corpora ship.
CEAS_08's "phishing" column actually means SPAM (CEAS = Conference on Email and
Anti-Spam; the 2008 challenge was spam filtering). A binary model trained on it
scored AUC 0.988 and still caught only 30.9% of Nazario, the real credential
phishing corpus - the emails it missed were CNN newsletters, romance-scam openers
and promo mail. So labels come from provenance instead: Nazario and
Nigerian_Fraud are phishing, everything else unwanted is spam. This mirrors the
SMS channel and makes "phishing recall" mean phishing.

SPLIT PROTOCOL. §5 mandates a temporal split, and one global cutoff cannot give
it: the corpora hold ZERO legitimate emails after 2010, so any cutoff past 2008
produces a 100%-phishing test set. Instead each corpus is cut on its own clock,
and the test half of every corpus is strictly later than its train half:

  ham/spam   CEAS_08     train 1-7 Aug 2008   test 8 Aug 2008
  phishing   Nigerian_Fraud  train <=2005     test 2006-2007
             Nazario         train <=2018     test 2019-2022
  cross-source  Enron + Ling + SpamAssassin, never seen in training (ham/spam)

MEASURED ABLATION, worth reporting. Holding out Nazario entirely and training the
phishing class on Nigerian_Fraud alone gives 15.1% recall on Nazario (median
phishing probability 0.004). The two corpora share no years at all
(Nigerian_Fraud 1998-2007, Nazario 2015-2022), so that split confounds corpus,
scam type and an 8-year gap. The honest reading: 419 advance-fee fraud does not
teach a model to recognise credential phishing - the subtypes need separate
representation in training.
"""
import re
from functools import lru_cache
from pathlib import Path

# ponytail: pandas before torch or pd.read_csv segfaults on Windows. See sms_model.
import data

import adversarial
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

CKPT = "distilbert-base-uncased"
OUT = Path(__file__).parent / "models" / "email_distilbert"
LABELS = ["ham", "spam", "phishing"]
MAX_LEN = 256  # §6
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# §6 persuasion principles. No corpus ships these labels, so they are weak-labelled
# by regex. ponytail: the aux head can only be as good as these patterns - it earns
# its place by generalising to paraphrases the regex misses, not by beating it.
# Hand-label a few hundred and retrain the head if the reason strings read wrong.
PERSUASION = {
    "authority": r"\b(bank|irs|paypal|administrator|official|security\s+team|"
                 r"government|federal|police|on\s+behalf\s+of|department)\b",
    "urgency":   r"\b(immediately|urgent|right\s+away|act\s+now|as\s+soon\s+as|"
                 r"within\s+\d+\s+(hours?|days?)|expires?|deadline|today)\b",
    "scarcity":  r"\b(limited|only\s+\d+|last\s+chance|hurry|while\s+supplies|"
                 r"exclusive|few\s+remaining)\b",
    "fear":      r"\b(suspend|block|terminat|clos(e|ed|ing)\s+your|legal\s+action|"
                 r"penalt|unauthori[sz]ed|breach|compromis|virus|arrest)\w*\b",
    "reward":    r"\b(won|winner|prize|lottery|free|bonus|inherit|million|"
                 r"congratulations|claim\s+your|reward)\b",
}
PRINCIPLES = list(PERSUASION)
_RX = {k: re.compile(v, re.I) for k, v in PERSUASION.items()}


# The spam class is dead on 2026 mail: measured 2026-08-27, ALL 73 messages in
# the two Gmail Spam boxes score ham (median spam prob 0.000) and the pre-modern
# checkpoint reaches only 2. Not a threshold problem - Gmail's spam box here IS
# the promotions genre (AIM newsletters, Simplilearn course mail) and v3 trained
# that genre as ham, because in the promotions box it is ham. What separates the
# two is consent and sender reputation, neither of which is in the text; same
# root cause as spam recall pinning at 0.609 on CEAS.
#
# So this rule does NOT try to reproduce Gmail. It covers one genre the model
# will otherwise always call ham: the cold paid-course / paid-internship pitch -
# "registration fee", referral code, AICTE/ISO certificates, a personal WhatsApp
# number, bcc'd to a student list. Fires only when a commercial ask (fee or
# referral) is present AND at least two other markers agree.
# ponytail: regex, not a head - there is exactly ONE labelled positive of this
# genre on disk, and a class you cannot train is a class you can only spot.
# Measured against 19,349 real ham messages from two mailboxes by
# `scratch/_solicitation_eval.py`; retrain a spam head the day the genre has a
# corpus.
SOLICITATION = {
    "bulk_salutation": r"\bdear\s+(students?|candidates?|aspirants?|applicants?|"
                       r"sir\s*/?\s*ma'?am|sir\s*/?\s*madam|all)\b",
    "fee":       r"\b(registration|enroll?ment|admission|course|program(me)?|"
                 r"training)\s+fee\b|\bfee\s*(of|is|:)",
    "referral":  r"\breferral\s+code\b|\bpromo\s+code\b[^.\n]{0,40}"
                 r"\b(concession|discount)\b",
    "whatsapp":  r"\bwhats\s?app\s*[:\-]?\s*\+?\d|\bphone\s*/\s*whats\s?app\b",
    "cert":      r"\b(aicte|iso\s*\d*)\b[^.\n]{0,40}certif|\bcertificate\s+of\s+"
                 r"(completion|excellence)\b|\bcompletion\s+certificate\b",
    "placement": r"\bplacement\s+assistance\b|\binternship\s+(program(me)?|opportunit)",
}
_SOL = {k: re.compile(v, re.I) for k, v in SOLICITATION.items()}
ASK = {"fee", "referral"}   # the commercial ask; without one this stays quiet


def solicitation(text):
    """-> marker names, or []. Non-empty means "unsolicited paid-course pitch"."""
    hits = sorted(k for k, r in _SOL.items() if r.search(text))
    return hits if ASK.intersection(hits) and len(hits) >= 3 else []


def persuasion_labels(texts):
    """-> (n, 5) multi-hot. Weak labels; see the note on PERSUASION."""
    return np.array([[1.0 if _RX[p].search(t) else 0.0 for p in PRINCIPLES]
                     for t in texts], dtype=np.float32)


class DualHead(nn.Module):
    """One encoder, two heads: ham/spam/phishing and persuasion (multi-label)."""

    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(CKPT)
        h = self.enc.config.dim
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Linear(h, len(LABELS))
        self.persuade = nn.Linear(h, len(PRINCIPLES))

    def forward(self, ids, mask):
        cls = self.drop(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.cls(cls), self.persuade(cls)


def encode(tok, texts, y=None, aux=None):
    # §9.9, same as sms_model: homoglyphs and zero-width chars folded out here.
    texts = [adversarial.normalize(t) for t in texts]
    e = tok(list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt")
    cols = [e["input_ids"], e["attention_mask"]]
    if y is not None:
        cols += [torch.tensor(list(y)), torch.tensor(aux)]
    return TensorDataset(*cols)


def train(df, epochs=2, bs=16, lr=3e-5, aux_weight=0.3):
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = DualHead().to(DEV)

    y = df.label_3.map(LABELS.index).values
    loader = DataLoader(encode(tok, df.text, y, persuasion_labels(df.text)),
                        batch_size=bs, shuffle=True)

    # phishing is ~10% of the pool; without weights it gets traded away for
    # accuracy on the two big classes, which is the one class that matters.
    counts = np.bincount(y, minlength=len(LABELS))
    w = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)),
                     dtype=torch.float, device=DEV)
    ce, bce = nn.CrossEntropyLoss(weight=w), nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(.1 * steps), steps)
    scaler = torch.amp.GradScaler(DEV)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for i, (ids, mask, yb, ab) in enumerate(loader):
            opt.zero_grad()
            with torch.amp.autocast(DEV, dtype=torch.float16):
                p, q = model(ids.to(DEV), mask.to(DEV))
                loss = ce(p, yb.to(DEV)) + aux_weight * bce(q, ab.to(DEV))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if i % 500 == 0:
                print(f"    step {i}/{len(loader)}  loss {loss.item():.4f}", flush=True)
        print(f"  epoch {ep + 1}  loss {tot / len(loader):.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "model.pt")
    tok.save_pretrained(OUT)
    return tok, model


def predict(tok, model, texts, bs=64):
    model.eval()
    loader = DataLoader(encode(tok, texts), batch_size=bs)
    cls, persuade = [], []
    with torch.no_grad(), torch.amp.autocast(DEV, dtype=torch.float16):
        for ids, mask in loader:
            p, q = model(ids.to(DEV), mask.to(DEV))
            cls.append(torch.softmax(p.float(), -1).cpu().numpy())
            persuade.append(torch.sigmoid(q.float()).cpu().numpy())
    return np.vstack(cls), np.vstack(persuade)


def report(name, tok, model, df):
    p, _ = predict(tok, model, df.text)
    y = df.label_3.map(LABELS.index).values
    # Average only over classes that actually occur in y. A class with zero
    # support scores F1 0 and drags macro-F1 down by a third for no reason.
    present = sorted(set(y))
    pred = p.argmax(1)
    macro = f1_score(y, pred, labels=present, average="macro", zero_division=0)
    print(f"\n--- {name}  (n={len(df):,})  macro-F1 {macro:.3f}")
    print(classification_report(y, pred, labels=present,
                                target_names=[LABELS[i] for i in present],
                                digits=3, zero_division=0))
    return macro


@lru_cache(maxsize=1)
def load():
    """Cached: fusion calls score() per message and reloading 66M params each
    time costs ~2s. lru_cache is the whole cache layer."""
    tok = AutoTokenizer.from_pretrained(OUT)
    model = DualHead().to(DEV)
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=DEV))
    return tok, model


def score(texts):
    """-> ((n,3) class probs, (n,5) persuasion probs). Fusion consumes column 2."""
    tok, model = load()
    return predict(tok, model, texts)


def explain(text, threshold=.5):
    """§4 reason string, built from the auxiliary head."""
    p, q = score([text])
    verdict = LABELS[int(p[0].argmax())]
    hits = [n for n, v in zip(PRINCIPLES, q[0]) if v >= threshold]
    tail = f"; uses {' and '.join(hits)}" if hits else ""
    return f"{verdict} (phishing {p[0][2]:.2f}, spam {p[0][1]:.2f}){tail}"


def phishing_split(e):
    """Temporal split inside EACH phishing corpus.

    Holding out Nazario entirely was tried and measured: 15.1% recall. The two
    corpora do not overlap in time at all (Nigerian_Fraud 1998-2007, Nazario
    2015-2022), so that split confounded corpus, scam type and an 8-year gap.
    Splitting within each keeps both scam types in training while the test half
    is still strictly later in time - a real §5 temporal split for this class.
    """
    p = e[e.label_3 == "phishing"].copy()
    yr = p.date.dt.year
    cut = p.corpus.map({"Nazario": 2018, "Nigerian_Fraud": 2005})
    older = yr <= cut
    return p[older | yr.isna()], p[~older & yr.notna()]


if __name__ == "__main__":
    e = data.load_email()
    aug = e[(e.corpus == "CEAS_08") & e.date.notna()]
    aug = aug[aug.date.dt.strftime("%Y-%m") == "2008-08"]
    ph_tr, ph_te = phishing_split(e)

    tr = data.pd.concat([aug[aug.date.dt.day <= 7], ph_tr])
    # §9.9: obfuscated twins for half the training rows. Test sets stay clean.
    tr = data.pd.concat([tr, adversarial.augment(tr)], ignore_index=True)
    te_time = data.pd.concat([aug[aug.date.dt.day == 8], ph_te])
    te_src = e[e.corpus.isin(["Enron", "Ling", "SpamAssasin"])]

    print(f"train         {len(tr):,} {tr.label_3.value_counts().to_dict()}")
    print(f"temporal test {len(te_time):,} {te_time.label_3.value_counts().to_dict()}")
    print(f"cross-source  {len(te_src):,} (ham/spam only, unseen corpora)")
    tok, model = train(tr)

    report("temporal (later mail, same corpora)", tok, model, te_time)
    report("cross-source (unseen corpora, ham/spam)", tok, model, te_src)

    for c in ["Nazario", "Nigerian_Fraud"]:
        held = ph_te[ph_te.corpus == c]
        if len(held):
            p, _ = predict(tok, model, held.text)
            print(f"held-out {c:15} n={len(held):5}  caught "
                  f"{100 * (p.argmax(1) == 2).mean():5.1f}%")

    print("\n" + explain("Your account has been suspended. Verify within 24 hours "
                         "or it will be permanently closed by the security team."))
    print(explain("Hi Priya, attaching the minutes from Tuesday's review meeting."))
    print(explain("LAST CHANCE - 70% off everything, offer expires today only!"))

    p, _ = score(["URGENT: your bank account is blocked, confirm your details now",
                  "Please find attached the quarterly figures we discussed."])
    assert p[0][2] > p[1][2], "phishing must outscore an ordinary work email"
    print("ok")
