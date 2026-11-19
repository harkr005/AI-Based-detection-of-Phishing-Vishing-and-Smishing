"""SMS channel: MobileBERT, three classes (ham / spam / smishing).

Spec §4 picks MobileBERT so this can run on the phone later, and §6 frames the
task as intent analysis - ham vs spam vs smishing - not binary spam detection.

Two numbers are always reported:
  in-source   held-out slice of the corpora it trained on   (the flattering one)
  cross-source  a corpus it has never seen at all           (the honest one)
The gap is how much of the score is the dataset rather than the smishing.

No temporal split here: none of the SMS corpora carry timestamps. Email and URL
do get temporal splits - see the report before quoting a protocol.
"""
from functools import lru_cache
from pathlib import Path

# data (pandas) MUST be imported before torch. Both ship libiomp5md.dll on
# Windows; if torch initialises OpenMP first, pd.read_csv segfaults the process
# with no traceback. Keep this line above the torch imports.
import data

import adversarial
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

CKPT = "google/mobilebert-uncased"
OUT = Path(__file__).parent / "models" / "sms_mobilebert"
LABELS = ["ham", "spam", "smishing"]
MAX_LEN = 64  # SMS average ~120 chars; 64 word-pieces covers the tail
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def encode(tok, texts, labels=None):
    # §9.9: strip zero-width chars and fold homoglyphs back to ASCII. Same call
    # in training and scoring, so there is no skew.
    texts = [adversarial.normalize(t) for t in texts]
    e = tok(list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt")
    cols = [e["input_ids"], e["attention_mask"]]
    if labels is not None:
        cols.append(torch.tensor(list(labels)))
    return TensorDataset(*cols)


def train(df, epochs=4, bs=32, lr=3e-5):
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, num_labels=len(LABELS)).to(DEV)

    y = df.label_3.map(LABELS.index).values
    loader = DataLoader(encode(tok, df.text, y), batch_size=bs, shuffle=True)

    # ham outnumbers smishing ~2:1 - weight the loss so the rare class still counts
    counts = np.bincount(y, minlength=len(LABELS))
    w = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float, device=DEV)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # MobileBERT's freshly initialised head emits logits in the millions, so the
    # first loss is ~1e6. Without warmup + clipping the first epoch is spent
    # undoing the damage AdamW does while recovering from it.
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
        print(f"  epoch {ep + 1}  loss {total / len(loader):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    return tok, model


def predict(tok, model, texts, bs=64, raw=False):
    """raw=True returns logits. explain() needs them: on confident messages the
    softmax pins to 1.0 in float32 and every word looks equally unimportant."""
    model.eval()
    loader = DataLoader(encode(tok, texts), batch_size=bs)
    out = []
    with torch.no_grad():
        for ids, mask in loader:
            logits = model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            out.append((logits if raw else torch.softmax(logits, -1)).cpu().numpy())
    return np.vstack(out)


def report(name, tok, model, df):
    p = predict(tok, model, df.text).argmax(1)
    y = df.label_3.map(LABELS.index).values
    macro = f1_score(y, p, average="macro")
    print(f"\n--- {name}  (n={len(df):,})  macro-F1 {macro:.3f}")
    print(classification_report(y, p, target_names=LABELS, digits=3, zero_division=0))
    return macro


@lru_cache(maxsize=1)
def load():
    """Cached: voice scoring calls score() once per 3s window, and reloading the
    model each time dominated the runtime. lru_cache is the whole cache layer."""
    return (AutoTokenizer.from_pretrained(OUT),
            AutoModelForSequenceClassification.from_pretrained(OUT).to(DEV))


def score(texts, raw=False):
    """-> (n, 3) probabilities over ham/spam/smishing. Fusion consumes column 2."""
    tok, model = load()
    return predict(tok, model, texts, raw=raw)


MAX_WORDS = 40  # explanation cost is one forward pass per word; SMS are short


def explain(text, top_k=3):
    """-> (label, probability, reason string). §4 requires a reason per verdict.

    Leave-one-word-out: drop each word, see how far the predicted class falls.
    The words that cost the most ARE the evidence, measured on the model itself.

    ponytail: this is LIME without the local linear fit - same perturb-and-watch
    idea, one batched pass instead of 1000 samples and a regression. Swap in real
    LIME only if the report needs the citation. On-device this runs ~n_words
    times the single-message cost, so call it for flagged messages, not for every
    SMS that arrives.
    """
    words = adversarial.normalize(text).split()[:MAX_WORDS]
    if not words:
        return LABELS[0], 1.0, "empty message"

    # row 0 is the intact message, row i+1 is the message without word i
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:])
                                    for i in range(len(words))]
    # Margin in logit space, not probability: a clear smishing message saturates
    # the softmax at 1.0, so every prob-drop collapses to float32 noise and no
    # word ever clears a fixed threshold. The margin over the runner-up class
    # keeps its dynamic range however confident the model is.
    lg = score(variants, raw=True)
    k = int(lg[0].argmax())
    other = np.delete(lg, k, axis=1).max(1)
    margin = lg[:, k] - other
    drop = margin[0] - margin[1:]
    p = float(np.exp(lg[0, k] - np.logaddexp.reduce(lg[0])))

    # 1% of the intact margin, so the cut scales with the model's confidence
    top = [i for i in sorted(range(len(words)), key=lambda i: -drop[i])[:top_k]
           if drop[i] > 0.01 * abs(margin[0])]
    if not top:
        return LABELS[k], p, f"{LABELS[k]}, no single word decides it"
    quoted = ", ".join('"%s"' % words[i].strip('"\'.,!?:;') for i in sorted(top))
    return LABELS[k], p, f"{LABELS[k]} - driven by {quoted}"


if __name__ == "__main__":
    s = data.load_sms()

    # Dataset_5971 is held out entirely - a corpus the model never sees.
    held = s[s.source == "Dataset_5971"]
    pool = s[s.source != "Dataset_5971"]
    # Split by group, not at random: the Indian rows are template-generated, so a
    # random split leaks near-twins into the test set. Costs the stratification -
    # groups win, the classes are large enough that the split stays close anyway.
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
                      .split(pool, groups=pool.group))
    tr, te = pool.iloc[i_tr], pool.iloc[i_te]

    # §9.9: half the training rows get an obfuscated twin. Test sets stay clean -
    # adversarial.py is what measures the attacked numbers.
    tr = data.pd.concat([tr, adversarial.augment(tr)], ignore_index=True)

    print(f"train {len(tr):,}  in-source test {len(te):,}  cross-source test {len(held):,}")
    tok, model = train(tr)

    in_src = report("in-source (same corpora)", tok, model, te)
    cross = report("cross-source (unseen corpus)", tok, model, held)
    print(f"\ngap: {in_src - cross:+.3f} macro-F1 - that is the dataset, not the model")

    p = score([
        "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked",
        "Hey, are we still meeting at 6 for dinner?",
    ])
    smish, ham = p[0][2], p[1][0]
    print(f"\nsanity: fake-KYC smishing={smish:.2f}  dinner-text ham={ham:.2f}")
    assert smish > 0.5 and ham > 0.5, "obvious cases must land in the right class"

    lab, prob, why = explain(
        "Your SBI KYC is expired. Update now at http://sbi-kyc-verify.tk or account blocked")
    print(f"reason: {why}  ({prob:.2f})")
    assert lab == "smishing" and prob > 0.5
    # the explanation must point at the lure, not at "your" or "now"
    assert any(w in why.lower() for w in ("kyc", "sbi", "http", "blocked", "expired")), why
    print("ok")
