"""One-epoch fine-tune of XLS-R-300m itself, vs the frozen-embedding heads of
_spoof_asv5.py. Asked 2026-10-25: does training the speech model's own top
layers buy back the phone line (ASV21 pstn) that the frozen ASV19+ASV5 head
lost (3.5 -> 6.5%) without losing its In-the-Wild gain?

Fixed before any result was seen:
  data   the SAME train rows as _spoof_asv5: ASV19 LA train + ASV5 train
         sample (all bonafide + 3,000 per attack). One pass, batch 8, seed 0.
         Each clip goes through spoof_model.telephony() with p = 0.5.
  model  feature encoder + transformer layers 0-11 frozen; layers 12-23 and a
         head (softmax-weighted sum of all 25 hidden states -> mean+std ->
         linear) trained. AdamW, 1e-5 speech layers / 1e-3 head, 300 warm-up
         steps, class-weighted cross-entropy, bf16 autocast. Inputs normalised
         per clip (XLS-R's feature-extractor convention).
  test   the SAME three as _spoof_asv5 (ASV21 5,600 draw, ASV5 eval slice,
         all of ITW clean + phone). No checkpoint selection: the one epoch's
         end state is the result, so no dev set is consulted.
  rule   a GPU crash ends it (Mohak, 2026-10-25): no retry, keep the frozen head.

ponytail: no silence trimming. The frozen heads read the same untrimmed first
4 s, so the comparison is like for like; all three tests are external, so a
silence shortcut would show up as a bad number there, not hide.
ponytail: checkpoints keep weights only, not Adam state (a 1.8 GB save would
itself trip the low-memory reaper); a resume restarts Adam's moments.

Writes no shipped model: everything under Datasets/vishing/finetune/. Rates only.

    python -m scratch._spoof_finetune              # epoch 1
    python -m scratch._spoof_finetune --epoch 2    # continues from epoch 1's weights

Epoch N reshuffles with seed N-1 and draws fresh phone flags. Both epochs' end
states are reported; neither is picked by its test numbers alone.
"""
import os
import sys
import time

import data  # before torch
import numpy as np
import pandas as pd
import spoof_model as sm
import torch
from torch import nn

D = data.DATA / "vishing" / "finetune"
E = int(sys.argv[sys.argv.index("--epoch") + 1]) if "--epoch" in sys.argv else 1
SUF = "" if E == 1 else f"_e{E}"                       # epoch 1 keeps its original file names
CKPT, FINAL = D / f"ckpt{SUF}.pt", D / f"final{SUF}.pt"
PREV = D / ("final.pt" if E == 2 else f"final_e{E - 1}.pt")
ITW = data.DATA / "vishing" / "in_the_wild" / "release_in_the_wild"
FROZEN = 12           # transformer layers 0..11 stay frozen
BS, WARM, SAVE_EVERY = 8, 300, 500


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.ssl = Wav2Vec2Model.from_pretrained(sm.XLSR, layerdrop=0.0)
        self.ssl.freeze_feature_encoder()
        for p in self.ssl.feature_projection.parameters():
            p.requires_grad_(False)
        for p in self.ssl.encoder.pos_conv_embed.parameters():
            p.requires_grad_(False)
        for layer in self.ssl.encoder.layers[:FROZEN]:
            layer.requires_grad_(False)
        # Frozen too, or its gradient makes autograd keep the frozen layers' activations.
        self.ssl.masked_spec_embed.requires_grad_(False)
        n = self.ssl.config.num_hidden_layers + 1
        self.mix = nn.Parameter(torch.zeros(n))
        self.out = nn.Linear(2 * self.ssl.config.hidden_size, 2)

    def forward(self, wav):
        h = torch.stack(self.ssl(wav, output_hidden_states=True).hidden_states)  # (L,B,T,H)
        h = (torch.softmax(self.mix, 0)[:, None, None, None] * h).sum(0)
        return self.out(torch.cat([h.mean(1), h.std(1)], -1))


def norm(wav):
    return (wav - wav.mean(1, keepdims=True)) / (wav.std(1, keepdims=True) + 1e-7)


def batch(paths, phone):
    return torch.from_numpy(norm(np.stack(
        [sm.load_audio(p, phone=bool(f)) for p, f in zip(paths, phone)])))


def train_rows():
    a19 = data.load_spoof("train")
    t5 = data.load_spoof5("train")
    assert len(t5) == 182_357, f"only {len(t5):,} ASV5 train flacs on disk"
    rng = np.random.default_rng(0)                       # = _spoof_asv5's sample
    t5 = pd.concat([t5[t5.label == 0]] +
                   [g.iloc[rng.permutation(len(g))[:3000]] for _, g in t5[t5.label == 1].groupby("attack")])
    d = pd.concat([a19[["path", "label"]], t5[["path", "label"]]]).reset_index(drop=True)
    r = np.random.default_rng(E - 1)
    return d.iloc[r.permutation(len(d))].reset_index(drop=True), r.random(len(d)) < 0.5


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(net):
    d, phone = train_rows()
    steps = len(d) // BS
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    trainable = lambda: {k: v for k, v in net.state_dict().items() if k in names}
    start = 0
    if E > 1:
        net.load_state_dict(torch.load(PREV, map_location="cpu"), strict=False)
        print(f"epoch {E}: starting from {PREV.name}", flush=True)
    if CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu")
        net.load_state_dict(ck["w"], strict=False)
        start = ck["step"]
        print(f"resumed at step {start:,} (Adam moments restart)", flush=True)
    ssl_p = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("ssl")]
    head_p = [net.mix, *net.out.parameters()]
    opt = torch.optim.AdamW([{"params": ssl_p, "lr": 1e-5}, {"params": head_p, "lr": 1e-3}],
                            weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    cw = torch.tensor(len(d) / (2 * np.bincount(d.label)), dtype=torch.float32, device=sm.DEV)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    print(f"train {len(d):,} clips ({(d.label == 0).sum():,} bonafide), {steps:,} steps, "
          f"trainable {sum(p.numel() for p in ssl_p + head_p) / 1e6:.0f}M params", flush=True)
    net.train()
    t0, run = time.time(), []
    for s in range(start, steps):
        warm = min(1.0, (s + 1) / WARM)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * warm
        i = slice(s * BS, (s + 1) * BS)
        x = batch(d.path.values[i], phone[i]).to(sm.DEV)
        y = torch.from_numpy(d.label.values[i]).to(sm.DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ssl_p + head_p, 1.0)
        opt.step()
        run.append(loss.item())
        if (s + 1) % 100 == 0:
            done = s + 1 - start
            rate = done * BS / (time.time() - t0)
            print(f"  step {s + 1:,}/{steps:,}  loss {np.mean(run[-100:]):.3f}  "
                  f"{rate:.1f} clips/s  eta {(steps - s - 1) * BS / rate / 60:.0f} min", flush=True)
        if (s + 1) % SAVE_EVERY == 0:
            save({"step": s + 1, "w": trainable()}, CKPT)
    save(trainable(), FINAL)
    CKPT.unlink(missing_ok=True)


@torch.no_grad()
def scores(net, tag, paths, phone=False):
    f = D / f"scores_{tag}{SUF}.npy"
    if not f.exists():
        net.eval()
        out = []
        for i in range(0, len(paths), 16):
            x = batch(paths[i:i + 16], [phone] * len(paths[i:i + 16])).to(sm.DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.softmax(net(x).float(), -1)[:, 1].cpu().numpy())
        np.save(f, np.concatenate(out))
        print(f"  scored {tag} ({len(paths):,})", flush=True)
    return np.load(f)


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    net = Net().to(sm.DEV)
    if not FINAL.exists():
        train(net)
    net.load_state_dict(torch.load(FINAL, map_location="cpu"), strict=False)

    s21 = data.load_spoof21()
    r = np.random.default_rng(0)
    d21 = pd.concat([g.iloc[r.permutation(len(g))[:400]] for _, g in s21.groupby(["codec", "label"])])
    e5 = data.load_spoof5("eval")
    r = np.random.default_rng(0)
    e5 = pd.concat([g.iloc[r.permutation(len(g))[:400]]
                    for _, g in e5.groupby(["codec", "label"])]).reset_index(drop=True)
    meta = pd.read_csv(ITW / "meta.csv")
    yitw = (meta.label == "spoof").astype(int).values
    itw = [str(ITW / f) for f in meta.file]

    p21 = scores(net, "a21", list(d21.path))
    p5 = scores(net, "e5", list(e5.path))
    pi = {ph: scores(net, f"itw_{'phone' if ph else 'clean'}", itw, ph) for ph in (False, True)}
    pst = (d21.codec == "pstn").values
    print(f"\n  {f'fine-tuned {E} epoch':<22} ASV21 EER {sm.eer(d21.label, p21):6.2%}  "
          f"pstn {sm.eer(d21.label[pst], p21[pst]):6.2%}  |  ASV5 eval {sm.eer(e5.label, p5):6.2%}  |  "
          f"ITW clean {sm.eer(yitw, pi[False]):6.2%}  phone {sm.eer(yitw, pi[True]):6.2%}  "
          f"real-voice mean {pi[True][yitw == 0].mean():.3f}")
    print("  frozen ASV19 + ASV5    ASV21 EER  5.36%  pstn  6.50%  |  ASV5 eval  9.48%  |  "
          "ITW clean 13.55%  phone 17.94%  real-voice mean 0.040   (_spoof_asv5 RESULT)")

# RESULT 2026-10-25, epoch 1 (resumed once at step 6,500 after a low-memory kill; Adam moments restarted there)
#   head                 ASV21   pstn  | ASV5 eval |  ITW clean  phone | ITW real-voice mean
#   fine-tuned 1 epoch    6.70%  6.37% |    8.58%  |   21.65%  18.33%  |  0.002
#   frozen ASV19+ASV5     5.36%  6.50% |    9.48%  |   13.55%  17.94%  |  0.040
# Fine-tuning fits the ASVspoof corpora a little better and real-world deepfakes worse (ITW clean +8 pts);
# the phone line is unchanged. Not shipped: the frozen ASV19+ASV5 head stays the pick. Epoch 2 paused (Mohak).
