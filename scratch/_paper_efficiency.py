"""Size and latency of every shipped model, for the paper's efficiency table.

Latency is the median of N single-item calls after a warm-up (batch size 1 -
what a live message or call costs), on this machine. Text inputs are public
(SMS: Dataset_5971; email: phishing_pot; URLs: the OpenPhish+Tranco feed);
voice uses public FTC robocall audio. Prints timings only, never text.
    python -m scratch._paper_efficiency
"""
import os
import platform
import statistics
import time
from pathlib import Path

import data   # before torch (Windows libiomp)
import torch

ROOT = Path(data.__file__).resolve().parent
M = ROOT / "models"


def size(p):
    p = Path(p)
    return (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size) / 1e6


def med(f, items, warm=3):
    for x in items[:warm]:
        f(x)
    ts = []
    for x in items:
        t = time.perf_counter()
        f(x)
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"hardware: CPU {platform.processor()} ({os.cpu_count()} threads), GPU {gpu}, torch {torch.__version__}")
    rows = []

    import sms_model
    s = data.load_sms()
    sms = s[s.source == "Dataset_5971"].text.fillna("").tolist()[:300]
    pipe = sms_model.load()
    n_feat = len(pipe.steps[0][1].vocabulary_)
    rows.append(("SMS char-LR", f"{n_feat * 3 / 1e6:.2f}M weights", size(M / "sms_charlr.pkl"),
                 med(lambda t: sms_model.score([t]), sms), "CPU"))

    import url_model
    urls = data.load_url_feed().url.tolist()[:300]
    rows.append(("URL XGBoost (400 trees)", "15 features", size(M / "url_xgb.json"),
                 med(lambda u: url_model.score([u]), urls), "CPU"))

    import email_model as em
    pot = data.load_phishing_pot().text.tolist()[:100]
    tok, model = em.load()
    rows.append(("Email DistilBERT v6 + persuasion head", f"{params(model):.0f}M params", size(M / "email_distilbert"),
                 med(lambda t: em.score([t]), pot), "GPU fp16"))

    import act_model
    import voice_model as vm
    rc = data.load_robocalls()
    texts = [t for t in rc.transcript.tolist() if isinstance(t, str) and len(t.split()) > 40][:60]
    atok, amod = act_model.load()
    rows.append(("Voice act model (DistilBERT)", f"{params(amod):.0f}M params", size(M / "act_distilbert"),
                 med(lambda t: act_model.predict(atok, amod, [t]), texts), "GPU, per transcript (<=256 tokens)"))

    import spoof_model as sm
    paths = [str(p) for p in rc.path.tolist()[:40]]
    sm.score(paths[:2])
    fe_model = sm._cached_frontend()
    rows.append(("Spoof: wav2vec2-XLS-R-300m (frozen) + LR head", f"{params(fe_model):.0f}M params",
                 size(M / "spoof_head.pkl"), med(lambda p: sm.score([p]), paths), "GPU, per 4-s clip"))

    calls = sorted((ROOT / "Datasets" / "vishing" / "calls").glob("*.wav"))[:5]
    import soundfile as sf
    vm.transcribe(str(calls[0]))
    dur = tt = 0.0
    for c in calls:
        dur += sf.info(str(c)).duration
        t = time.perf_counter()
        vm.transcribe(str(c))
        tt += time.perf_counter() - t
    rows.append(("Whisper small (faster-whisper)", "244M params", float("nan"),
                 1000 * tt / (dur / 60), f"GPU fp16, per audio minute (RTF {tt / dur:.3f})"))

    print(f"\n{'component':48} {'size':>16} {'disk MB':>8} {'median ms':>10}  device")
    for name, n, mb, ms, dev in rows:
        print(f"{name:48} {n:>16} {mb:8.1f} {ms:10.2f}  {dev}")
    a = ROOT / "android" / "app" / "src" / "main" / "assets"
    print(f"\nphone assets: sms_charlr.bin {size(a / 'sms_charlr.bin'):.2f} MB, url_xgb.bin {size(a / 'url_xgb.bin'):.2f} MB")


if __name__ == "__main__":
    main()
