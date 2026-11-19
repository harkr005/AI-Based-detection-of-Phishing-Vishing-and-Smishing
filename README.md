# AI-Based Detection of Phishing, Vishing and Smishing Attacks

Final-year B.E. (Information Science) project, BMS Institute of Technology and Management, Bengaluru.

One system that scores social-engineering attacks across **three channels — email (phishing), SMS (smishing), voice (vishing)** — by combining content NLP with a **behavioural intelligence engine**, and emits an explainable verdict.

**Thesis:** content changes with every campaign, behaviour does not. Prior work does email *or* SMS *or* voice and is almost entirely content-based. Here all three feed one fusion layer, and sender/caller behaviour carries the decision when the content is unfamiliar.

Full design document: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

## Architecture

```
INGEST        email (IMAP)        SMS (Android / gateway)     call audio (3s windows, 1s overlap)
                 |                        |                            |
PREPROCESS    header parse           text clean                     ASR (faster-whisper)
              URL extract            URL extract                    audio frames
                 |                        |                            |
MODELS        DistilBERT 3-class     MobileBERT 3-class          act layer + rule primitives
              + persuasion head      (on-device)                 wav2vec2-XLS-R spoof head
                        \            XGBoost URL model          /
                         \                  |                  /
BEHAVIOUR      per-sender / per-caller profiles -> deviation score in [0,1]
                                    |
FUSION         adaptive weighted sum (+ LR stacker as comparison)
                                    |
OUTPUT         SAFE 0-0.35 | SUSPICIOUS 0.35-0.70 | PHISHING 0.70-1.0  + reason strings
```

## Repository layout

| Path | What it is |
|---|---|
| `api.py` | FastAPI sidecar — `POST /sms`, `/email`, `/voice/text`, `/voice/file`, `GET /health` |
| `data.py` | Dataset loaders, all normalised to `1 = malicious` |
| `sms_model.py` | Character 3–5-gram TF-IDF + logistic regression, 3-class ham / spam / smishing (MobileBERT kept as an experiment) |
| `email_model.py` | DistilBERT, 3-class ham / spam / phishing + auxiliary persuasion head |
| `url_model.py` | XGBoost on domain-level lexical features, TreeSHAP reason strings |
| `attachments.py` | Email attachments: dangerous file types, links inside PDF / HTML / calendar files |
| `voice_model.py` | Streaming transcript risk — rule primitives, asymmetric EMA ratchet, §3 bands |
| `act_model.py` | Call-level act/primitive classifier for the voice channel |
| `speech_model.py` | Transcript-classifier experiment (kept as a documented negative result) |
| `spoof_model.py` | Frozen wav2vec2-XLS-R + logistic spoof head, phone-augmented |
| `behaviour.py` | Behavioural profiles and deviation scoring per channel |
| `fusion.py` | Adaptive weighted sum, escalation gate, LR stacker |
| `thread.py` | Thread-level rolling SMS scoring |
| `adversarial.py` | Unicode normalisation + obfuscation augmentation (shared by both transformers) |
| `llm_primitives.py` | LLM-assisted primitive labelling for the voice evidence layer |
| `store.py` | Postgres persistence for scans and feedback |
| `scratch/` | Evaluation and training scripts behind the numbers below |
| `web/` | Next.js dashboard — auth, per-channel scan pages, risk timeline, OCR input |

## Not in this repository

`Datasets/` (~47 GB), `models/` (~4.4 GB of trained weights) and `papers/` are gitignored. The training scripts rebuild the weights; each model file's `__main__` trains and evaluates it. Corpora used are listed in §5 of `PROJECT_SPEC.md`.

## Running it

```bash
pip install torch transformers scikit-learn xgboost pandas numpy soundfile \
            faster-whisper imageio-ffmpeg fastapi uvicorn psycopg[binary] joblib scipy pypdf

python -m uvicorn api:app --reload      # sidecar on :8000  (needs models/)

cd web && npm install && npm run dev    # dashboard on :3000
```

`web/.env.local`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres, shared with `store.py` |
| `AUTH_SECRET` | Session signing |
| `SIDECAR_URL` | Defaults to `http://127.0.0.1:8000` |
| `GEMINI_API_KEY` | OCR route |

> On Windows, `import data` must precede `import torch` — a libiomp conflict segfaults `pd.read_csv` otherwise.

## Measured results

Re-measured 2026-10-28 on the shipped weights (email v6, the phonemix spoof
head), each on a split the model never trained on. `scratch/_paper_metrics.py`
(SMS, email, URL) and `scratch/_paper_spoof.py` reproduce them without
retraining; fusion and voice come from `python fusion.py` and `python
voice_model.py`, the new-voice and attachment rows from
`scratch/_spoof_phonemix.py` and `scratch/_attach_eval.py`.

| Channel | Metric | Result |
|---|---|---|
| SMS | cross-source macro-F1 | **0.859** (0.980 in-source); smishing recall 0.786 cross-source. Word list from public SMS only, so nothing private ships in the APK (was 0.870 / 0.807 with the inbox in it) |
| Email | phishing precision / recall, temporal split | **0.990 / 0.986**; 0.916 / 0.862 on 2025–26 mail (false alarms 1.9% of that ham, 1.2% of Enron-Spam ham) |
| Email attachments | phishing with attachments not SAFE / false alarms added on 1,606 real mails with attachments | **93.0% → 93.8%** / **+0.0 pt** |
| URL | holdout AUC, OpenPhish vs Tranco | **0.920** (recall 0.803 at 0.5) |
| Fusion | spam recall on CEAS day-8 stream | **0.611 → 0.912** once behaviour escalates content-clean mail |
| Voice spoof | caught, modern AI voices never trained on, phone channel (41 unseen MLAAD tools / Edge-TTS) | **85.8% / 96.7%** (previous head: 20.0% / 20.0%) |
| Voice spoof | real voices flagged as AI: EdAcc video calls (phone) / 105 calls from 7 unseen scam-baiting channels | **0.6%** / **6.7%** (previous head: 2.9% / 1.9%) |
| Voice spoof | EER, In-the-Wild real-world deepfakes (clean / phone) | **13.3% / 16.3%** (previous head 13.6% / 17.9%; ASVspoof 2019-only: 38.8% / 42.9%) |
| Voice spoof | EER, ASVspoof 2021 LA labelled telephony / ASVspoof 5 eval | **5.82% pooled, 8.13% pstn** / **14.23%** (previous head 5.36%, 6.50% / 9.48%) |
| Voice content | accuracy, 1,598 Scam Dialogue calls (synthetic, same source both classes) | **86.4%** (AUC 0.917, false alarms 3.1%) |
| Voice content | accuracy / AUC, 27 recorded two-party calls | **81.5% / 0.850** |
| Voice content | false alarms, 723 real bank calls / caught, 105 calls from 7 unseen scam-baiting channels | **0.3%** / **96.2%** |

### Caveats that must travel with these numbers

- The CEAS "has-URL" flag is a **corpus artifact** and is withheld from both fusers: alone it scores AUC 0.395 (inverted), because 2008 ham was mailing-list digests and 2008 spam was plain-text. Including it inflates the stacker to a false 0.986.
- PhiUSIIL encodes its own collection method — 100% of its legitimate URLs are HTTPS with no path. Training on scheme/path features gives AUC 0.331 on live data, *worse than chance*. Domain-level lexical features only.
- The shipped spoof head (2026-10-27) trades lab scores for modern voices: trained with 143-tool MLAAD fakes and many real-voice sources (incl. real phone calls), it catches 4x more unseen modern AI voices but scores worse on the ASVspoof lab sets, and it **failed one condition fixed before testing**: real scam-call audio from 7 unseen YouTube channels is flagged as AI 6.7% of the time vs 1.9% before (7 of 105 calls). It was promoted by choice. The synthetic-voice slot only ever raises risk, never clears a call.
- The attachment gain is small because the content model already catches 93% of phishing that carries a file; the layer is for mail whose text is bland and whose payload is the file. A QR-code check is built (branch `qr-check`) but off: phishing_pot holds no QR codes, so its benefit could not be measured.
- The spoof head is **phone-augmented on purpose even though its studio EER is worse** (0.78% → 1.07%). A clean-trained head scores real human voices at mean 0.841 once passed through a real PSTN channel — deployed on a phone line it would accuse nearly every caller. That is a sign flip, not a degradation.
- `speech_model.py` reached in-source AUC 1.000 and real-call AUC 0.576 because every positive was ASR output and every negative was crowd-written text. It learned "Whisper transcript = scam". Never train scam-vs-legit across different transcription provenance.
- The 24.5% robocall figure is a **catch rate, not an accuracy** — that corpus has no benign half — and it is measured on monologues, so it does not test the rolling escalation or speaker separation. It is also in-source for `act_model`, which was trained on LLM labels over this corpus. An earlier README quoted 79.5%; that predates the shipped call-level act layer and no longer describes the model.
- Email 3-class accuracy on the temporal split is only 0.846 because spam, not phishing, is missed (spam-or-phishing recall 0.747). That is the spam ceiling the fusion row lifts.
- On live two-party calls, `peak_risk` is a maximum over windows, so a 17-minute call gets ~500 chances to cross the band and a 53-word robocall gets ~7. Length-normalised, per-window separation against real bank calls is far weaker than the call-level figure suggests. Never quote the call-level number alone.

## Status

Phases 1–2 closed (per-channel models, behaviour engine, fusion). Voice channel and the web dashboard are the active work. SMS behaviour rules and SPF/DKIM/Received parsing ship unmeasured — no public corpus carries a sender field or full headers.
