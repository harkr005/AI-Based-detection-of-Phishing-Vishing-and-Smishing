# PROJECT CONTEXT PROMPT — AI-Based Detection of Phishing, Vishing, and Smishing Attacks Using NLP and Behavioural Intelligence

You are assisting a 4-person final-year B.E. (Information Science) team building the
project described below. Use this as authoritative context for all answers. When I ask
for code, designs, diagrams, or report text, stay consistent with these choices.
If something here conflicts with what I ask, flag it rather than silently deviating.

---

## 1. PROJECT IDENTITY

- Title: AI-Based Detection of Phishing, Vishing, and Smishing Attacks Using NLP and
  Behavioural Intelligence
- Institution: BMS Institute of Technology and Management, Bengaluru
- Team: 4 members. Phase 1 review completed; currently executing Phase 2.
- Goal: One unified system that detects social-engineering attacks across three
  channels — EMAIL (phishing), SMS (smishing), VOICE (vishing) — combining semantic
  NLP with behavioural anomaly detection, and produces an explainable verdict in
  real time.

## 2. CORE THESIS (the claim we defend)

Content changes with every campaign; behaviour does not. Existing work does email OR
SMS OR voice, and is almost entirely content-based. We combine all three channels
into one fusion layer and add a behavioural intelligence engine that carries the
decision when content is unfamiliar. Detection runs on-device where possible, so
message content never leaves the user's phone — only the verdict does.

## 3. SYSTEM ARCHITECTURE

```
INGESTION
  Email :     (fallback: IMAP IDLE)
  SMS   : Android BroadcastReceiver (SMS_RECEIVED) -> WorkManager
  Voice : call audio -> 3s sliding windows, 1s overlap

PREPROCESS FORK
  text clean | URL extract | header/metadata parse | ASR (voice) | audio frames

PARALLEL MODELS
  NLP model      (DistilBERT / MobileBERT)
  URL model      (XGBoost on lexical + host features)
  Audio model    (wav2vec2-XLS-R + AASIST-style spoof head)

BEHAVIOURAL INTELLIGENCE ENGINE  <- the project's differentiator
  per-sender / per-caller profiles, deviation scoring (Isolation Forest + z-score)

FUSION / RISK SCORER
  adaptive weighted sum, plus a Logistic Regression stacker as comparison
  output: SAFE (0-0.35) | SUSPICIOUS (0.35-0.70) | PHISHING (0.70-1.0) + reason strings

OUTPUT
  Android notification + web dashboard (WebSocket live feed) + explanation
```

## 4. MODELS PER CHANNEL

| Channel | Primary model | Benchmark/comparison | Why |
|---|---|---|---|
| Email | DistilBERT fine-tuned, ONNX export | RoBERTa, BERT+LSTM | speed/accuracy balance |
| SMS   | MobileBERT, TFLite INT8, on-device | DistilBERT | ~0.39s, ~74MB, no GPU |
| Voice (semantic) | faster-whisper ASR -> DistilBERT intent head | — | real-time transcript scoring |
| Voice (spoof) | wav2vec2-XLS-R front-end + spoof head | RawNet2, AASIST | synthetic-voice signal |
| URL (shared) | XGBoost | RF, stacked ensemble | fast, explainable via SHAP |
| Behaviour | Isolation Forest + rolling z-score | rule baseline | unsupervised, zero-day resistant |
| Novelty | Autoencoder on embeddings (reconstruction error) | cosine distance to centroids | flags unseen patterns |

Explainability: SHAP (XGBoost), attention/LIME (transformers), rule tags (behaviour).
Every verdict must emit a human-readable reason string.

## 5. DATASETS

EMAIL
  - Nazario Phishing Corpus + Enron (legitimate) — standard balanced blend
  - Kaggle Phishing Email Dataset, SpamAssassin, CEAS-2008

SMS
  - UCI SMS Spam Collection (5,574) — baseline only; spam != smishing
  - Mendeley SMS Phishing Dataset — labelled ham / spam / smishing (3-class ground truth)
  - SmishTank community-sourced smishing dataset — freshest real samples
  - SELF-COLLECTED: Indian code-mixed (Hindi / Kannada / Hinglish) smishing,
    consent-recorded and anonymised. This is a genuine research gap.
  - Forum mining: r/Scams, r/india, Twitter screenshots -> OCR -> labelled set

URL
  - PhiUSIIL (UCI id=967): 235,795 URLs, 54 features, 100,945 phishing / 134,850 legit
  - PhishTank + OpenPhish (live feed), Tranco top-1M (benign)

VOICE
  - ASVspoof 2019 / 2021 / 5 for spoof detection (2021 includes telephone-channel audio)
  - Fake-or-Real (FoR), Mozilla Common Voice for real speech
  - TRANSCRIPTS ARE THE REAL GAP: no public vishing transcript corpus exists.
    Mitigation: author 300-500 scripted scam transcripts covering Indian scam types
    (bank OTP, KYC update, digital arrest, lottery, tech support, courier/customs,
    electricity disconnection), recorded by team members. Document the synthetic
    method honestly; prior work legitimises synthetic generation for minority classes.

SPLIT PROTOCOL: temporal split (train on older, test on newer), NOT random split.
Random splits leak future campaigns and inflate accuracy.

## 6. HOW NLP WORKS — PER CHANNEL

EMAIL
  Input: subject + body (cleaned, max_len 256).
  Signals learned: urgency framing, authority impersonation, credential/financial
  requests, deceptive phrasing, structural anomalies, persuasion principles.
  Auxiliary labels: train on persuasion type (authority / urgency / scarcity / fear /
  reward), not just binary. This transfers to unseen lures and improves reason strings.
  Robustness: trained with obfuscation augmentation (see section 9).

SMS
  Input: full message (short, avg ~120 chars) — text alone is weak, so URL score
  carries heavy weight. Intent-analysis framing: classify ham / spam / smishing.
  Runs fully on-device (TFLite INT8), never leaves the phone.
  Thread-level scoring: score the conversation with a rolling window, not each SMS
  in isolation — conversational scams start innocent and turn malicious later.

VOICE
  Input: streaming ASR transcript from 3s sliding windows with 1s overlap.
  Same DistilBERT intent head as email, plus a keyword/NER layer for high-risk
  entities (OTP, CVV, Aadhaar, PAN, "account blocked", "police", "arrest warrant",
  "remote access", "gift card").
  Rolling risk = exponential moving average across windows. Alert requires the score
  to stay above threshold for 2 consecutive windows — prevents single-word false alarms.
  Scam calls escalate over time; the rising risk curve is itself the signal.

## 7. HOW BEHAVIOURAL ANALYTICS WORKS — PER CHANNEL

Principle: build a rolling profile per sender/caller; score deviation from it.
Store profiles in Redis (server) / Room DB (on-device). Output a single
behaviour_anomaly_score in [0,1].

EMAIL behavioural features
  - sender vs reply-to domain mismatch
  - display name vs actual domain mismatch
  - SPF / DKIM / DMARC result
  - first-contact flag (no prior history with this sender)
  - send-hour deviation from that sender's historical baseline
  - Received-header path / timezone anomaly (useful when other fields are spoofed)
  - stylometry drift: avg sentence length, punctuation habits, greeting style
  - attachment type/frequency vs that sender's norm

SMS behavioural features
  - sender ID format vs Indian DLT header convention (e.g. AX-HDFCBK)
  - 10-digit personal number claiming to be an institution  <- very strong signal
  - first-time sender flag
  - URL present in a first-ever message from this sender
  - message burst rate from one sender
  - odd-hour delivery vs that sender's baseline
  - known-contact skip rule: do not flag saved contacts (kills false positives)

VOICE behavioural features
  - unknown / spoofed / withheld caller ID
  - caller ID country or circle mismatch vs claimed identity
  - time-to-first-sensitive-ask (scam calls ask early)
  - speech rate spike, interruption rate, pressure/repetition patterns
  - call duration vs typical for that caller
  - cross-channel link: an SMS from a related identity within 24h before/after the call

CROSS-CHANNEL CORRELATION (novel — nobody publishes this)
  Link events by sender identity across channels. A smishing SMS followed by a call
  from a related number within 24h escalates the risk score of BOTH events.

## 8. FUSION LOGIC

Adaptive weighting — trust structure when content is uncertain:

```python
if nlp_confidence < 0.60:
    weights = { nlp: 0.25, url: 0.30, behaviour: 0.45 }
else:
    weights = { nlp: 0.50, url: 0.30, behaviour: 0.20 }
```

Voice fusion: 0.65 * semantic + 0.20 * synthetic_voice + 0.15 * behaviour.
Compare weighted-sum vs a trained Logistic Regression stacker; report both.
Uncertainty: use prediction entropy or MC-dropout variance. High variance routes to
SUSPICIOUS — never to SAFE.

## 9. HANDLING NEW / UNSEEN ATTACKS

1. Semantic + persuasion-principle training  -> generalises to new phrasing
2. Behaviour layer                           -> zero-day resistant by construction
3. Autoencoder novelty detector on embeddings-> flags "unlike anything seen"
4. Wide SUSPICIOUS band + uncertainty routing-> never silently marks novel as safe
5. Campaign clustering (embeddings + SimHash)-> one report labels a whole campaign;
   push signatures to devices in minutes (hot tier)
6. Incremental learning on new labelled batches (warm, days);
   full fine-tune monthly (cold)
7. Drift monitoring: mean confidence trend, % in SUSPICIOUS band, autoencoder
   reconstruction error, embedding distribution shift (PSI/MMD), user FP/FN feedback.
   Breach -> event-driven retrain; otherwise scheduled.
8. Active learning: label only uncertain + novel samples, not everything.
9. Adversarial augmentation in training: character swaps, homoglyphs, whitespace
   injection, zero-width chars, synonym substitution, LLM paraphrase.
10. QR/quishing: decode QR in image attachments -> feed URL to existing URL engine.
11. Optional LLM escalation tier for the ~5% SUSPICIOUS band only (cost/latency).

## 10. DEPLOYMENT MODES

MODE 1 — ON-DEVICE (primary contribution)
  Android APK, RECEIVE_SMS runtime permission, TFLite INT8 model.
  QR-based pairing to the web dashboard: website shows single-use token (120s TTL,
  Redis) -> app scans -> server mints device JWT -> app POSTs verdicts.
  Only verdict + scores + reasons cross the wire. Privacy modes:
    Private (default): verdict only
    Preview: + first ~40 chars, PII-redacted
    Contribute (opt-in): + full anonymised body for retraining
  PII redaction runs on-device before serialization (OTP, account no., Aadhaar, PAN,
  amounts, emails).
  Offline: detection always works; events queue in Room (synced=false) and flush via
  WorkManager when connectivity returns.

MODE 2 — GATEWAY (install-free demo + honeypot)
  Provisioned number (MSG91/Kaleyra/Exotel preferred in India; Twilio elsewhere).
  Inbound webhook -> validate signature -> dedup on MessageSid -> redact -> same
  model stack server-side (ONNX) -> Postgres -> WebSocket to dashboard.
  Optional TwiML auto-reply with the verdict.
  Doubles as a honeypot for fresh in-the-wild samples.

Both modes emit an IDENTICAL event schema with a `source` field, so one dashboard,
one table, one feed. Framing: one system, two deployment profiles.

## 15. HOW TO RESPOND TO ME

- Use structured, point-based answers in simple language.
- Include ASCII or SVG diagrams where they help.
- Give real-world examples (prefer Indian scam scenarios: KYC, OTP, digital arrest,
  courier/customs, electricity disconnection).
- For debugging, give a dry run of the logic rather than just corrected code.
- Keep responses concise; I will ask for expansion where I need it.
- If I propose something that contradicts this plan or is technically impossible,
  say so plainly and explain why.
