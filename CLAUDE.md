# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scam detector across email, SMS and voice: per-channel content models, plus a
behavioural engine (sender/caller profiles), fused into one explainable verdict.
It's a final-year B.E. project. `PROJECT_SPEC.md` is the design document, and code
comments cite it by section (`§7`, `§10`). README's "Caveats" section explains
why several obvious features are deliberately withheld; read it before
"improving" a model.

Three programs share one Postgres:

- **Python sidecar** (repo root): FastAPI in `api.py`, the only place models run
  for web traffic. Next.js can't import torch, so every web route proxies here.
- **Web** (`web/`): Next.js 16 / React 19 dashboard. It owns the database schema,
  auth, Gmail polling and the APK download page (`/phone`).
- **Android** (`android/`): Kotlin app. It scores SMS **fully on-device** with
  exported copies of the Python models. It has no pairing, no upload path and no
  INTERNET permission: nothing it scans ever reaches the website.

## Commands

```bash
python api.py                          # sidecar on 127.0.0.1:8000. No --reload: models load once via lru_cache
cd web && npm run dev                  # dashboard on :3000
cd web && npm run lint                 # eslint (hooks/use-mobile.js has one pre-existing error)
cd android && ./gradlew testDebugUnitTest --console=plain -q   # minutes; results in app/build/test-results/
```

Tests are self-checking scripts, not a framework. `/selfcheck` maps changed files
to the right ones.

- **Web**, run from `web/`: `node --env-file=.env.local lib/<name>.test.mjs`
  (`hash`, `gmail`). `gmail` hits the real Postgres.
  `gmail.test.mjs` must only ever `pollOne` its own test user, never `pollAll`.
- **Python**: `python behaviour.py | thread.py | store.py | voice_model.py | fusion.py | data.py`
  run asserts under `__main__`. Plus `python scratch/_mail_parser_check.py` and
  `python scratch/_url_mail_check.py`.
- **Careful: `__main__` RETRAINS and overwrites weights** in `sms_model.py`,
  `url_model.py`, `email_model.py`, `act_model.py`, `spoof_model.py` and
  `scratch/_train_*.py`. They are not tests. Retraining goes through `/promote-model`.
- **Android single test:** `./gradlew testDebugUnitTest --tests com.sentinel.app.RedactTest`

On Windows, **`import data` must come before `import torch`** in any Python entry
point (a libiomp conflict segfaults `pd.read_csv`). Every module already does this; keep it.

## Architecture

**Verdict pipeline (`api.py`)**: channel model(s) → `behaviour.py` deviation
features from the per-sender profile (`store.py`, Postgres table
`sender_profiles`) → `fusion.py` adaptive weighted sum plus escalation gate →
SAFE < 0.35 ≤ SUSPICIOUS < 0.70 ≤ PHISHING, with reason strings. Scoring a
message also *updates* the sender profile, so never score the same message twice.
That's why `hasScan` is checked before scoring and `/email/parse` parses without scoring.

- SMS: `sms_model.py` is char 3-5 TF-IDF + LogReg (`models/sms_charlr.pkl`), not
  MobileBERT (that was an experiment, kept for comparison). Plus `url_model.py`
  (XGBoost, domain-lexical features only) and `thread.py` (`find_urls`, the one
  URL extractor every path shares).
- Email: DistilBERT 3-class in `email_model.py`. `models/email_distilbert` is a
  byte-copy of whichever `email_distilbert_vN` was last promoted.
  `data._pot_body` is the text every email model reads; `readable_body` is for display only.
  `_pot_body` drops hrefs, so link checks read `data.link_hosts` instead.
  Email links are scored by a separate MAIL link model (`url_model.score(...,
  mail=True)`, `models/url_xgb_v4.json`): the Public Suffix List
  (`Datasets/URL/public_suffix_list.dat`) splits each host, the registrable
  domain gets the usual 15 features and the part in front gets `sub_*` ones.
  Two mail-only layers on top: brand mismatch (`url_model.brand_mismatch`,
  brands from Tranco top-1000 + the registry) and the DMARC-passing,
  top-1M sender's own links scored by domain alone (`sender_owner`). The brand
  list reads Tranco top-1M once per process (~7 s, first email after a restart).
  SMS, `thread.py` and the phone stay on `url_xgb`. Missing file or PSL -> email
  falls back to `url_xgb`. Check: `python scratch/_url_mail_check.py`.
- Gmail-only email layers (headers exist only on `/email/raw`; pasted mail
  abstains): `behaviour.auth_fail` (SPF/DKIM/DMARC), the repeat-sender trust
  damp (`behaviour.trusted` + `fusion.trust_damp`, never armed by an off-domain
  link), and crowd reports (`store.crowd`: distinct users' `is_a_threat`
  feedback on Gmail scans; model verdicts never count). On connect,
  `gmail.js seedProfiles` folds metadata-only headers into the profiles via
  `/email/seed`. Numbers: `scratch/_auth_eval.py`, `scratch/_trust_eval.py`.
- Every sidecar link (pasted SMS, pasted mail, Gmail) also goes through Google
  Safe Browsing v5 (`safe_browsing.py`): only 4-byte SHA-256 prefixes leave the
  machine, a listed link is raised to 0.95, an unlisted one is left alone, and
  no key or no network abstains. Website only; the phone stays offline.
  Check: `python safe_browsing.py` (live test against Google's test URLs).
- Voice: `voice_model.py` rule primitives with an asymmetric EMA ratchet,
  `act_model.py`, and `spoof_model.py`. Caller number (optional, typed on the
  voice page): `behaviour.voice_features` - a bank/financial claim from a
  non-1600 number (TRAI 2026), withheld, international, and a risky SMS from
  the same number (`store.risky_sms`). No number = abstain. Unmeasured: no
  corpus has caller numbers; the claim's reach is in `scratch/_caller_claim_eval.py`.
- Unknown metadata abstains rather than defaulting to a neutral value. The scan
  clock is not the message clock: `sent_at` comes from the message's own `Date:`
  header or the SMSC timestamp.
- Header trust (`behaviour.registered`) requires the carrier prefix
  (`AX-HDFCBK-S`) *and* an entity from the TRAI registry file. Nothing is
  hand-typed into the trust list.

**Web (`web/`)**
- `lib/db.js` holds the whole schema as one idempotent `SCHEMA` string, run on
  first pool use. Migrations are `ALTER … IF NOT EXISTS` lines appended there.
  There's no ORM; all queries live in `db.js`.
- `lib/auth.js`: an HMAC session cookie, nothing else. Phone pairing was removed
  on 2026-10-28; an old `device_codes` table may still exist in the database and
  is unused.
- `scans.source` is `web` | `gmail` (older rows may say `app`, from the phone
  before pairing was removed). Gmail has its own page (`/gmail`) and the dashboard
  list excludes it. `(user_id, device_id, message_hash)` has a partial unique
  index, which makes Gmail ingestion idempotent.
- `/phone` is a static download page for `public/sentinel.apk`, which is
  gitignored: copy `android/app/build/outputs/apk/debug/app-debug.apk` there by
  hand after a build.
- Gmail is **polled, not pushed**: `instrumentation.js` starts `lib/gmail.js
  pollAll` every 60 s (history.list). Refresh tokens are AES-GCM sealed under
  `AUTH_SECRET`, so rotating that secret signs everyone out *and* expires every
  Gmail link. Mail bodies are never stored; View fetches them live.
- Env (`web/.env.local`): `DATABASE_URL`, `AUTH_SECRET`, `SIDECAR_URL`,
  `GEMINI_API_KEY` (OCR), `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` (Gmail),
  `GOOGLE_SAFE_BROWSING` (API key). The sidecar reads `DATABASE_URL` and
  `GOOGLE_SAFE_BROWSING` from here too (`store.env`).

**Android (`android/app/src/main/java/com/sentinel/app/`)**
- `SmsReceiver` → `MessageStore` (Room-backed) → `Classifier` → `LocalScan`
  (`SmsModel`, `UrlModel`, `Behaviour`, `Fusion`, all Kotlin ports). Results stay
  in Room (`ScanDb`); nothing is uploaded.
- The Kotlin ports must match Python exactly. After any retrain, run
  `python export_android.py`: it writes `assets/*.bin` and the parity fixtures
  that `ScanParityTest` checks. If parity fails, the fix is usually re-exporting,
  not editing Kotlin.
- The only thing that can leave the phone is the Audit report, and only when the
  user shares it; its text and `terms` go through `Redact` first.

## Data and weights

`Datasets/`, `models/` and `papers/` are gitignored. **Some weights and datasets
come from private mailboxes and a private SMS inbox** (`Datasets/personal/`,
email v2+, the SMS model since the inbox retrain, `url_xgb_v4` - trained on
mail link hosts). Never publish them, and never
print message text in eval output: report rates and counts. Two tracked files
come from that inbox: `android/.../assets/sms_charlr.bin`, and the parity fixtures
`export_android.py` writes (30 corpus rows sampled with `load_sms`, already
redacted). The GitHub remote is private. Regenerate both without the inbox
before the repo is ever made public. `scratch/` holds the
eval and training scripts behind every quoted number; their docstrings record
what each measurement means and its caveats.

## Claude Code setup in this repo

- `.claude/settings.json` hooks: edits to `.env*`, `models/` and Android
  `assets/*.bin` are blocked; web JS files get `eslint --fix` after each edit.
- `/selfcheck` runs the right checks for what changed. `/promote-model` (user-only)
  runs retrain → evaluate → export → parity.
- The `privacy-reviewer` agent reviews anything touching the upload path,
  redaction, `/api/scans`, Gmail, or reports and logs.
