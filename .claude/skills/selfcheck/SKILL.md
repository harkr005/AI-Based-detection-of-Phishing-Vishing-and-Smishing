---
name: selfcheck
description: Run the self-checks that cover what changed - web lib/*.test.mjs, the Python modules' __main__ asserts and scratch/*_check.py, and the Android Gradle unit tests. Use after editing code, before committing, or when asked to "run the tests".
---

# Self-check

This repo has no single test runner. Each part checks itself its own way. Pick
the checks by what changed (`git status --short` plus `git diff --name-only HEAD`),
run them, and report each one as passed, failed (with its output), or skipped (and why).

## Web (`web/`): run from `web/`

The tests are plain Node scripts, not a framework. They need the env file because
most of them hit the real Postgres:

```bash
node --env-file=.env.local lib/<name>.test.mjs
```

| Changed | Run |
|---|---|
| `lib/hash.js`, `lib/password.js` | `hash.test.mjs` |
| `lib/gmail.js`, `app/api/gmail/*`, `lib/db.js`, `lib/auth.js` | `gmail.test.mjs` |

- `gmail.test.mjs` fakes Google and the sidecar, and polls ONLY its own test user
  (`pollOne`). Never change it to call `pollAll`: that once expired a real
  account's Gmail link.
- "skipped message poison: Model server 422" in its output is expected: that's the
  poison-message case being exercised.
- Route files that no test imports: syntax-check with
  `node --input-type=module --check < app/api/.../route.js`.
- Lint: `node node_modules/eslint/bin/eslint.js <files>`. `hooks/use-mobile.js`
  has one known error that predates this setup; ignore it unless you touched it.

## Python (repo root)

Fast module checks (asserts under `if __name__ == "__main__"`):

| Module | Time | Notes |
|---|---|---|
| `python behaviour.py` | ~1 s | DLT registry, header trust, sender features |
| `python thread.py` | ~2 s | URL extraction, thread ratchet |
| `python store.py` | ~3 s | uses a `_selftest` user in Postgres, cleans up |
| `python voice_model.py` | ~30 s | negation / act rules |
| `python fusion.py` | ~50 s | loads models |
| `python data.py` | ~50 s | loads the corpora |
| `python scratch/_mail_parser_check.py` | ~1 min | `data._pot_body` / `readable_body` |

**Never run these as a "check", because they RETRAIN and overwrite models:**
`sms_model.py`, `url_model.py`, `email_model.py`, `act_model.py`,
`spoof_model.py`, and anything `scratch/_train_*.py`. That's `/promote-model`'s job.

## Android (`android/`)

```bash
cd android && ./gradlew testDebugUnitTest --console=plain -q
```

It takes minutes, so run it in the background. Read results from
`app/build/test-results/testDebugUnitTest/*.xml` (count `failures=` / `errors=`).
`ScanParityTest` compares the Kotlin port against Python's own numbers in
`src/test/resources/*_parity.json`. If it fails after a Python change, the fix is
usually `python export_android.py`, not editing Kotlin.

## Report

One line per check: name, pass/fail, time. Paste the failing output in full.
Never call something passed without having run it.
