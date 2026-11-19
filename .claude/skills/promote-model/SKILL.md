---
name: promote-model
description: Retrain, evaluate, and promote the email or SMS model, then re-export the phone's copy and prove parity. User-invoked only - it overwrites shipped weights.
disable-model-invocation: true
---

# Promote a model

Argument: `email` or `sms`. If none is given, ask which one.

**The weights come from private mail and SMS.** Never push, upload, publish or
demo them, and never print message text in a report: rates and counts only.

## Email (DistilBERT, `models/email_distilbert*`)

1. **Train under a NEW name.** `_train_unified.py` refuses to overwrite an existing
   checkpoint on purpose:
   `python scratch/_train_unified.py --out email_distilbert_vN`
   (N = the next free number; `ls models`). It takes a long time, so run it in the background.
2. **Evaluate against the shipped model, on the same slices.**
   `python scratch/_inbox_eval.py`, and `python scratch/_parser_fix_eval.py` if
   the parser changed. Print every `model.pt`'s md5 first: `models/email_distilbert` is a
   byte-copy of whichever version was last promoted, so "shipped" and "vN-1" can
   be the same file and show up as two identical columns.
3. **Stop and show the user the table.** They decide whether to promote. Never promote on
   your own reading of the numbers.
4. **Promote (only after the user says yes).** Copy `models/email_distilbert_vN/*` over
   `models/email_distilbert/`. The protection hook blocks edits under `models/`
   on purpose, so ask the user to run the copy command themselves.
5. Record the new md5 in the project-status memory, marking the old note superseded.

## SMS (char TF-IDF + LogReg, `models/sms_charlr.pkl`)

1. `python sms_model.py` retrains in place (in-source and cross-source reports
   plus sanity asserts). Note the numbers from before and after.
2. `python scratch/_inbox_verdicts.py` and `python scratch/_smish_verdicts.py`
   print both sides of the trade: false alarms on real legitimate SMS against
   smishing missed. Show both to the user.
3. **Re-export for the phone:** `python export_android.py`. This rewrites
   `android/app/src/main/assets/*.bin` and the parity fixtures. Skip it and the phone
   silently keeps scoring with the old model.
4. **Prove parity:** `cd android && ./gradlew testDebugUnitTest`.
   `ScanParityTest` and `SmsModelTest` must pass.
5. Stop and let the user review the diff before anything is committed.

## Never

- Retrain as a side effect of "running the tests" (see `/selfcheck`).
- Train on Gmail's own SPAM labels. They're for evaluation only, since training
  on them would just be imitating Gmail.
