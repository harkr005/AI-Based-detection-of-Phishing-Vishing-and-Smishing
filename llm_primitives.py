"""Zero-shot scam-ACT detection with a local LLM (the ACL-2026 primitive idea,
without the corpus that made every trained version fail).

WHY. Three trained approaches were tried on this channel and all three learned
PROVENANCE instead of intent:
  - `sms_model` on speech: real-call AUC 0.494. It reads transactional banking
    vocabulary as the alarm, so a legitimate bank call scores 0.994.
  - `speech_model` (FTC scam vs SGD legit): in-source AUC 1.000, real-call
    0.576 with every call at 0.999 - it learned "ASR transcript = scam".
  - Phrase mining over the same two corpora returns `press to`, `please press`,
    `this call is` - IVR and announcement structure, not scam acts.
The common cause is that no corpus of LEGITIMATE calls exists in the same
register as the scam calls, so any discriminative method finds the register.

A zero-shot judge has no training corpus, so it has no register to shortcut on.
It is asked only what the caller is DOING - acts, not topics - which is the one
thing that separates a real bank call from a scam impersonating one: both say
"account" and "verify", only one demands a code, a payment or secrecy.

ponytail: the LLM runs at 3B on 8 GB, and only on the voice channel. The ACL
paper uses an LLM to build data and ships a small detector; if this scores well
the same distillation is the upgrade path, not a bigger model at serve time.
"""
import json
import re

import data
import numpy as np
import torch

import os
MODEL = os.environ.get("LLM_PRIMITIVES_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# 4-bit lets a 7-8B model fit the 8 GB card. Sim & Kim (2506.06180) run
# Llama3-8B at 8-bit with LoRA r=8 and reach GPT-4o-comparable accuracy on
# their adversarial set, so size is worth testing before settling for 3B.
FOURBIT = os.environ.get("LLM_PRIMITIVES_4BIT", "") == "1"
CHUNK_WORDS = 250          # enough context to judge an act, ~6 chunks/call
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Acts, not topics. Each is something the CALLER DOES; none of them is a word a
# legitimate agent would need. "Mentions your account" is deliberately absent -
# that is exactly the cue that broke every trained version.
ACTS = {
    "credential": ("asks the person to reveal or type a one-time code, OTP, PIN, "
                   "CVV, password, full card number, or Social Security number", 0.65),
    "payment":    ("demands money: a transfer, a fee, cryptocurrency, or buying "
                   "gift cards / prepaid vouchers", 0.60),
    "remote":     ("asks to install an app or grant remote or screen access to "
                   "the person's device", 0.65),
    "threat":     ("threatens arrest, legal action, a fine, or immediate closure "
                   "or suspension of an account or service", 0.45),
    "secrecy":    ("tells the person to keep the call secret, not tell family or "
                   "bank staff, or not to hang up", 0.50),
    "urgency":    ("demands the person act within minutes or immediately or lose "
                   "something", 0.30),
    "prize":      ("claims an unrequested prize, refund, or compensation is owed", 0.35),
}
ORDER = list(ACTS)

PROMPT = """You are analysing a segment of a telephone call transcript.
Decide, for each listed behaviour, whether the CALLER actually does it in this segment.

Judge only what the caller DOES, not what the call is about. A legitimate bank,
delivery or support agent will discuss accounts, cards, payments and verification
- that alone is never one of these behaviours.

Behaviours:
{acts}

Transcript segment:
\"\"\"{text}\"\"\"

Reply with ONLY a JSON object, keys exactly {keys}, values true or false."""


def _model(_cache={}):
    if "m" not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        kw = {}
        if FOURBIT:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, device_map=DEV, **kw).eval()
        _cache["m"] = (tok, mdl)
    return _cache["m"]


def chunks(text, n=CHUNK_WORDS):
    w = text.split()
    if len(w) <= n:
        return [" ".join(w)] if w else []
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]


def detect(segments, max_new=64, bs=16, keys=None, criteria=False, spans=False,
           uncertain=False):
    """-> (len(segments), len(keys or ORDER)) boolean array, or an array of
    strings when spans=True (verbatim evidence quotes, "" where absent) or
    uncertain=True ("yes" / "no" / "unclear").

    Batched. Decoder-only generation needs LEFT padding, so the prompt stays
    flush against the generated answer; right padding silently produces garbage.

    spans=True is the ACL-2026 evidence LOCATOR. It exists so a transcript can be
    split into evidence and background: cut the quotes out and what is left is a
    negative from the SAME corpus, which is the only known cure for this
    channel's register shortcut. Quotes need room - pass max_new >= 128.

    uncertain=True adds the third answer from CHI 2025 (arXiv 2502.03964), whose
    live-call study found the same false alarms this project measures - identity
    verification, financial-information sharing and manufactured urgency are all
    things legitimate callers do. Their fix is to let the judge DECLINE. Only
    "yes" counts as fired, so an ambiguous segment costs a miss instead of a
    false alarm on a real bank call.
    """
    tok, mdl = _model()
    keys = list(keys or ORDER)
    if criteria:
        acts_txt = chr(10).join(f"{i+1}. {k}" + chr(10) + f"   {CRITERIA[k]}"
                                for i, k in enumerate(keys))
    else:
        acts_txt = chr(10).join(f"- {k}: {ACTS[k][0]}" for k in keys)
    text_out = spans or uncertain
    template = (PROMPT_SPAN if spans else PROMPT_UNC if uncertain
                else PROMPT_CRI if criteria else PROMPT)
    side, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = (np.full((len(segments), len(keys)), "", object) if text_out
           else np.zeros((len(segments), len(keys)), bool))
    try:
        for i in range(0, len(segments), bs):
            batch = segments[i:i + bs]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": template.format(
                    acts=acts_txt, keys=json.dumps(keys), text=seg)}],
                add_generation_prompt=True, tokenize=False) for seg in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(DEV)
            with torch.no_grad():
                gen = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for b in range(len(batch)):
                txt = tok.decode(gen[b, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                m = re.search(r"\{.*?\}", txt, re.S)
                if not m:
                    continue
                try:
                    got = json.loads(m.group())
                except json.JSONDecodeError:
                    continue
                for j, k in enumerate(keys):
                    v = got.get(k, "" if text_out else False)
                    out[i + b, j] = (str(v or "").strip().lower() if uncertain
                                     else str(v or "").strip() if spans else bool(v))
    finally:
        tok.padding_side = side
    return out


# Acts NO legitimate caller performs. Measured, not assumed: on 150 real recorded
# bank calls these three fire 0.0% of the time, while payment fires on 13.3% of
# them - MORE than on the scam half. So caution has to be applied per act, not to
# the judge as a whole.
CLEAN_ACTS = ("threat", "secrecy", "prize")


def call_risk(text, criteria=False, hybrid=True):
    """-> (risk in [0,1], acts fired). Weights summed and capped, the same shape
    as voice_model.rule_score, so §3's bands keep their meaning.

    hybrid=True asks the clean acts with the two-valued prompt and the rest with
    the three-valued one (CHI 2025's "unclear"). Measured on 150 HarperValleyBank
    legitimate calls + 150 FTC scam calls, one chunk each:

        prompt              scam caught   false alarms on real bank calls
        two-valued              22.7%              14.0%
        three-valued             6.7%               1.3%
        hybrid                  14.7%               0.0%

    The trade is deliberate and it is the same base-rate argument as the SMS act
    gate: a phone carries far more legitimate calls than scam ones, so 8 points of
    catch is worth 14 points of false alarms. Pass hybrid=False for the old
    behaviour.
    """
    segs = chunks(text)
    if not segs:
        return 0.0, []
    if not hybrid:
        hits = detect(segs, criteria=criteria).any(0)
        return act_risk(k for k, h in zip(ORDER, hits) if h)
    clean = [k for k in ORDER if k in CLEAN_ACTS]
    rest = [k for k in ORDER if k not in CLEAN_ACTS]
    fired = [k for k, h in zip(clean, detect(segs, keys=clean, criteria=True).any(0)) if h]
    unc = detect(segs, keys=rest, uncertain=True)
    fired += [k for k, h in zip(rest, (unc == "yes").any(0)) if h]
    return act_risk(fired)


# A scam pairs acts; a real business does a dual-use act on its own - payment on a
# collection call, identity checks on a bank call, urgency on a sales call (the
# ACL-2026 paper's "logical decision over primitives"). So with COMBINE a lone act
# outside CLEAN_ACTS counts half its weight; any second act, or a clean act,
# counts in full. Rule fixed 2026-10-24 before it was measured, ON since
# 2026-10-25 (`_voice_llm_eval.py`, shipped act model, off -> on):
#   27 real calls AUC .731 -> .850 | HVB B false alarms 0.3% -> 0.3%
#   held-out scambait >=.35 88.4 -> 84.2%, >=.70 69.9 -> 69.9% | Korean FSS 65.7 -> 61.9%
#   105 calls from 7 new scambait channels >=.35 96.2 -> 96.2%, >=.70 77.1 -> 77.1%
# n=27 is noisy; the direction, not the third decimal. False = the old plain sum.
COMBINE = True


def act_risk(fired):
    """-> (risk in [0,1], acts fired). Weights summed and capped, the same shape
    as voice_model.rule_score, so §3's bands keep their meaning. The one place
    acts become a score: the judge, act_model and voice_model all call it."""
    fired = list(fired)
    w = sum(ACTS[k][1] for k in fired)
    if COMBINE and len(fired) == 1 and fired[0] not in CLEAN_ACTS:
        w /= 2
    return float(min(w, 1.0)), fired


# Expert criteria, per Sim & Kim (arXiv 2506.06180): for small models, putting
# structured human knowledge in the prompt beats Chain-of-Thought, which they
# measured as actively harmful. Each act gets an explicit DOES-NOT-COUNT clause,
# because the measured failure of the bare-name prompt was precisely that
# payment, credential and urgency fire on legitimate collection, bank and sales
# calls - real businesses perform those acts too.
CRITERIA = {
    "credential": ("COUNTS: the caller asks the person to read out or type a one-time "
                   "code, OTP, PIN, CVV, full card number, or Social Security number. "
                   "DOES NOT COUNT: asking the person to confirm details the company "
                   "already holds (name, date of birth, address, last four digits), or "
                   "any verification when the person placed the call."),
    "payment":    ("COUNTS: the caller directs the person to send money, wire funds, buy "
                   "gift cards or prepaid vouchers, or pay a fee to release something. "
                   "DOES NOT COUNT: quoting a price for a service the person asked "
                   "about, taking normal payment, or discussing a genuine existing bill "
                   "or debt the person already owes."),
    "remote":     ("COUNTS: the caller asks the person to install software or an app so "
                   "the caller can see or control their device. DOES NOT COUNT: guiding "
                   "the person through an app or website they already use themselves."),
    "threat":     ("COUNTS: the caller threatens arrest, a warrant, police, a court case, "
                   "deportation, or immediate termination of an account or service, used "
                   "as leverage to make the person comply. DOES NOT COUNT: factually "
                   "stating an overdue balance or a normal consequence of non-payment."),
    "secrecy":    ("COUNTS: the caller tells the person not to tell family, bank staff or "
                   "police, or not to hang up or end the call. DOES NOT COUNT: routine "
                   "confidentiality or data-protection statements."),
    "urgency":    ("COUNTS: the caller insists the person must act within minutes or lose "
                   "money or be arrested. DOES NOT COUNT: a sales offer expiring, or "
                   "ordinary scheduling."),
    "prize":      ("COUNTS: the caller claims a prize, refund or compensation the person "
                   "never asked for. DOES NOT COUNT: a refund the person requested, or "
                   "a discount on something they enquired about."),
}

PROMPT_CRI = """You are a fraud analyst reviewing a segment of a telephone call transcript.

For each behaviour below, decide whether the CALLER actually does it in this segment.
Apply the criteria literally. Legitimate banks, delivery firms, debt collectors and
support agents routinely discuss accounts, cards, payments, refunds and verification -
subject matter alone is never evidence. Judge only the caller's actions.

{acts}

Transcript segment:
\"\"\"{text}\"\"\"

Reply with ONLY a JSON object, keys exactly {keys}, values true or false."""


# ACL-2026 evidence locator. Same criteria, but the answer is the caller's own
# words instead of a boolean, so the span can be CUT OUT of the transcript. The
# "copy verbatim" instruction is load-bearing: a paraphrase cannot be found in
# the text, so it cannot be removed, and the background negative silently keeps
# its evidence.
PROMPT_SPAN = '''You are a fraud analyst reviewing a segment of a telephone call transcript.

For each behaviour below, find the CALLER's own words that show it.

{acts}

Transcript segment:
"""{text}"""

Reply with ONLY a JSON object, keys exactly {keys}. Each value is the caller's
exact words showing that behaviour, COPIED VERBATIM from the segment above, or
an empty string "" if the caller does not do it. Never paraphrase, never explain.'''


# CHI 2025's third answer. The instruction to prefer "unclear" over "yes" is the
# whole mechanism: on this channel every measured false alarm came from a
# legitimate caller doing something a scammer also does, and a judge that must
# pick a side will pick the alarm.
PROMPT_UNC = '''You are a fraud analyst reviewing a segment of a telephone call transcript.

For each behaviour below, decide whether the CALLER actually does it in this segment.
Apply the criteria literally. Legitimate banks, delivery firms, debt collectors and
support agents routinely discuss accounts, cards, payments, refunds and verification -
subject matter alone is never evidence. Judge only the caller's actions.

{acts}

Transcript segment:
"""{text}"""

Answer "yes" only if the caller clearly does it, "no" if they clearly do not, and
"unclear" if the segment does not settle it - a partial call, an ambiguous phrase,
or something a legitimate caller could equally be doing. Prefer "unclear" over "yes"
when in doubt.

Reply with ONLY a JSON object, keys exactly {keys}, values "yes", "no" or "unclear".'''
