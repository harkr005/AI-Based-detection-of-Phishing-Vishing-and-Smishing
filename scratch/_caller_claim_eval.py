"""How often does a call CLAIM to be a bank / financial company?

behaviour.FIN_CLAIM is the half of the voice caller rule that reads the
transcript; the other half is the caller number (not 1600-series). No corpus
has caller numbers, so the rule itself cannot be measured. What CAN be measured
is its reach: the share of calls in which the claim fires at all, i.e. the most
the 1600 rule could ever catch, and how often it fires on legitimate talk.

Regex written BEFORE this ran and not tuned on it. Rates and counts only.

CAVEATS that travel with the numbers:
  - HarperValley calls are INBOUND (the customer rings the bank), so the 1600
    rule would never apply to them; their rate is how often ordinary bank talk
    trips the claim pattern, an upper bound on false alarms from an outbound
    bank call made from a non-1600 number (which TRAI forbids since 2026-01).
  - Scambait victims are mostly in the US/UK, so the 1600 rule (India) would not
    apply there either; the rate says how often scammers claim to be financial.
"""
import data
import behaviour as b

sets = {
    "authored scam": data.load_transcripts().query("label == 1").text,
    "authored legit": data.load_transcripts().query("label == 0").text,
    "scambait (live scam)": data.load_scambait().transcript,
    "FTC robocalls": data.load_robocalls().transcript,
    "HarperValley (legit bank)": data.load_harpervalley().transcript,
}
for name, texts in sets.items():
    hit = texts.map(lambda t: bool(b.FIN_CLAIM.search(str(t))))
    print(f"{name:28s} n={len(texts):5d}   financial claim fires {hit.mean():6.1%}")

# RESULT 2026-10-12: authored scam 16.0% / legit 20.0% (n=25 each), scambait
# 35.1% (465), FTC robocalls 3.8% (1,353), HarperValley 95.3% (1,446, inbound -
# "thank you for calling ... bank"). Reach is a third of live scam calls at
# most; ordinary bank talk trips the claim nearly always, so the rule is only
# as good as the 1600 mandate: a real bank calling from a 10-digit number gets
# flagged, which since 2026 is a bank breaking the rule, not a false alarm.
