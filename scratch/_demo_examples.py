"""Demo/test examples for the web UI, scored against the live sidecar.

Prints what the system ACTUALLY says next to what it SHOULD say, so a demo
never walks into a known failure by accident. Run with api.py up.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

SMS = [
    # --- should be SAFE -------------------------------------------------
    ("SAFE", "VM-HDFCBK", "Dear Customer, 520614 is your SECRET OTP to authenticate your login to Internet Banking. Do not share it with anyone."),
    ("SAFE", "AD-BESCOM", "Dear Consumer, your BESCOM bill for July 2026 of Rs.1,842.00 is generated. Due date 28-08-2026. RR No 4471290."),
    ("SAFE", "JD-AMAZON", "Your Amazon.in order 407-2298814 has been shipped. Expected delivery: 22 Aug 2026."),
    ("SAFE", "AX-JIOSMS", "Recharge of Rs.299 for 98XXXXXX78 is successful. Your plan is valid till 19-09-2026."),
    ("SAFE", "VM-IRCTCI", "PNR 2841009117 CNF. Train 12658 SBC-MAS on 24-08-2026, Coach B4, Berth 32 LB. Happy journey."),
    ("SAFE", "AD-MANIPL", "Your appointment with Dr. Mehta is confirmed for 21-08-2026 at 5:30 PM, Manipal Hospital."),
    ("SAFE", "", "Running late, stuck in traffic near Silk Board. Start without me."),
    ("SAFE", "", "Ya! when are you taking your practical lessons? I start in June."),
    # --- should be FLAGGED ----------------------------------------------
    ("FLAG", "9812345678", "Dear Customer, your SBI KYC is incomplete. Account will be blocked today. Share your OTP to verify at sbi-kyc-verify.tk/update"),
    ("FLAG", "9004471290", "Your electricity connection will be disconnected tonight at 9:30 PM as your previous bill was not updated. Contact officer 8829104471 immediately."),
    ("FLAG", "7679046492", "Dear customer your PAYTM KYC has been expired, It will be blocked within 24 hour, please contact customer care immediately."),
    ("FLAG", "8291047712", "Congratulations! You have won Rs.25,00,000 in the KBC lucky draw. To claim, pay processing fee of Rs.6,500 and send your Aadhaar and bank details."),
    ("FLAG", "9928104471", "Your parcel is held at customs. Pay clearance charge of Rs.1,299 at courier-clear-in.xyz/pay within 12 hours or it will be returned."),
    ("FLAG", "VM-HDFCBNK", "Dear Customer, your account will be blocked. Please share your OTP to complete KYC verification immediately."),
    ("FLAG", "9445102284", "This is CBI. A case has been registered against your Aadhaar for money laundering. Do not tell anyone. Transfer Rs.50,000 to the verification account now."),
    ("FLAG", "8841029917", "Sir I am from Amazon delivery. To confirm your order please install AnyDesk and share the code on your screen."),
]

EMAIL = [
    ("SAFE", "GitHub <noreply@github.com>", "A new public key was added to your account",
     "A new SSH key was added to your GitHub account. If you did not add this key, you can review your account settings."),
    ("SAFE", "Google Calendar <calendar-notification@google.com>", "Reminder: Phase 3 review",
     "This is a reminder that Phase 3 review starts in 30 minutes at 11:00 AM. Location: Seminar Hall 2."),
    ("SAFE", "Uber Receipts <receipts@uber.com>", "Your Wednesday trip with Uber",
     "Thanks for riding. Total Rs.284.00 charged to your card ending 6791. Trip from Indiranagar to BMSIT, 24 minutes."),
    ("SAFE", "Priya <priya.n@bmsit.in>", "notes from today",
     "Hey, attached are the minutes from the review. Action items are highlighted, add anything I missed before Friday."),
    ("SAFE", "Springer <alerts@springernature.com>", "New articles matching your saved search",
     "6 new articles match your saved search 'smishing detection'. View them in your saved searches on SpringerLink."),
    ("FLAG", "HDFC Bank Security <alerts@hdfc-secure-in.tk>", "Immediate action required: account suspension",
     "We detected unusual sign-in activity. Your net banking access will be suspended within 24 hours unless you re-verify. Verify now: http://hdfc-secure-in.tk/verify?id=8842 Failure to act will result in permanent closure."),
    ("FLAG", "Microsoft 365 <no-reply@micros0ft-secure.xyz>", "Your password expires today",
     "Your Microsoft 365 password expires today. To keep your mailbox active, confirm your credentials at http://micros0ft-secure.xyz/login within 2 hours or access will be revoked."),
    ("FLAG", "Accounts Payable <finance@vendor-invoices-in.top>", "Overdue invoice INV-88291",
     "Please find the overdue invoice attached. Payment of Rs.4,82,000 must be remitted today to our updated bank account. Kindly confirm once the transfer is initiated."),
    ("FLAG", "Barrister Kwame <barr.kwame88@gmail.com>", "URGENT AND CONFIDENTIAL",
     "I am contacting you regarding the estate of a deceased client who shares your surname. The sum of USD 8.5 million can be released to you as next of kin. Kindly send your bank details and a copy of your identification. Keep this strictly confidential."),
    ("FLAG", "IT Helpdesk <support@bmsit-helpdesk.online>", "Mailbox storage full - action needed",
     "Your mailbox has exceeded its storage quota and outgoing mail has been disabled. Re-validate your account credentials here to restore service: http://bmsit-helpdesk.online/webmail"),
]


def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))


def main():
    ok = bad = 0
    print("SMS" + " " * 60 + "expect  actual      risk")
    print("-" * 96)
    for want, sender, text in SMS:
        d = post("/sms", {"text": text, "sender": sender})
        got = "SAFE" if d["verdict"] == "SAFE" else "FLAG"
        mark = "ok " if got == want else "MISS"
        ok, bad = ok + (got == want), bad + (got != want)
        print(f"{mark} [{sender or 'no sender':11s}] {text[:52]:52s} {want:6s} {d['verdict']:10s} {d['risk']:.3f}")

    print("\nEMAIL" + " " * 58 + "expect  actual      risk")
    print("-" * 96)
    for want, sender, subj, body in EMAIL:
        d = post("/email", {"sender": sender, "subject": subj, "body": body})
        got = "SAFE" if d["verdict"] == "SAFE" else "FLAG"
        mark = "ok " if got == want else "MISS"
        ok, bad = ok + (got == want), bad + (got != want)
        print(f"{mark} {subj[:44]:44s} {sender[:22]:22s} {want:6s} {d['verdict']:10s} {d['risk']:.3f}")

    print(f"\n{ok} correct, {bad} wrong")


if __name__ == "__main__":
    main()
