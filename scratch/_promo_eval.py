"""Can the model tell an Indian marketing SMS from an Indian smishing SMS?

Scores the previous checkpoint against the retrained one on messages NEITHER
has been trained on. Every brand here is deliberately ABSENT from _gen_promo.py
and every phrasing is hand-written, so a model that merely memorised the 51
generated templates scores no better than it did before - which is the point.

PROMO must land spam. SMISH must land smishing. HAM must land ham.
"""
import data  # noqa: F401  # before torch

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

import sms_model
from sms_model import LABELS, OUT

PREV = OUT.parent / "sms_mobilebert_prev"

# The message that started this: a real Subway coupon, scored smishing 0.631.
SUBWAY = ("Last 2 days of your Subway Offer: 30% off on 299. Hurry! Visit Now. "
          "Code : xec5wyenna. Valid till 19 Aug.TnC. Ignore if already redeemed")

# Legitimate bulk marketing. Brands outside the generator's list on purpose.
PROMO = [
    SUBWAY,
    "Meesho Mega Sale! Kurtis from Rs 199, sarees from Rs 299. Free delivery. Shop now https://meesho.com/sale",
    "ZEPTO: Get Rs 150 off on your first 3 orders this week. Use code ZEPTO150. Order in 10 mins!",
    "boAt Monsoon Sale is LIVE. Flat 60% off on earbuds + extra 10% on prepaid. Visit https://boat-lifestyle.com",
    "Tata 1mg: Flat 25% off on medicines + free doorstep delivery. Order before 30-Sep at https://1mg.com",
    "CRED: Pay your credit card bill and win cashback up to Rs 1000. Offer ends tonight. Open the CRED app",
    "Cleartrip Big Sale: domestic flights from Rs 1399. Use code TRIPLOW for extra Rs 500 off. Book now",
    "Zivame End of Season Sale - Buy 2 Get 2 Free on all lingerie. Hurry, limited stock! https://zivame.com",
    "FirstCry: Extra 30% off on baby products above Rs 999. Code BABY30. Valid till Sunday. T&C apply",
    "Physics Wallah: Admissions open for JEE 2027 batch. Get 40% scholarship. Register at https://pw.live",
    "Practo Care: Book a full body health checkup at 50% off. Free home sample collection. Book now",
    "Shoppers Stop First Citizen: your 2500 reward points expire on 12-Nov. Redeem in store or online",
    "Groww: Start investing in mutual funds from Rs 100. Zero commission on direct plans. Download the app",
    "Wow! Momo: Buy any 2 momo plates and get 1 free this weekend at all outlets. Show this SMS to redeem",
    "Snapdeal Diwali Dhamaka - up to 80% off on home decor. Click here https://snapdeal.com/diwali T&C apply",
    "Pepperfry: Flat Rs 5000 off on sofas above Rs 29999 + no cost EMI. Visit your nearest studio till 20-Oct",
    "Vedantu: Free live masterclass on Physics this Saturday 6 PM. Limited seats, register at https://vedantu.com",
    "Chaayos: Your favourite chai at 20% off all week. Use code CHAI20 on the app. Valid till 25-Aug",
    "Yatra.com: Book any hotel and get flat 35% off with code STAY35. Offer ends in 24 hrs. Hurry!",
    "Slice: Get your credit line upgraded up to Rs 2 lakh. No annual fee. Apply on the slice app",
]

# Fraud. These must NOT move: a promo class that costs smishing recall is a
# regression dressed up as a fix.
SMISH = [
    "Dear Customer, your SBI account will be blocked today. Update KYC immediately at http://sbi-kyc.tk",
    "Your electricity connection will be disconnected tonight 9:30 PM as your previous bill is not updated. Contact 9876543210",
    "Congratulations! You have won Rs 12,50,000 in the KBC lucky draw. Call 8877665544 to claim your prize",
    "Your parcel is on hold at customs due to incomplete address. Update at http://india-post-track.vip within 24 hrs",
    "Income Tax refund of Rs 15,490 is approved. Verify your bank account at http://itr-refund-verify.xyz",
    "Dear user, your PAN card is not linked to Aadhaar. Your account is suspended. Click http://pan-link-now.cc",
    "e-Challan alert: fine of Rs 1500 pending on vehicle KA05MH2233. Pay now at http://echallan-pay.top or face legal action",
    "Your HDFC net banking will expire today. Re-activate at http://hdfcbank.com.secure-login.tk immediately",
    "Dear Customer, we could not deliver your order due to an invalid address. Reconfirm here http://bit.ly/3xk9del",
    "Your mobile number will be disconnected in 2 hours as documents are not verified. Call 7788990011 now",
    "URGENT: unauthorised transaction of Rs 49,999 detected on your card. If not you, verify at http://card-block.win",
    "You have a pending cashback of Rs 2500 in your wallet. Claim within 1 hour: https://rewardclaim.vip/in",
]

# Real transactional service SMS. Must stay ham - the class the promo rows sit
# next to, and the one last session drove to 0% false positives.
HAM = [
    "Rs.2,500.00 debited from A/c XX4471 on 21-Aug-26 to VPA rahul@okaxis. Not you? Call 18002586161 -HDFCBK",
    "OTP for your Flipkart order is 447281. Do not share it with anyone. -FLPKRT",
    "Your Airtel bill of Rs 649 for Aug 2026 is generated. Due date 05-Sep-26. Pay on the Airtel Thanks app -AIRTEL",
    "Dear Consumer, power supply to your premises will be affected on 24-Aug from 10 AM to 2 PM for maintenance -BESCOM",
    "Your train PNR 4728193021 is CONFIRMED, coach B4 seat 32, 22-Aug 06:15 SBC-MAS -IRCTCI",
    "Your appointment with Dr. Sharma is confirmed for 23-Aug at 5:30 PM. Reply CANCEL to cancel -APOLLO",
]


def load_from(path):
    return (AutoTokenizer.from_pretrained(path),
            AutoModelForSequenceClassification.from_pretrained(path).to(sms_model.DEV))


def rate(tok, model, texts, want):
    """-> (hit rate, per-message predicted label + probability)"""
    p = sms_model.predict(tok, model, texts)
    k = p.argmax(1)
    return (k == LABELS.index(want)).mean(), [(LABELS[i], float(p[r, i]))
                                              for r, i in enumerate(k)]


def main():
    models = [("before", PREV)] if PREV.exists() else []
    models.append(("after", OUT))

    print(f"{'':8s} {'promo -> spam':>15s} {'smish -> smishing':>19s} {'ham -> ham':>12s}")
    detail = {}
    for name, path in models:
        tok, model = load_from(path)
        r = {}
        for want, texts in (("spam", PROMO), ("smishing", SMISH), ("ham", HAM)):
            r[want], detail[name, want] = rate(tok, model, texts, want)
        print(f"{name:8s} {r['spam']*100:14.0f}% {r['smishing']*100:18.0f}% "
              f"{r['ham']*100:11.0f}%")
        del model
        torch.cuda.empty_cache()

    for want, texts in (("spam", PROMO), ("smishing", SMISH), ("ham", HAM)):
        miss = [(t, detail[models[-1][0], want][i])
                for i, t in enumerate(texts)
                if detail[models[-1][0], want][i][0] != want]
        print(f"\n--- {want}: {len(miss)}/{len(texts)} still wrong ---")
        for t, (lab, p) in miss:
            print(f"  {lab:8s} {p:.2f}  {t[:88]}")

    print("\n--- the message that started this ---")
    for name, path in models:
        tok, model = load_from(path)
        p = sms_model.predict(tok, model, [SUBWAY])[0]
        print(f"  {name:8s} " + "  ".join(f"{l}={v:.3f}" for l, v in zip(LABELS, p)))
        del model
        torch.cuda.empty_cache()

    tok, model = load_from(OUT)
    got, _ = rate(tok, model, PROMO, "spam")
    got_s, _ = rate(tok, model, SMISH, "smishing")
    assert got_s >= 0.9, f"smishing recall collapsed to {got_s:.0%} - the fix cost more than it bought"
    assert got >= 0.7, f"only {got:.0%} of held-out promos read as spam"


if __name__ == "__main__":
    main()
    print("\nok")
