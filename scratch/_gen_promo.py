"""Generate the promotional-SMS spam class -> Dataset(Indian)/sms/promo_sms.csv

WHY THIS FILE EXISTS. Every Indian corpus here supplies ham (transactional
service SMS) and smishing (fraud), and none supplies spam (bulk marketing). So
the SMS model learned "offer + urgency + CTA + link = smishing", because in
training that was true with no counter-examples: explicit CTAs split 92 smishing
/ 4 spam / 1 ham. A real Subway coupon scored smishing 0.631; deleting "Visit
Now." alone moved it to spam 0.999.

THIS IS SYNTHETIC and must be declared as such in the report - the same standing
as india_fraud_detection_FINAL.csv, which is also template-generated. It is
authored to be ADVERSARIAL TO THE FAILURE MODE: every row carries the features
the model currently reads as fraud - urgency, a CTA, a code, a rupee amount, a
link - over a benign ask. A promo set without those teaches nothing, because
those are exactly the tokens in dispute.

The discriminator being taught is the ASK, not the packaging:
  spam     buy / redeem / shop / book        -> commercial, ignorable
  smishing verify / unblock / pay fine / OTP -> credential or coerced payment

Bank, wallet and loan promos are included on purpose: they share vocabulary with
the scams they sit next to, so they are the rows that carry the boundary.
"""
import random

import pandas as pd

from data import INDIA

# (brand, real domain, DLT-style header, sector). Deliberately overlaps the
# brands that appear in the smishing rows - if brand identity separated the
# classes, the model would learn a brand list instead of the shape of a sales
# pitch. The sector tag exists so a template only draws a brand that could
# plausibly have sent it: an unsectored draw produced "Start a SIP with IRCTC"
# and "Free demo class at Blinkit", text no Indian phone has ever received.
BRANDS = [
    ("Flipkart", "flipkart.com", "FLPKRT", "shop"),
    ("Amazon", "amazon.in", "AMAZON", "shop"),
    ("Myntra", "myntra.com", "MYNTRA", "shop"),
    ("AJIO", "ajio.com", "AJIOIN", "shop"),
    ("Nykaa", "nykaa.com", "NYKAAA", "shop"),
    ("BigBasket", "bigbasket.com", "BIGBSK", "shop"),
    ("Blinkit", "blinkit.com", "BLNKIT", "shop"),
    ("Tata CLiQ", "tatacliq.com", "TATCLQ", "shop"),
    ("Croma", "croma.com", "CROMAA", "shop"),
    ("Reliance Digital", "reliancedigital.in", "RELDIG", "shop"),
    ("Lenskart", "lenskart.com", "LNSKRT", "shop"),
    ("Decathlon", "decathlon.in", "DECATH", "shop"),
    ("Bata", "bata.in", "BATAIN", "shop"),
    ("Titan", "titan.co.in", "TITANW", "shop"),
    ("Swiggy", "swiggy.com", "SWIGGY", "food"),
    ("Zomato", "zomato.com", "ZOMATO", "food"),
    ("Dominos", "dominos.co.in", "DOMINO", "food"),
    ("Subway", "subway.co.in", "SUBWAY", "food"),
    ("KFC", "kfc.co.in", "KFCIND", "food"),
    ("MakeMyTrip", "makemytrip.com", "MMTRIP", "travel"),
    ("Goibibo", "goibibo.com", "GOIBIB", "travel"),
    ("IRCTC", "irctc.co.in", "IRCTCI", "travel"),
    ("Ola", "olacabs.com", "OLACAB", "travel"),
    ("Uber", "uber.com", "UBERIN", "travel"),
    ("Airtel", "airtel.in", "AIRTEL", "telecom"),
    ("Jio", "jio.com", "JIOIND", "telecom"),
    ("Vi", "myvi.in", "VIINDA", "telecom"),
    ("Paytm", "paytm.com", "PAYTMM", "wallet"),
    ("PhonePe", "phonepe.com", "PHONPE", "wallet"),
    ("HDFC Bank", "hdfcbank.com", "HDFCBK", "bank"),
    ("ICICI Bank", "icicibank.com", "ICICIB", "bank"),
    ("Axis Bank", "axisbank.com", "AXISBK", "bank"),
    ("SBI Card", "sbicard.com", "SBICRD", "bank"),
    ("Kotak", "kotak.com", "KOTAKB", "bank"),
    ("Bajaj Finserv", "bajajfinserv.in", "BAJAJF", "bank"),
    ("PolicyBazaar", "policybazaar.com", "PLCYBZ", "insurance"),
    ("Apollo Pharmacy", "apollopharmacy.in", "APOLLO", "pharma"),
    ("PharmEasy", "pharmeasy.in", "PHRMSY", "pharma"),
    ("Netmeds", "netmeds.com", "NETMED", "pharma"),
    ("cult.fit", "cult.fit", "CULTFT", "fitness"),
    ("BookMyShow", "bookmyshow.com", "BKMYSH", "ott"),
    ("Hotstar", "hotstar.com", "HOTSTR", "ott"),
    ("Unacademy", "unacademy.com", "UNACAD", "edu"),
    ("Urban Company", "urbancompany.com", "URBNCO", "service"),
]

CATS = {
    "shop": ["fashion", "electronics", "groceries", "footwear", "beauty products",
             "kitchen appliances", "mobiles", "watches", "eyewear", "home decor",
             "sportswear", "luggage", "furniture", "toys"],
    "food": ["pizzas", "subs and wraps", "burgers", "desserts", "biryani",
             "combo meals", "beverages"],
    "pharma": ["medicines", "health supplements", "wellness products"],
    "shop_or_food": None,   # filled below
}
CATS["shop_or_food"] = CATS["shop"] + CATS["food"]
CTA = ["Visit Now", "Shop Now", "Order Now", "Buy Now", "Click here",
       "Book Now", "Grab Now", "Explore Now"]
URGE = ["Hurry", "Limited stock", "Today only", "Offer ends tonight",
        "Last chance", "While stocks last", "Ends in 24 hrs", "Do not miss out"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SALES = ["Big Billion Days", "End of Season Sale", "Republic Day Sale",
         "Great Indian Festival", "Monsoon Carnival", "Independence Day Sale",
         "Diwali Dhamaka", "Summer Clearance", "Anniversary Sale"]

# (sectors the template may draw a brand from, text). () = any sector.
# {b} brand  {d} domain  {h} DLT header  {p} percent  {rs}/{min}/{cb} rupees
# {code} coupon  {date} dd-Mon  {n} small int  {cat} category  {cta} {urge} {sale}
SHOPFOOD = ("shop", "food")
TEMPLATES = [
    # --- the exact shape that broke: last-days + CTA + code, no link ---
    (SHOPFOOD, "Last {n} days of your {b} Offer: {p}% off on {rs}. Hurry! {cta}. Code : {code}. Valid till {date}.TnC. Ignore if already redeemed"),
    ((), "Only {n} days left! Flat {p}% OFF at {b}. Use code {code}. {cta}. Offer valid till {date}. T&C apply"),
    (SHOPFOOD, "{urge}! Your {b} coupon of Rs {rs} expires on {date}. Apply code {code} at checkout. {cta}"),
    # --- discount + brand's own link (the link-phobia counter-example) ---
    (("shop",), "{b}: FLAT {p}% OFF on {cat}. {cta} https://{d}/sale Use code {code}. {urge}. T&C apply -{h}"),
    (("shop",), "Get {p}% off on {cat} at {b}. Shop at https://{d}/offers before {date}. {urge}!"),
    (("shop",), "{sale} is LIVE at {b}! Up to {p}% off across {cat}. {cta}: https://{d} -{h}"),
    (SHOPFOOD, "Rs {rs} OFF on orders above Rs {min} at {b}. Code {code}. {cta} https://{d}/app T&C apply"),
    (("shop",), "New arrivals in {cat} at {b}, starting Rs {rs}. {cta} https://{d}/new -{h}"),
    (("shop",), "{b} {sale}: extra {p}% off on already discounted {cat}. Visit https://{d}/deals. {urge}"),
    # --- food delivery ---
    (("food",), "{b}: Get {p}% OFF up to Rs {rs} on your next order. Use code {code}. {cta}. Valid till {date} -{h}"),
    (("food",), "Hungry? Flat Rs {rs} off on orders above Rs {min} on {b}. Code {code}. Order before {date}. T&C apply"),
    (("food",), "{b} weekend treat: buy 1 get 1 free on all {cat}. Code {code}. {urge}! {cta}"),
    (("food",), "Your favourite meal at {p}% off on {b} today. No coupon needed. {cta} https://{d}"),
    # --- wallet / cashback: shares vocabulary with reward scams on purpose ---
    (("wallet",), "{b}: Get Rs {cb} cashback on your next recharge of Rs {min} or more. Offer valid till {date}. T&C apply -{h}"),
    (("wallet",), "Flat Rs {cb} cashback on bill payments above Rs {min} via {b}. Use code {code}. {cta}"),
    (("wallet",), "Scratch and win up to Rs {cb} cashback on your next {b} transaction. Offer ends {date} -{h}"),
    (("wallet",), "{b} rewards: Rs {cb} cashback credited on UPI payments above Rs {min} this week. T&C apply"),
    # --- bank / card marketing: the closest legitimate neighbour of smishing ---
    (("bank",), "{b} Credit Card offer: {p}% instant discount up to Rs {rs} on {cat}. Valid till {date}. T&C apply -{h}"),
    (("bank",), "Enjoy {p}% off with your {b} Debit Card on {b2} this {sale}. Min spend Rs {min}. T&C apply -{h}"),
    (("bank",), "{b}: You are eligible for a pre-approved Personal Loan of up to Rs {lakh} lakh at {rate}% p.a. Apply at https://{d}/loan -{h}"),
    (("bank",), "Upgrade to a {b} Lifetime Free Credit Card. No joining fee. Apply now https://{d}/cards T&C apply -{h}"),
    (("bank",), "{b}: EMI offer - convert purchases above Rs {min} into No Cost EMI. Know more at https://{d}/emi -{h}"),
    (("bank",), "Your {b} card is eligible for a limit increase up to Rs {lakh} lakh. Check eligibility on the {b} app -{h}"),
    # --- insurance / investment marketing ---
    (("insurance",), "{b}: Compare term plans and save up to Rs {rs} on premium. Get a free quote at https://{d} T&C apply -{h}"),
    (("insurance",), "Renew your car insurance with {b} and get {p}% off on own damage premium. Renew at https://{d}/renew -{h}"),
    (("bank", "insurance"), "Start a SIP with {b} from just Rs {rs}/month. Mutual funds are subject to market risks. Invest at https://{d}"),
    # --- telecom recharge ---
    (("telecom",), "{b}: Recharge with Rs {rs} and get {n}GB/day data + unlimited calls for {days} days. Recharge at https://{d} -{h}"),
    (("telecom",), "Special offer for you: Rs {rs} pack now at Rs {min} on the {b} app. Valid till {date}. {cta} -{h}"),
    (("telecom",), "{b} Rs {rs} plan: {days} days validity, unlimited 5G data. Upgrade now on https://{d}/recharge -{h}"),
    # --- travel ---
    (("travel",), "{b} {sale}: flights from Rs {fare}. Use code {code} for extra Rs {rs} off. Book by {date}. {cta} https://{d}"),
    (("travel",), "Book hotels at up to {p}% off with {b}. Code {code}. {urge}! Visit https://{d}/hotels -{h}"),
    (("travel",), "{b}: Flat {p}% off on your next {n} rides. Code {code}. Valid till {date}. T&C apply -{h}"),
    # --- pharmacy / health / fitness ---
    (("pharma",), "{b}: Flat {p}% OFF + extra Rs {rs} off on medicines above Rs {min}. Code {code}. Order at https://{d} -{h}"),
    (("pharma",), "Free home sample collection + {p}% off on full body checkup at {b}. Book at https://{d}/labs {urge}"),
    (("fitness",), "{b} membership at {p}% off this {sale}. Join {n} centres near you. {cta} https://{d} -{h}"),
    # --- entertainment / subscription ---
    (("ott",), "{b}: Get {n} months free on your annual plan. Offer valid till {date}. Subscribe at https://{d} -{h}"),
    (("ott",), "Buy 1 Get 1 free on movie tickets this weekend with {b}. Code {code}. {cta} https://{d} T&C apply"),
    (("ott",), "Your {b} plan is ending on {date}. Renew now at {p}% off and keep watching. https://{d}/renew -{h}"),
    # --- education ---
    (("edu",), "{b}: Admissions open. Get up to {p}% scholarship on all courses. Enrol before {date}. Apply https://{d} -{h}"),
    (("edu",), "Free demo class at {b} this weekend. Limited seats. Register at https://{d}/demo {urge}!"),
    # --- referral / loyalty / cart ---
    ((), "Refer a friend to {b} and both get Rs {cb} off. Share your code {code}. T&C apply -{h}"),
    ((), "You have {pts} reward points on {b} expiring on {date}. Redeem at https://{d}/rewards -{h}"),
    (SHOPFOOD, "You left items in your {b} cart! Complete your order in {n} hrs and get {p}% off. Code {code}. {cta}"),
    (("shop",), "{b}: Your wishlist item is back in stock at {p}% off. {urge}! {cta} https://{d}"),
    # --- local / service ---
    (("service",), "{b} is now available in your area. Book salon and cleaning services at flat {p}% off. Code {code} -{h}"),
    (("shop",), "New {b} store opening near you on {date}! Flat {p}% off on {cat} for the first {n} days. {cta}"),
    # --- plain, no link, no code: promos that look almost transactional ---
    (("shop",), "{b}: Flat {p}% off on {cat} this weekend at all stores. Offer valid till {date}. T&C apply -{h}"),
    (SHOPFOOD, "Dear Customer, enjoy {p}% off on your next {b} purchase above Rs {min}. Show this SMS at the counter -{h}"),
    (("shop",), "{b} {sale} starts {date}. Up to {p}% off storewide. See you there! -{h}"),
    # --- with the DLT-mandated opt-out tail ---
    (("shop",), "{b}: Flat Rs {rs} off on {cat}. Code {code}. {cta} https://{d}. To opt out reply STOP to {h}"),
    (("shop", "pharma"), "Get {p}% off on {cat} at {b} till {date}. T&C apply. To unsubscribe reply STOP -{h}"),
]


def code(rng, n=10):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(n))


def render(rng, sectors, tpl):
    pool = [x for x in BRANDS if not sectors or x[3] in sectors]
    b, d, h, sec = rng.choice(pool)
    b2 = rng.choice([x for x in BRANDS if x[3] == "shop"])[0]
    c = code(rng, rng.choice([8, 9, 10]))
    return tpl.format(
        b=b, d=d, h=h, b2=b2,
        cat=rng.choice(CATS.get(sec, CATS["shop"])),
        p=rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80]),
        rs=rng.choice([49, 99, 149, 199, 249, 299, 399, 499, 599, 750, 999, 1499, 2000]),
        min=rng.choice([199, 299, 499, 599, 999, 1499, 1999, 2999, 4999]),
        cb=rng.choice([25, 50, 75, 100, 150, 200, 250, 500]),
        fare=rng.choice([1199, 1499, 1799, 1999, 2499, 2999]),
        lakh=rng.choice([2, 3, 5, 8, 10, 15, 20]),
        rate=rng.choice(["10.5", "10.99", "11.5", "12.75", "13.99"]),
        days=rng.choice([28, 56, 84, 90, 180, 365]),
        pts=rng.choice([250, 500, 1200, 2400, 3750]),
        n=rng.choice([2, 3, 4, 5, 7, 10]),
        code=c.upper() if rng.random() < .5 else c,
        date=f"{rng.randint(1, 28)}-{rng.choice(MONTHS)}",
        cta=rng.choice(CTA), urge=rng.choice(URGE), sale=rng.choice(SALES))


def build(target=750, seed=0):
    """-> DataFrame(text, label, source, group). group = template id, so the
    GroupShuffleSplit in sms_model keeps every rendering of one template on the
    same side of the split. Grouping on the opening words would not: these start
    with the brand, so one template would scatter across 44 groups and its twins
    would land on both sides.
    """
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < target:
        i = rng.randrange(len(TEMPLATES))
        t = render(rng, *TEMPLATES[i])
        if t in seen:
            continue
        seen.add(t)
        rows.append({"text": t, "label": "spam", "source": "india_promo",
                     "group": f"promo{i:03d}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    out = INDIA / "promo_sms.csv"
    d.to_csv(out, index=False)
    print(f"{out}  {len(d)} rows, {d.group.nunique()} templates, "
          f"{d.text.nunique()} unique texts")
    assert d.text.is_unique and d.group.nunique() == len(TEMPLATES)
    # the point of the corpus: it must carry the tokens the model misreads
    has_cta = d.text.str.contains("|".join(CTA), case=False).mean()
    has_url = d.text.str.contains("https://").mean()
    print(f"carries a CTA {has_cta:.0%}   carries a link {has_url:.0%}")
    assert has_cta > .3 and has_url > .3, "a promo set without CTAs/links teaches nothing"
    for t in d.text.sample(8, random_state=1):
        print(" ", t[:118])
