"""Build the multilingual training corpus for the scam-intent classifier.

The corpus is *synthetic*: sentences are composed from slot-filled templates
that reproduce the scripts documented in I4C / RBI / MHA advisories and news
reports of Indian scam calls, in English, Devanagari Hindi and Hinglish.
Benign data deliberately includes hard negatives: genuine bank SMS footers,
police awareness messages, news about scams, delivery updates and ordinary
work / family calls that share vocabulary with scams.

Usage:  python scripts/build_corpus.py  ->  data/scam_corpus.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "scam_corpus.jsonl"

S = {
    "agency": ["CBI", "Mumbai Police", "Delhi Police cyber cell", "Narcotics Control Bureau", "Enforcement Directorate",
               "customs department", "TRAI", "crime branch", "RBI", "income tax department"],
    "agency_hi": ["सीबीआई", "मुंबई पुलिस", "साइबर क्राइम", "कस्टम विभाग", "प्रवर्तन निदेशालय", "ट्राई"],
    "agency_hl": ["CBI", "Mumbai police", "crime branch", "cyber cell", "customs", "TRAI"],
    "courier": ["FedEx", "DHL", "Blue Dart", "a courier company"],
    "contraband": ["MDMA", "drugs", "five fake passports", "illegal items", "140 grams of narcotics"],
    "amount": ["2 lakh", "50,000", "9 lakh 80 thousand", "1.5 crore", "25,000", "4,999", "3 lakh"],
    "hours": ["2 hours", "two hours", "24 hours", "30 minutes"],
    "app": ["AnyDesk", "TeamViewer", "QuickSupport", "RustDesk", "UltraViewer"],
    "bank": ["SBI", "HDFC", "ICICI", "your bank", "Axis Bank", "PNB"],
    "name": ["Rahul", "Priya", "Amit", "Sunita", "Arjun", "Neha", "Vikram", "Meera"],
    "city": ["Mumbai", "Delhi", "Kanpur", "Nagpur", "Bengaluru", "Lucknow", "Pune"],
    "product": ["iPhone", "laptop", "saree", "headphones", "groceries", "books"],
}

SCAM = {
    "digital_arrest": [
        "I am calling from {agency}. A parcel booked in your name contains {contraband}.",
        "Your Aadhaar card has been used to open a bank account for money laundering.",
        "There is an arrest warrant against you. You are under digital arrest.",
        "Do not tell anyone about this call, this is a confidential national matter.",
        "Stay on the video call and keep your camera on until the investigation is complete.",
        "To prove your innocence transfer {amount} to the RBI verification account.",
        "Your money will be returned after verification, do not disconnect the call.",
        "This is officer from {agency}, your name has come up in a money laundering case.",
        "Go to a room alone and lock the door, the Supreme Court has issued an order.",
        "If you disconnect we will send the local police to arrest you within {hours}.",
        "मैं {agency_hi} से बोल रहा हूँ, आपके नाम से पार्सल में ड्रग्स मिले हैं।",
        "आपके आधार का इस्तेमाल मनी लॉन्ड्रिंग में हुआ है, आप डिजिटल अरेस्ट में हैं।",
        "किसी को मत बताना, कॉल मत काटना, वरना गिरफ्तार कर लिए जाओगे।",
        "जांच के बाद पैसे वापस मिल जाएंगे, अभी {amount} सुरक्षित खाते में ट्रांसफर करें।",
        "main {agency_hl} se bol raha hoon, aapke naam se parcel mein drugs mile hain",
        "aapka aadhaar money laundering case mein use hua hai, aap digital arrest mein ho",
        "kisi ko mat batana aur call mat kaatna warna giraftar ho jaoge",
        "verification ke liye {amount} paise transfer karo, baad mein refund ho jayega",
        "camera on rakho aur akele kamre mein jao, case darj ho chuka hai",
    ],
    "tech_support": [
        "Your computer is infected with a virus. Call Microsoft support toll free now.",
        "Windows has been blocked due to suspicious activity. Do not restart your computer.",
        "Access to this PC has been blocked for security reasons. Contact support immediately.",
        "I am from Microsoft technical support, please install {app} so I can fix the virus.",
        "Your firewall has detected spyware, your personal data is at risk, error code 0x80070424.",
        "Please download {app} and give me the code shown on the screen.",
        "Now open your net banking so I can process the refund for our service.",
        "आपका कंप्यूटर ब्लॉक हो गया है, वायरस मिला है, तुरंत टोल फ्री नंबर पर कॉल करें।",
        "aapke computer mein virus hai, {app} download karo main theek kar deta hoon",
        "screen share karo, main aapka refund process kar deta hoon",
    ],
    "kyc_bank": [
        "Dear customer your {bank} KYC has expired, your account will be blocked today.",
        "Your PAN card is not linked, update KYC immediately or account will be suspended.",
        "Please share the OTP you just received to complete your KYC verification.",
        "Tell me your card number, expiry date and CVV to stop the unauthorised transaction.",
        "I am calling from {bank} KYC department, your account will be frozen in {hours}.",
        "आपका केवाईसी खत्म हो गया है, खाता बंद हो जाएगा, ओटीपी बताइए।",
        "{bank} se bol raha hoon, KYC update karna hai, OTP bata do",
        "aapka account block ho jayega, abhi card ka number aur pin batao",
    ],
    "refund_upi": [
        "You have a refund of {amount}, enter your UPI PIN to receive the money.",
        "I have sent a collect request, just enter the PIN to receive the cashback.",
        "Scan this QR code and the refund will be credited to your account.",
        "I sent you extra money by mistake, please return it by scanning the QR.",
        "पैसे पाने के लिए पिन डालिए, रिफंड तुरंत आ जाएगा।",
        "refund ke liye QR scan karo aur UPI pin daalo, paise aa jayenge",
        "pin daaloge to paise aayenge, cashback ka request bheja hai",
    ],
    "telecom_utility": [
        "This is TRAI, your SIM will be deactivated within {hours} due to illegal activity.",
        "Your mobile number will be blocked, press 9 to speak to the officer.",
        "Dear consumer your electricity connection will be disconnected tonight at 9:30 pm, bill not updated.",
        "Call the electricity officer immediately to avoid power cut today.",
        "आपका सिम दो घंटे में बंद हो जाएगा, तुरंत बात करें।",
        "आज रात बिजली कट जाएगी, बिल बकाया है, इस नंबर पर तुरंत कॉल करें।",
        "aapka sim band ho jayega 2 ghante mein, 9 dabaiye",
        "bijli ka connection kat jayega aaj raat, bill pending hai, abhi payment karo",
    ],
    "investment_job": [
        "Join our trading group for guaranteed returns of 30 percent every week.",
        "Get IPO allotment through our institutional account, pay {amount} to book.",
        "Part time job: like YouTube videos and earn 5000 daily, first prepaid task is {amount}.",
        "Your earnings are locked, pay the tax of {amount} to withdraw your profit.",
        "Double your money in 15 days with our crypto plan, assured profit.",
        "घर बैठे कमाई, पक्का मुनाफ़ा, पहले {amount} जमा करें।",
        "ghar baithe kamai karo, youtube like karo, pehle task ke liye paise bhejo",
        "pakka munafa hai sir, trading group join karo, paise double",
    ],
    "lottery_prize": [
        "Congratulations you have won 25 lakh in the KBC lucky draw.",
        "To claim your prize money pay the processing fee of {amount}.",
        "Your number has won a lottery, pay tax to release the cash prize.",
        "आपकी लॉटरी लगी है, इनाम के लिए {amount} फीस जमा करें।",
        "aapki KBC lottery lagi hai, inaam lene ke liye registration fee bharo",
    ],
}

BENIGN = [
    # everyday calls
    "Hi {name}, are we still meeting at 5 in {city}?", "Mom said dinner is ready, come home soon.",
    "The project review is moved to Thursday afternoon.", "Can you send me the slides before the meeting?",
    "Your {product} order has been shipped and will arrive tomorrow.", "Please join the team call on Teams at 3 pm.",
    "I will transfer your share of the dinner bill on UPI tonight.", "Let us book the train tickets for Diwali.",
    "The doctor appointment is confirmed for Monday morning.", "Did you finish the assignment for the aerospace lab?",
    "Traffic is bad near the station, I will be late by ten minutes.", "Happy birthday {name}, have a great year!",
    "The electricity bill for this month is 1,240 rupees, I paid it on the app.",
    "My flight from {city} lands at 7, can you pick me up?", "Please review the pull request when you are free.",
    "The internet is slow today, let us restart the router.", "Grandpa wants to video call on Sunday.",
    "The courier guy called, he will deliver the {product} after lunch.", "Our quarterly results look good.",
    # hard negatives (scam vocabulary, benign intent)
    "Do not share your OTP with anyone. {bank} will never ask for your PIN.",
    "Police never arrest anyone over a video call, beware of digital arrest scams.",
    "News: Mumbai police arrested a gang running fake call centres in {city}.",
    "RBI announced a new repo rate today, EMIs may change.",
    "Your OTP for login is 482913. It is valid for 10 minutes. Do not share it with anyone.",
    "I installed TeamViewer so the IT team at office can fix my work laptop.",
    "Awareness session: report cyber fraud by calling 1930 or at cybercrime.gov.in.",
    "The customs clearance for our company shipment is done, invoice attached.",
    "My KYC update at the bank branch is done, they checked my Aadhaar in person.",
    "This is a scam alert from your bank: never share card details on calls.",
    "The judge adjourned the hearing to next month, said the lawyer.",
    "I got a refund of the cancelled ticket in my account.",
    "The antivirus finished a scan and found no threats.",
    "पुलिस कभी वीडियो कॉल पर गिरफ्तार नहीं करती, सावधान रहें।",
    "बैंक कभी ओटीपी नहीं माँगते, किसी को ओटीपी न बताएं।",
    "कल शाम को घर पर पूजा है, जल्दी आना।", "मीटिंग तीन बजे है, स्लाइड भेज देना।",
    "आपका पार्सल कल तक पहुँच जाएगा।", "बिजली का बिल ऐप से भर दिया है।",
    "kal milte hain {city} mein, chai peete hain", "bhai assignment bhej de jaldi",
    "mummy ka phone aaya tha, ghar kab aa rahe ho", "OTP kisi ko mat batana, bank kabhi nahi maangta",
    "office ka laptop IT wale anydesk se theek kar rahe hain", "exam kal hai, padhai kar raha hoon",
    "order deliver ho gaya, thank you", "train late hai 2 ghante, main pahunch ke call karta hoon",
]


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, vals in S.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(vals), 1)
    return out


def augment(text: str, rng: random.Random) -> str:
    """Simulate ASR/OCR noise: drop punctuation, casing, filler words, typos."""
    t = text
    if rng.random() < 0.4:
        t = t.lower()
    if rng.random() < 0.3:
        t = t.replace(",", "").replace(".", "")
    if rng.random() < 0.25:
        t = rng.choice(["sir ", "madam ", "hello ", "ji ", "dekhiye ", "listen "]) + t
    if rng.random() < 0.15 and len(t) > 12:
        i = rng.randrange(1, len(t) - 1)
        t = t[:i] + t[i + 1:]
    return t


def build(n_per_template: int = 12, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for fam, temps in SCAM.items():
        for tpl in temps:
            for _ in range(n_per_template):
                rows.append({"text": augment(fill(tpl, rng), rng), "label": fam})
    # multi-sentence scam turns (how ASR segments actually look)
    for fam, temps in SCAM.items():
        for _ in range(60):
            k = rng.randint(2, 3)
            rows.append({"text": " ".join(fill(t, rng) for t in rng.sample(temps, k=min(k, len(temps)))), "label": fam})
    for tpl in BENIGN:
        for _ in range(n_per_template * 2):
            rows.append({"text": augment(fill(tpl, rng), rng), "label": "benign"})
    for _ in range(300):
        rows.append({"text": " ".join(fill(t, rng) for t in rng.sample(BENIGN, 2)), "label": "benign"})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"wrote {len(rows)} rows -> {OUT}")
    print(Counter(r["label"] for r in rows))
