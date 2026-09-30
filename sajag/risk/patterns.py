"""Scam knowledge base: families, multilingual cues and advice.

Every cue belongs to a *tactic* (the psychological lever a scammer pulls) and
is written in three scripts that Indian callers actually use:

* English
* Hindi in Devanagari
* Hinglish (Hindi written in Latin script)

The scam families and tactics follow the typologies published by the Indian
Cyber Crime Coordination Centre (I4C), RBI's "BE(A)WARE" booklet and the
"digital arrest" advisories issued by the Ministry of Home Affairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Tactics: the building blocks every scam script is made of.
# weight = how much a single hit of this tactic raises suspicion on its own.
# --------------------------------------------------------------------------
TACTICS: dict[str, dict] = {
    "authority": {
        "label": "Impersonates an authority",
        "label_hi": "सरकारी अधिकारी होने का दावा",
        "weight": 0.18,
    },
    "threat": {
        "label": "Threatens arrest, penalty or blocking",
        "label_hi": "गिरफ्तारी या खाता बंद करने की धमकी",
        "weight": 0.22,
    },
    "urgency": {
        "label": "Creates artificial urgency",
        "label_hi": "जल्दबाज़ी का दबाव",
        "weight": 0.12,
    },
    "secrecy": {
        "label": "Demands secrecy or isolation",
        "label_hi": "किसी को न बताने का दबाव",
        "weight": 0.28,
    },
    "payment": {
        "label": "Asks you to move money",
        "label_hi": "पैसे भेजने को कहना",
        "weight": 0.26,
    },
    "credential": {
        "label": "Asks for OTP, PIN or card details",
        "label_hi": "ओटीपी / पिन / कार्ड की जानकारी माँगना",
        "weight": 0.32,
    },
    "remote_access": {
        "label": "Asks you to install a screen-sharing app",
        "label_hi": "स्क्रीन शेयर ऐप डाउनलोड करवाना",
        "weight": 0.34,
    },
    "lure": {
        "label": "Promises prizes, refunds or guaranteed returns",
        "label_hi": "इनाम या पक्का मुनाफ़ा का लालच",
        "weight": 0.16,
    },
    "fake_error": {
        "label": "Shows a fake virus / system error",
        "label_hi": "नकली वायरस चेतावनी",
        "weight": 0.24,
    },
    "identity_bait": {
        "label": "Claims your Aadhaar / SIM / parcel was misused",
        "label_hi": "आधार / सिम / पार्सल के दुरुपयोग का दावा",
        "weight": 0.2,
    },
}


@dataclass(frozen=True)
class Cue:
    tactic: str
    phrases: tuple[str, ...]
    families: tuple[str, ...] = ()
    boost: float = 0.0  # extra weight for very specific phrases


def _c(tactic: str, phrases: list[str], families: list[str] | None = None, boost: float = 0.0) -> Cue:
    return Cue(tactic, tuple(p.lower() for p in phrases), tuple(families or ()), boost)


# --------------------------------------------------------------------------
# Cue library. Latin phrases are matched on word boundaries after
# normalisation; Devanagari phrases are matched as substrings.
# --------------------------------------------------------------------------
CUES: list[Cue] = [
    # ---------------- authority ----------------
    _c("authority", [
        "cbi", "central bureau of investigation", "enforcement directorate", "ed officer",
        "narcotics control bureau", "ncb", "cyber crime branch", "cyber cell", "crime branch",
        "mumbai police", "delhi police", "police station", "inspector", "dcp", "ips officer",
        "customs officer", "customs department", "income tax department", "trai", "department of telecom",
        "dot officer", "rbi officer", "reserve bank", "sebi", "supreme court", "high court", "judge",
        "interpol", "fedex", "dhl", "blue dart", "microsoft support", "windows support",
        "apple support", "technical support", "bank manager", "kyc department", "electricity board",
        "bijli vibhag", "thana", "police wale", "cbi se", "crime branch se",
        "पुलिस", "सीबीआई", "थाना", "साइबर क्राइम", "प्रवर्तन निदेशालय", "कस्टम", "आरबीआई",
        "सुप्रीम कोर्ट", "जज", "इंस्पेक्टर", "बैंक मैनेजर", "बिजली विभाग", "ट्राई",
    ]),
    _c("authority", [
        "i am calling from cbi", "calling from mumbai police", "this is officer", "badge number",
        "case officer", "investigating officer", "main cbi se bol raha", "main police se bol raha",
        "मैं पुलिस से बोल रहा", "मैं सीबीआई से बोल रहा",
    ], ["digital_arrest"], boost=0.12),

    # ---------------- threat ----------------
    _c("threat", [
        "arrest warrant", "non bailable warrant", "you will be arrested", "arrest you", "fir registered",
        "fir has been registered", "case registered against you", "legal action", "jail", "court case",
        "money laundering", "hawala", "drug trafficking", "illegal parcel", "account will be blocked",
        "account will be frozen", "account suspended", "sim will be blocked", "sim will be deactivated",
        "connection will be disconnected", "power will be cut", "electricity will be disconnected",
        "penalty", "fine of", "blacklisted", "your number will be blocked",
        "giraftar", "giraftaar", "arrest kar", "arrest ho jaoge", "jail jaoge", "case darj",
        "account band", "account block", "sim band", "bijli kat", "connection kat",
        "गिरफ्तार", "वारंट", "जेल", "मनी लॉन्ड्रिंग", "केस दर्ज", "एफआईआर", "खाता बंद", "खाता ब्लॉक",
        "सिम बंद", "बिजली कट", "कनेक्शन कट", "कानूनी कार्रवाई", "जुर्माना",
    ]),
    _c("threat", [
        "digital arrest", "you are under digital arrest", "under investigation", "your aadhaar is linked",
        "डिजिटल अरेस्ट", "digital arrest mein",
    ], ["digital_arrest"], boost=0.25),

    # ---------------- urgency ----------------
    _c("urgency", [
        "immediately", "right now", "within 2 hours", "within two hours", "within 24 hours",
        "today itself", "before 6 pm", "last chance", "urgent", "hurry", "act now", "time is running out",
        "abhi", "turant", "jaldi", "aaj hi", "do ghante", "2 ghante",
        "तुरंत", "अभी", "जल्दी", "आज ही", "दो घंटे", "2 घंटे", "आखिरी मौका",
    ]),

    # ---------------- secrecy / isolation ----------------
    _c("secrecy", [
        "do not tell anyone", "don't tell anyone", "dont tell anyone", "do not inform", "don't inform your family",
        "keep this confidential", "this is confidential", "national secret", "do not disconnect",
        "don't disconnect", "do not cut the call", "stay on the call", "stay on video call", "keep your camera on",
        "go to a room alone", "lock the room", "do not contact your bank", "don't go to the police station",
        "kisi ko mat batana", "kisi ko nahi batana", "ghar walon ko mat batana", "call mat kaatna",
        "phone mat kaatna", "camera on rakho", "akele kamre mein",
        "किसी को मत बताना", "किसी को मत बताइए", "किसी को नहीं बताना", "कॉल मत काटना", "फोन मत काटना",
        "कैमरा चालू रखें", "गोपनीय", "अकेले कमरे में",
    ], ["digital_arrest"], boost=0.05),

    # ---------------- payment ----------------
    _c("payment", [
        "transfer the money", "transfer money", "transfer the amount", "send money", "pay now",
        "make the payment", "rtgs", "neft", "imps", "upi", "scan the qr", "scan this qr",
        "safe account", "secure account", "rbi account", "verification amount", "security deposit",
        "refundable deposit", "processing fee", "registration fee", "clearance fee", "tax to release",
        "gift card", "google play card", "amazon voucher", "bitcoin", "usdt", "crypto wallet",
        "paise transfer", "paisa bhejo", "paise bhejo", "payment karo", "fees bharo", "jama karo",
        "पैसे ट्रांसफर", "पैसे भेजो", "पैसे भेजिए", "भुगतान", "जमा करें", "फीस", "यूपीआई", "क्यूआर",
        "सुरक्षित खाता",
    ]),
    _c("payment", [
        "transfer all your money", "for verification we need to transfer", "money will be returned after verification",
        "funds will be refunded after investigation", "verification ke liye paise",
        "जांच के बाद पैसे वापस", "वेरिफिकेशन के लिए पैसे",
    ], ["digital_arrest"], boost=0.2),

    # ---------------- credential harvesting ----------------
    _c("credential", [
        "otp", "one time password", "tell me the otp", "share the otp", "read the otp", "upi pin",
        "enter your pin", "atm pin", "cvv", "card number", "expiry date", "net banking password",
        "login password", "aadhaar number", "pan number", "verify your identity with otp",
        "otp batao", "otp bata do", "pin batao", "pin daalo", "card ka number",
        "ओटीपी", "पिन बताइए", "पिन बताओ", "कार्ड नंबर", "सीवीवी", "पासवर्ड", "आधार नंबर",
    ], boost=0.05),
    _c("credential", [
        "enter pin to receive", "enter your upi pin to receive", "pin daaloge to paise aayenge",
        "पैसे पाने के लिए पिन",
    ], ["refund_upi"], boost=0.25),

    # ---------------- remote access ----------------
    _c("remote_access", [
        "anydesk", "any desk", "teamviewer", "team viewer", "rustdesk", "ultraviewer", "quicksupport",
        "quick support", "airdroid", "anyviewer", "supremo", "remote desktop", "screen share",
        "share your screen", "screen sharing", "install this app", "download this app", "apk file",
        "screen share karo", "app download karo", "स्क्रीन शेयर", "ऐप डाउनलोड", "एनीडेस्क",
    ], boost=0.05),

    # ---------------- lures ----------------
    _c("lure", [
        "you have won", "lottery", "lucky draw", "kbc", "prize money", "cash prize", "guaranteed returns",
        "double your money", "assured profit", "100 percent profit", "ipo allotment", "insider tip",
        "stock tips", "trading group", "part time job", "work from home job", "like youtube videos",
        "prepaid task", "task based", "refund of", "cashback", "loan approved", "instant loan", "pre approved loan",
        "inaam", "lottery lagi", "paise double", "pakka munafa", "ghar baithe kamai",
        "इनाम", "लॉटरी", "पक्का मुनाफ़ा", "पैसे डबल", "घर बैठे कमाई", "लोन मंज़ूर", "रिफंड",
    ]),

    # ---------------- fake error / scareware ----------------
    _c("fake_error", [
        "your computer is infected", "virus detected", "your pc is blocked", "windows has been blocked",
        "your computer has been locked", "trojan", "spyware alert", "security breach detected",
        "call microsoft", "call windows support", "toll free", "do not restart your computer",
        "do not shut down", "error code", "firewall warning", "your personal data is at risk",
        "pornographic virus", "access to this pc has been blocked", "contact support immediately",
        "आपका कंप्यूटर ब्लॉक", "वायरस",
    ], ["tech_support"], boost=0.05),

    # ---------------- identity bait ----------------
    _c("identity_bait", [
        "your aadhaar", "aadhaar card has been used", "your aadhaar is being misused", "sim card issued on your aadhaar",
        "parcel in your name", "parcel booked in your name", "package contains drugs", "mdma", "passports",
        "fake passports", "your name has come up", "your bank account is used", "kyc expired", "kyc update",
        "kyc pending", "pan card not linked", "electricity bill not updated", "bill unpaid",
        "aapke naam se parcel", "aapka aadhaar", "kyc update karna", "bill pending",
        "आपके नाम से पार्सल", "आपका आधार", "आपके आधार", "पार्सल में ड्रग्स", "केवाईसी", "बिल बकाया",
    ]),
]


# --------------------------------------------------------------------------
# Scam families: which tactics define them + what to tell the user.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    id: str
    name: str
    name_hi: str
    signature: tuple[str, ...]  # tactics whose co-occurrence is typical
    advice: str
    advice_hi: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


FAMILIES: dict[str, Family] = {f.id: f for f in [
    Family(
        "digital_arrest", "Digital arrest / fake police", "डिजिटल अरेस्ट / नकली पुलिस",
        ("authority", "threat", "secrecy", "payment", "identity_bait"),
        "No police, CBI or court ever arrests anyone over a video call or asks you to move money for 'verification'. "
        "Hang up, tell a family member, and call 1930 or report at cybercrime.gov.in.",
        "पुलिस, सीबीआई या कोर्ट कभी वीडियो कॉल पर गिरफ्तार नहीं करते और 'वेरिफिकेशन' के लिए पैसे नहीं माँगते। "
        "कॉल काटें, परिवार को बताएं और 1930 पर कॉल करें या cybercrime.gov.in पर शिकायत करें।",
        ("digital arrest", "cbi", "warrant", "money laundering", "narcotics", "parcel", "गिरफ्तार"),
    ),
    Family(
        "tech_support", "Fake tech support / scareware", "नकली टेक सपोर्ट",
        ("fake_error", "authority", "remote_access", "urgency", "payment"),
        "Microsoft, Apple and your bank never show pop-ups asking you to call a number. Do not call, do not install "
        "remote-access apps. Close the browser with Ctrl+Shift+Esc > End task.",
        "माइक्रोसॉफ्ट या बैंक कभी पॉप-अप में फोन नंबर देकर कॉल करने को नहीं कहते। कोई रिमोट ऐप इंस्टॉल न करें।",
        ("virus", "microsoft", "toll free", "blocked", "anydesk"),
    ),
    Family(
        "kyc_bank", "KYC / bank account block", "केवाईसी / बैंक खाता बंद",
        ("identity_bait", "threat", "urgency", "credential"),
        "Banks never ask for OTP, PIN or CVV on a call or link. Call the number printed on your card or passbook instead.",
        "बैंक कभी कॉल या लिंक पर ओटीपी, पिन या सीवीवी नहीं माँगते। कार्ड पर लिखे नंबर पर खुद कॉल करें।",
        ("kyc", "otp", "account blocked", "pan"),
    ),
    Family(
        "refund_upi", "UPI refund / collect-request trick", "यूपीआई रिफंड धोखा",
        ("lure", "credential", "payment"),
        "You never need to enter a UPI PIN or scan a QR code to RECEIVE money. Decline the request.",
        "पैसे पाने के लिए कभी यूपीआई पिन डालने या क्यूआर स्कैन करने की ज़रूरत नहीं होती।",
        ("refund", "upi pin", "qr", "cashback"),
    ),
    Family(
        "telecom_utility", "SIM / electricity disconnection", "सिम / बिजली कटने की धमकी",
        ("authority", "threat", "urgency", "payment"),
        "TRAI, DoT and electricity boards do not call to disconnect services. Verify on the official app or Sanchar Saathi.",
        "ट्राई, दूरसंचार विभाग या बिजली विभाग कॉल करके कनेक्शन काटने की धमकी नहीं देते।",
        ("sim", "trai", "electricity", "bijli"),
    ),
    Family(
        "investment_job", "Investment / task-job fraud", "निवेश / टास्क जॉब धोखा",
        ("lure", "payment", "urgency"),
        "Guaranteed returns and 'pay to unlock earnings' are always fraud. Check SEBI registration before investing.",
        "पक्का मुनाफ़ा या 'कमाई निकालने के लिए पैसे जमा करें' हमेशा धोखा है।",
        ("returns", "ipo", "task", "trading group", "part time"),
    ),
    Family(
        "lottery_prize", "Lottery / prize / KBC", "लॉटरी / इनाम धोखा",
        ("lure", "payment", "urgency"),
        "You cannot win a lottery you never entered. Never pay a 'processing fee' to claim a prize.",
        "जिस लॉटरी में आपने भाग नहीं लिया, उसे आप जीत नहीं सकते। इनाम के लिए कोई फीस न दें।",
        ("lottery", "kbc", "prize", "इनाम"),
    ),
]}

GENERIC_ADVICE = (
    "Pause. Do not share OTPs, PINs or screen access and do not transfer money while on this call. "
    "Verify through an official number you look up yourself. National cyber-fraud helpline: 1930."
)
GENERIC_ADVICE_HI = (
    "रुकिए। इस कॉल के दौरान ओटीपी, पिन या स्क्रीन एक्सेस साझा न करें और पैसे ट्रांसफर न करें। "
    "खुद ढूंढे गए आधिकारिक नंबर से पुष्टि करें। राष्ट्रीय साइबर हेल्पलाइन: 1930।"
)

# Processes that give a stranger control of the PC (checked by the system sentinel).
REMOTE_ACCESS_PROCESSES: dict[str, str] = {
    "anydesk": "AnyDesk",
    "teamviewer": "TeamViewer",
    "rustdesk": "RustDesk",
    "ultraviewer": "UltraViewer",
    "tv_w32": "TeamViewer",
    "tv_x64": "TeamViewer",
    "quicksupport": "TeamViewer QuickSupport",
    "anyviewer": "AnyViewer",
    "supremo": "Supremo",
    "airdroid": "AirDroid",
    "remoting_host": "Chrome Remote Desktop",
    "screenconnect": "ScreenConnect",
    "zohoassist": "Zoho Assist",
    "splashtop": "Splashtop",
}

# Apps in which scam calls happen on a PC.
CALL_APPS: dict[str, str] = {
    "whatsapp": "WhatsApp",
    "skype": "Skype",
    "ms-teams": "Microsoft Teams",
    "teams": "Microsoft Teams",
    "zoom": "Zoom",
    "telegram": "Telegram",
    "phoneexperiencehost": "Phone Link",
    "yourphone": "Phone Link",
    "signal": "Signal",
    "googlemeet": "Google Meet",
}

# Screen text that shows money is about to move.
PAYMENT_SCREEN_HINTS: tuple[str, ...] = (
    "enter upi pin", "upi pin", "pay to", "amount to be paid", "beneficiary", "add payee", "imps", "rtgs",
    "neft", "transfer funds", "fund transfer", "net banking", "debit card number", "scan and pay",
    "payment successful", "one time password", "otp sent",
)
