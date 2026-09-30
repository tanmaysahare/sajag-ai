"""Golden-hour incident report for the 1930 helpline.

Money lost to a scam can often be frozen if the victim reports it quickly
(the "golden hour" that I4C, state police and banks keep stressing), but
victims under stress struggle to recall what the caller claimed, which UPI
ID or account they paid, and when. Sajag already holds that evidence in RAM.

When (and only when) the user presses "Prepare 1930 report", this module turns
the last 15 minutes of evidence into a ready-to-read complaint: a timeline,
the identities the caller claimed, every UPI ID, phone number, account number,
IFSC, amount and link seen on the call or screen, and the next steps. Nothing
is written to disk unless the user chooses to save it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .risk.patterns import FAMILIES

_UPI = re.compile(r"\b([a-zA-Z0-9][\w.\-]{1,60}@[a-zA-Z][a-zA-Z0-9]{1,30})\b(?!\.[a-zA-Z])")
_EMAIL = re.compile(r"\b[\w.\-]+@[\w\-]+\.[a-zA-Z.]{2,}\b")
_PHONE_IN = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?([6-9]\d{4}[\s\-]?\d{5})(?!\d)")
_PHONE_INTL = re.compile(r"(?<!\d)(\+\d{1,3}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{3,4})(?!\d)")
_ACCOUNT = re.compile(r"(?:a/?c|account|acct|khata|खाता)[^\d]{0,20}(\d{9,18})", re.IGNORECASE)
_IFSC = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
_AMOUNT = re.compile(
    r"((?:rs\.?|inr|₹)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|crore|thousand))?"
    r"|\b\d[\d,.]*\s?(?:lakh|crore)(?:\s\d[\d,]*\s?(?:thousand|hazaar))?\b|\b\d[\d,.]*\s?(?:thousand|hazaar)\b"
    r"|\b\d[\d,]*\s?rupees\b)",
    re.IGNORECASE)
_URL = re.compile(r"\b((?:https?://|www\.)\S+|[a-z0-9\-]+\.(?:com|in|net|org|xyz|top|info|live|site|app)\b)",
                  re.IGNORECASE)


@dataclass
class Entities:
    upi_ids: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    accounts: set[str] = field(default_factory=set)
    ifsc: set[str] = field(default_factory=set)
    amounts: set[str] = field(default_factory=set)
    links: set[str] = field(default_factory=set)
    claimed_identities: set[str] = field(default_factory=set)
    remote_apps: set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        return {k: sorted(v) for k, v in self.__dict__.items()}

    def merge(self, other: "Entities") -> None:
        for k, v in other.__dict__.items():
            getattr(self, k).update(v)


def extract_entities(text: str) -> Entities:
    """Pull payment and contact details out of one transcript segment or screen."""
    e = Entities()
    if not text:
        return e
    emails = set(_EMAIL.findall(text))
    for m in _UPI.findall(text):
        if not any(m in em for em in emails):
            e.upi_ids.add(m.lower())
    for m in _PHONE_IN.findall(text):
        e.phones.add("+91 " + re.sub(r"[\s\-]", "", m))
    for m in _PHONE_INTL.findall(text):
        if not m.replace(" ", "").replace("-", "").startswith("+91"):
            e.phones.add(re.sub(r"\s+", " ", m))
    e.accounts.update(_ACCOUNT.findall(text))
    e.ifsc.update(_IFSC.findall(text.upper()))
    e.amounts.update(a.strip() for a in _AMOUNT.findall(text))
    e.links.update(u.lower().rstrip(".,") for u in _URL.findall(text) if "cybercrime.gov.in" not in u.lower())
    return e


NEXT_STEPS = [
    "Hang up and do not call the number back.",
    "Call 1930 (National Cyber Crime Helpline) now. Read out the details below.",
    "Also file online at https://cybercrime.gov.in and note the acknowledgement number.",
    "Call your bank's official number (on your card / passbook) to block cards, UPI and net banking.",
    "Uninstall any remote-access app the caller asked you to install and change your banking passwords.",
    "Tell a family member. Scammers rely on you staying silent.",
]
NEXT_STEPS_HI = [
    "कॉल काटें, उस नंबर पर वापस कॉल न करें।",
    "अभी 1930 (राष्ट्रीय साइबर हेल्पलाइन) पर कॉल करें और नीचे दी गई जानकारी बताएं।",
    "https://cybercrime.gov.in पर भी शिकायत दर्ज करें।",
    "अपने बैंक के आधिकारिक नंबर पर कॉल करके कार्ड, यूपीआई और नेट बैंकिंग बंद करवाएं।",
    "कॉलर द्वारा इंस्टॉल करवाया गया रिमोट ऐप हटाएं और पासवर्ड बदलें।",
    "परिवार के किसी सदस्य को तुरंत बताएं।",
]


def build_report(events: list[dict], state: dict) -> dict:
    """Build the report from the guardian's in-memory events and current risk state."""
    ents = Entities()
    timeline = []
    t0 = None
    for ev in events:
        kind = ev.get("type")
        if kind not in ("transcript", "screen", "system", "alert"):
            continue
        t = (ev.get("finding") or {}).get("timestamp") or ev.get("t", time.time())
        t0 = t0 if t0 is not None else t
        if kind in ("transcript", "screen"):
            f = ev.get("finding", {})
            text = f.get("text", "")
            ents.merge(extract_entities(text))
            for evd in f.get("evidence", []):
                if evd.get("tactic") == "authority":
                    ents.claimed_identities.add(evd.get("phrase", ""))
            if f.get("tactics"):
                summary = " ".join(text.split())
                timeline.append({"t": t, "source": "call" if kind == "transcript" else "screen",
                                  "summary": summary[:150] + ("..." if len(summary) > 150 else ""),
                                  "tactics": f.get("tactics", [])})
        elif kind == "system":
            for app in ev.get("remote_access") or []:
                if app not in ents.remote_apps:
                    ents.remote_apps.add(app)
                    timeline.append({"t": t, "source": "system", "summary": f"{app} started during the call",
                                     "tactics": ["remote_access"]})
        elif kind == "alert":
            s = ev.get("state", {})
            timeline.append({"t": t, "source": "sajag", "summary": f"Sajag raised {s.get('level', '').upper()} "
                                                                    f"({s.get('family_name') or 'scam'})", "tactics": []})
    timeline.sort(key=lambda r: r["t"])
    for row in timeline:
        row["at"] = time.strftime("%H:%M:%S", time.localtime(row["t"]))
    fam = FAMILIES.get(state.get("family") or "")
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_level": state.get("level"),
        "risk_score": state.get("score"),
        "scam_type": fam.name if fam else "Suspected cyber fraud",
        "timeline": timeline,
        "entities": ents.as_dict(),
        "next_steps": NEXT_STEPS,
        "next_steps_hi": NEXT_STEPS_HI,
    }
    report["text"] = render_text(report)
    return report


def render_text(r: dict) -> str:
    e = r["entities"]

    def line(label, vals):
        return f"- {label}: {', '.join(vals) if vals else 'not captured'}"

    out = [
        "SAJAG AI - CYBER FRAUD INCIDENT SUMMARY (prepared on this PC, nothing uploaded)",
        f"Prepared: {r['generated_at']}    Scam type: {r['scam_type']}    Sajag risk: {str(r['risk_level']).upper()}",
        "",
        "DETAILS TO READ OUT ON 1930 / ENTER ON cybercrime.gov.in",
        line("Caller claimed to be", e["claimed_identities"]),
        line("Phone numbers", e["phones"]),
        line("UPI IDs", e["upi_ids"]),
        line("Bank account numbers", e["accounts"]),
        line("IFSC codes", e["ifsc"]),
        line("Amounts mentioned", e["amounts"]),
        line("Links / websites", e["links"]),
        line("Remote-access apps used", e["remote_apps"]),
        "",
        "TIMELINE",
    ]
    for row in r["timeline"]:
        tac = f" [{', '.join(row['tactics'])}]" if row["tactics"] else ""
        out.append(f"  {row['at']}  ({row['source']}) {row['summary']}{tac}")
    out += ["", "WHAT TO DO NOW"] + [f"  {i}. {s}" for i, s in enumerate(r["next_steps"], 1)]
    out += ["", "अभी क्या करें"] + [f"  {i}. {s}" for i, s in enumerate(r["next_steps_hi"], 1)]
    return "\n".join(out)
