"""Scripted, reproducible scenarios for judges without a Snapdragon PC.

Each scenario is a timeline of what Sajag's sentinels would observe: call
transcript segments (what Whisper produces), screen frames (rendered here as
images and read back through the real OCR engine) and system snapshots
(process lists). Screens are clearly watermarked as simulated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .signals.system import snapshot

SCENARIOS: dict[str, dict] = {
    "digital_arrest": {
        "title": "Digital arrest over a WhatsApp video call (Hinglish)",
        "expected": "danger",
        "timeline": [
            {"at": 0, "system": ["WhatsApp.exe", "explorer.exe"]},
            {"at": 4, "call": "Hello, main Mumbai police cyber crime branch se Inspector Sharma bol raha hoon."},
            {"at": 12, "call": "Aapke naam se FedEx parcel mein drugs aur paanch fake passports mile hain."},
            {"at": 20, "call": "Your Aadhaar card has been used in a money laundering case, an arrest warrant is issued."},
            {"at": 26, "screen": "warrant"},
            {"at": 34, "call": "Ye national secret hai, kisi ko mat batana aur call mat kaatna, camera on rakho."},
            {"at": 45, "call": "Verification ke liye apne account ka paisa RBI safe account mein transfer karo, jaanch ke baad wapas milega."},
            {"at": 52, "screen": "bank_transfer"},
        ],
    },
    "tech_support": {
        "title": "Fake Microsoft pop-up + AnyDesk remote access",
        "expected": "danger",
        "timeline": [
            {"at": 0, "system": ["msedge.exe", "explorer.exe"]},
            {"at": 3, "screen": "scareware"},
            {"at": 15, "system": ["msedge.exe", "PhoneExperienceHost.exe"]},
            {"at": 18, "call": "Hello, I am from Windows technical support, your computer is infected with a trojan."},
            {"at": 27, "call": "Please download AnyDesk and tell me the nine digit code on your screen."},
            {"at": 35, "system": ["msedge.exe", "PhoneExperienceHost.exe", "AnyDesk.exe"]},
            {"at": 44, "call": "Now open your net banking so I can process the refund of the service charge."},
            {"at": 50, "screen": "bank_transfer"},
        ],
    },
    "kyc_otp": {
        "title": "KYC expiry call asking for OTP (Hindi)",
        "expected": "warning",
        "timeline": [
            {"at": 0, "system": ["PhoneExperienceHost.exe"]},
            {"at": 3, "call": "नमस्ते, मैं एसबीआई केवाईसी विभाग से बोल रहा हूँ।"},
            {"at": 10, "call": "आपका केवाईसी खत्म हो गया है, आज ही अपडेट नहीं किया तो खाता बंद हो जाएगा।"},
            {"at": 18, "call": "अभी आपके फोन पर एक ओटीपी आया होगा, वो ओटीपी बताइए।"},
        ],
    },
    "family_call": {
        "title": "Normal family video call (should stay safe)",
        "expected": "safe",
        "timeline": [
            {"at": 0, "system": ["WhatsApp.exe"]},
            {"at": 3, "call": "Beta, khana kha liya? Papa pooch rahe the Diwali pe kab aa rahe ho."},
            {"at": 10, "call": "Train ki ticket book kar di hai, 28 tarikh ko pahunch jaunga."},
            {"at": 18, "call": "Bank wale ka message aaya tha, OTP kisi ko mat batana, bank kabhi nahi maangta."},
            {"at": 25, "call": "Achha theek hai, dadi ko pranam bolna."},
        ],
    },
    "office_it": {
        "title": "Office IT fixing a laptop over TeamViewer (legit, low risk)",
        "expected": "caution",
        "timeline": [
            {"at": 0, "system": ["Teams.exe"]},
            {"at": 4, "call": "Hi, this is Ravi from the IT helpdesk about your ticket for the VPN issue."},
            {"at": 12, "call": "I will connect through TeamViewer from the company portal to update the VPN client."},
            {"at": 18, "system": ["Teams.exe", "TeamViewer.exe"]},
            {"at": 26, "call": "Done, please restart once and let me know if the VPN connects."},
        ],
    },
}

# ------------------------------------------------------------------ screens
SCREENS = {
    "scareware": {
        "bg": (0, 90, 170),
        "lines": [
            ("Windows Defender Security Center", 34, (255, 255, 255)),
            ("Access to this PC has been blocked for security reasons", 26, (255, 255, 255)),
            ("Your computer is infected with Trojan Spyware", 26, (255, 230, 120)),
            ("Your personal data is at risk. Do not restart your computer.", 22, (255, 255, 255)),
            ("Error code: 0x80070424", 22, (255, 255, 255)),
            ("Call Windows Support Toll Free: +1-800-000-0000", 28, (255, 230, 120)),
        ],
    },
    "warrant": {
        "bg": (250, 248, 240),
        "lines": [
            ("CENTRAL BUREAU OF INVESTIGATION", 32, (20, 20, 20)),
            ("ARREST WARRANT", 34, (170, 0, 0)),
            ("Case: Money Laundering and Narcotics", 24, (20, 20, 20)),
            ("The accused is under digital arrest until investigation is complete.", 20, (20, 20, 20)),
            ("Keep this confidential. Do not tell anyone.", 22, (20, 20, 20)),
            ("Transfer funds to the RBI safe account for verification.", 22, (20, 20, 20)),
        ],
    },
    "bank_transfer": {
        "bg": (255, 255, 255),
        "lines": [
            ("Net Banking - Fund Transfer", 32, (0, 60, 130)),
            ("Add Payee / Beneficiary: RBI Safe Account", 24, (20, 20, 20)),
            ("Transfer mode: IMPS / RTGS", 24, (20, 20, 20)),
            ("Amount to be paid: Rs 9,80,000", 26, (20, 20, 20)),
            ("Enter the One Time Password (OTP) sent to your mobile", 22, (20, 20, 20)),
        ],
    },
}


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_screen(kind: str, size=(1280, 720)) -> Image.Image:
    spec = SCREENS[kind]
    img = Image.new("RGB", size, spec["bg"])
    d = ImageDraw.Draw(img)
    y = 70
    for text, fs, col in spec["lines"]:
        f = _font(fs)
        w = d.textlength(text, font=f)
        d.text(((size[0] - w) / 2, y), text, font=f, fill=col)
        y += int(fs * 2.1)
    band = _font(18)
    d.rectangle([0, size[1] - 40, size[0], size[1]], fill=(40, 40, 40))
    d.text((20, size[1] - 32), "SIMULATED SCAM SCREEN FOR SAJAG AI DEMO - NOT A REAL DOCUMENT", font=band,
           fill=(255, 255, 255))
    return img


def run_scenario(name: str, guardian, realtime: bool = False, use_ocr: bool = True, on_step=None) -> list[dict]:
    """Replay a scenario through a Guardian. Returns the per-step risk trace."""
    sc = SCENARIOS[name]
    guardian.reset()
    t0 = time.time()
    trace = []
    last_at = 0.0
    for step in sc["timeline"]:
        if realtime:
            time.sleep(max(0.0, step["at"] - last_at) * 0.25)
        last_at = step["at"]
        t = t0 + step["at"]
        if "system" in step:
            st = guardian.ingest_system(snapshot(step["system"]), t)
            what = "system: " + ", ".join(step["system"])
        elif "call" in step:
            st = guardian.ingest_transcript(step["call"], t)
            what = "call: " + step["call"]
        else:
            img = render_screen(step["screen"])
            if use_ocr:
                st = guardian.ingest_screen_image(img, t)
            else:
                st = guardian.ingest_screen_text("\n".join(l[0] for l in SCREENS[step["screen"]]["lines"]), t)
            what = "screen: " + step["screen"]
        row = {"at": step["at"], "what": what, "score": st.score, "level": st.level, "family": st.family}
        trace.append(row)
        if on_step:
            on_step(row, st)
    return trace


def export_scenarios(out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, sc in SCENARIOS.items():
        (out / f"{name}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8")
    for kind in SCREENS:
        render_screen(kind).save(out / f"screen_{kind}.png")
