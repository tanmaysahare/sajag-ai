"""Alert channels: console, Windows toast, spoken warning, full-screen interrupt.

Design choice: for WARNING and DANGER the alert *interrupts*. Scam victims are
typically kept in a state of fear and focus by the caller; a small toast in
the corner is easy to miss. A full-screen, high-contrast bilingual card with a
spoken warning breaks that spell. Every channel degrades gracefully if its
optional dependency is missing.
"""

from __future__ import annotations

import logging
import threading

from ..risk.explain import template_explanation
from ..risk.fusion import RiskState

log = logging.getLogger("sajag.alerts")


class Notifier:
    def __init__(self, speak: bool = True, overlay: bool = True, toast: bool = True, lang: str = "en"):
        self.speak_enabled = speak
        self.overlay_enabled = overlay
        self.toast_enabled = toast
        self.lang = lang
        self.history: list[dict] = []

    def notify(self, state: RiskState) -> dict:
        exp = template_explanation(state, self.lang)
        exp_hi = template_explanation(state, "hi")
        record = {"level": state.level, "score": state.score, "family": state.family, "en": exp, "hi": exp_hi}
        self.history.append(record)
        log.warning("[%s %.2f] %s | %s", state.level.upper(), state.score, exp["headline"], "; ".join(exp["why"][:3]))
        if state.level in ("warning", "danger"):
            if self.toast_enabled:
                self._toast(exp)
            if self.speak_enabled:
                threading.Thread(target=self._speak, args=(exp["headline"] + " " + exp["advice"],), daemon=True).start()
            if self.overlay_enabled and state.level == "danger":
                threading.Thread(target=self._overlay, args=(exp, exp_hi), daemon=True).start()
        return record

    @staticmethod
    def _toast(exp: dict) -> None:
        try:  # pragma: no cover - Windows only
            from win11toast import toast

            toast("Sajag AI", f"{exp['headline']}\n{exp['advice']}", duration="long")
        except Exception:
            pass

    @staticmethod
    def _speak(text: str) -> None:
        try:  # pragma: no cover - needs an audio device
            import pyttsx3

            eng = pyttsx3.init()
            eng.setProperty("rate", 150)
            eng.say(text)
            eng.runAndWait()
        except Exception:
            pass

    @staticmethod
    def _overlay(exp: dict, exp_hi: dict) -> None:
        try:  # pragma: no cover - needs a display
            import tkinter as tk

            root = tk.Tk()
            root.attributes("-topmost", True)
            root.attributes("-fullscreen", True)
            root.configure(bg="#B00020")
            for txt, size in ((exp["headline"], 40), (exp_hi["headline"], 34), (exp["advice"], 20),
                              (exp_hi["advice"], 18), ("Helpline 1930  |  cybercrime.gov.in", 24)):
                tk.Label(root, text=txt, fg="white", bg="#B00020", font=("Segoe UI", size, "bold"),
                         wraplength=1200, justify="center").pack(pady=14)
            tk.Button(root, text="I understand, close", font=("Segoe UI", 18), command=root.destroy).pack(pady=30)
            root.mainloop()
        except Exception:
            pass
