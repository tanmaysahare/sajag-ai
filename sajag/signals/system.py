"""System sentinel: remote-access tools, active call apps, payment screens.

These signals need no neural network at all, but they are the strongest
context for the fusion engine: a remote-access tool starting *while* a call
app is active is the single most common step before money leaves the account.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

from ..risk.normalize import normalize
from ..risk.patterns import CALL_APPS, PAYMENT_SCREEN_HINTS, REMOTE_ACCESS_PROCESSES

log = logging.getLogger("sajag.system")


@dataclass
class SystemSnapshot:
    remote_access: list[str] = field(default_factory=list)
    call_apps: list[str] = field(default_factory=list)
    foreground_title: str = ""

    @property
    def call_active(self) -> bool:
        return bool(self.call_apps)


def _match(name: str, table: dict[str, str]) -> str | None:
    n = name.lower().replace(" ", "")
    for key, label in table.items():
        if key in n:
            return label
    return None


def foreground_window_title() -> str:
    if platform.system() != "Windows":
        return ""
    try:  # pragma: no cover - Windows only
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def snapshot(process_names: list[str] | None = None) -> SystemSnapshot:
    """Inspect running processes (or a supplied list, for tests and demos)."""
    if process_names is None:
        import psutil

        process_names = []
        for p in psutil.process_iter(["name"]):
            try:
                if p.info.get("name"):
                    process_names.append(p.info["name"])
            except Exception:  # pragma: no cover
                continue
    remote, calls = set(), set()
    for name in process_names:
        ra = _match(name, REMOTE_ACCESS_PROCESSES)
        if ra:
            remote.add(ra)
        ca = _match(name, CALL_APPS)
        if ca:
            calls.add(ca)
    return SystemSnapshot(sorted(remote), sorted(calls), foreground_window_title())


def is_payment_screen(screen_text: str) -> bool:
    t = normalize(screen_text)
    return sum(1 for h in PAYMENT_SCREEN_HINTS if h in t) >= 2
