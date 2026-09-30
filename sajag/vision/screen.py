"""Screen sentinel: cheap change detection first, NPU OCR only when needed.

The expensive step (OCR) is gated by a 64-bit difference hash of a thumbnail.
If the screen has not meaningfully changed, nothing runs on the NPU. This is
the main reason the always-on screen guard costs almost no battery.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger("sajag.screen")


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class ScreenFrame:
    image: Image.Image
    t: float
    changed: bool
    hash: int


class ScreenWatcher:
    def __init__(self, interval_s: float = 2.0, change_bits: int = 6, monitor: int = 1, max_width: int = 1600):
        self.interval_s = interval_s
        self.change_bits = change_bits
        self.monitor = monitor
        self.max_width = max_width
        self._last_hash: int | None = None
        self.frames_seen = 0
        self.frames_ocr = 0

    def grab(self) -> Image.Image:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[min(self.monitor, len(sct.monitors) - 1)]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if img.width > self.max_width:
            h = int(img.height * self.max_width / img.width)
            img = img.resize((self.max_width, h), Image.LANCZOS)
        return img

    def check(self, img: Image.Image) -> ScreenFrame:
        h = dhash(img)
        changed = self._last_hash is None or hamming(h, self._last_hash) >= self.change_bits
        if changed:
            self._last_hash = h
            self.frames_ocr += 1
        self.frames_seen += 1
        return ScreenFrame(img, time.time(), changed, h)

    @property
    def skip_ratio(self) -> float:
        return 1.0 - (self.frames_ocr / self.frames_seen) if self.frames_seen else 0.0
