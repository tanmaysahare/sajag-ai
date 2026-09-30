"""The Sajag guardian: wires sentinels -> risk engine -> fusion -> alerts."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .risk.engine import TextRiskAnalyzer
from .risk.fusion import RiskFusion, RiskState
from .signals.system import SystemSnapshot, is_payment_screen

log = logging.getLogger("sajag")


@dataclass
class GuardianConfig:
    screen: bool = True
    audio: bool = True
    system: bool = True
    screen_interval_s: float = 2.0
    audio_window_s: float = 6.0
    system_interval_s: float = 3.0
    language: str = "hi"  # Whisper decoder language: "hi" covers Hindi + Hinglish, "en" for English
    whisper_dir: str | None = None
    vad_model: str | None = None
    alerts: bool = True
    store_transcripts: bool = False  # privacy: nothing is persisted unless the user opts in
    extra: dict = field(default_factory=dict)


class Guardian:
    def __init__(self, config: GuardianConfig | None = None, factory=None, notifier=None):
        from .npu import default_factory

        self.config = config or GuardianConfig()
        self.factory = factory or default_factory()
        self.analyzer = TextRiskAnalyzer(factory=self.factory)
        self.fusion = RiskFusion()
        self.notifier = notifier
        self.events: deque[dict] = deque(maxlen=500)
        self.state: RiskState = self.fusion.evaluate()
        self.subscribers: list[Callable[[dict], None]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ocr = None
        self.stats = {"segments": 0, "screens": 0, "screens_skipped": 0, "alerts": 0}

    # ------------------------------------------------------------------ events
    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self.subscribers.append(fn)

    def _emit(self, kind: str, data: dict) -> None:
        ev = {"type": kind, "t": time.time(), **data}
        self.events.append(ev)
        for fn in list(self.subscribers):
            try:
                fn(ev)
            except Exception:  # pragma: no cover
                pass

    def _update(self, now: float | None = None) -> RiskState:
        st = self.fusion.evaluate(now)
        self.state = st
        self._emit("risk", {"state": st.as_dict()})
        if self.fusion.escalated(st) and st.level != "safe":
            self.stats["alerts"] += 1
            record = self.notifier.notify(st) if (self.notifier and self.config.alerts) else None
            self._emit("alert", {"state": st.as_dict(), "record": record})
        return st

    # ------------------------------------------------------------------ inputs
    def ingest_transcript(self, text: str, t: float | None = None, speaker: str = "caller") -> RiskState:
        with self._lock:
            t = t or time.time()
            f = self.analyzer.analyze(text, source="call", timestamp=t)
            self.fusion.add_finding(f)
            self.stats["segments"] += 1
            self._emit("transcript", {"speaker": speaker, "finding": f.as_dict() if self.config.store_transcripts
                                      else {**f.as_dict(), "text": text}})
            return self._update(t)

    def ingest_screen_text(self, text: str, t: float | None = None) -> RiskState:
        with self._lock:
            t = t or time.time()
            f = self.analyzer.analyze(text, source="screen", timestamp=t)
            self.fusion.add_finding(f)
            if is_payment_screen(text):
                self.fusion.set_context(payment_screen=True, t=t)
            self.stats["screens"] += 1
            self._emit("screen", {"finding": f.as_dict()})
            return self._update(t)

    def ingest_screen_image(self, image, t: float | None = None) -> RiskState:
        if self._ocr is None:
            from .vision.ocr import ScreenOCR

            self._ocr = ScreenOCR(factory=self.factory)
        text = self._ocr.read_text(image)
        return self.ingest_screen_text(text, t)

    def ingest_system(self, snap: SystemSnapshot, t: float | None = None) -> RiskState:
        with self._lock:
            t = t or time.time()
            self.fusion.set_context(call_active=snap.call_active, call_app=", ".join(snap.call_apps) or None,
                                    remote_access=snap.remote_access, t=t)
            self._emit("system", {"remote_access": snap.remote_access, "call_apps": snap.call_apps})
            return self._update(t)

    # ------------------------------------------------------------------ live loops
    def _screen_loop(self) -> None:
        from .vision.screen import ScreenWatcher

        watcher = ScreenWatcher(interval_s=self.config.screen_interval_s)
        while not self._stop.is_set():
            try:
                frame = watcher.check(watcher.grab())
                if frame.changed:
                    self.ingest_screen_image(frame.image, frame.t)
                else:
                    self.stats["screens_skipped"] += 1
            except Exception as exc:  # pragma: no cover - hardware specific
                log.warning("screen loop: %s", exc)
            self._stop.wait(self.config.screen_interval_s)

    def _system_loop(self) -> None:
        from .signals.system import snapshot

        while not self._stop.is_set():
            try:
                self.ingest_system(snapshot())
            except Exception as exc:  # pragma: no cover
                log.warning("system loop: %s", exc)
            self._stop.wait(self.config.system_interval_s)

    def _audio_loop(self) -> None:  # pragma: no cover - needs audio hardware + models
        from .audio.asr import AIHubWhisperASR
        from .audio.capture import CallAudioCapture
        from .audio.vad import SileroVAD

        if not self.config.whisper_dir:
            log.warning("audio sentinel disabled: no Whisper model dir (run `python -m sajag fetch-models`)")
            return
        asr = AIHubWhisperASR(self.config.whisper_dir, factory=self.factory, language=self.config.language)
        vad = SileroVAD(self.config.vad_model, factory=self.factory)
        cap = CallAudioCapture()
        cap.start()
        while not self._stop.is_set():
            if not self.fusion.context.get("call_active"):
                cap.read(1.0, timeout=1.0)  # drain; ASR only runs during calls
                continue
            audio = cap.read(self.config.audio_window_s)
            for seg in vad.segments(audio):
                tr = asr.transcribe(seg.audio)
                if tr.text:
                    self.ingest_transcript(tr.text)
        cap.stop()

    def start(self) -> None:
        loops = []
        if self.config.system:
            loops.append(self._system_loop)
        if self.config.screen:
            loops.append(self._screen_loop)
        if self.config.audio:
            loops.append(self._audio_loop)
        for fn in loops:
            threading.Thread(target=fn, daemon=True, name=fn.__name__).start()
        log.info("Sajag guardian running (%s)", ", ".join(f.__name__.strip("_") for f in loops))

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        with self._lock:
            self.fusion.reset()
            self.state = self.fusion.evaluate()
            self.events.clear()
