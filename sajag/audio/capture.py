"""Call audio capture: system loopback (the caller) + microphone (the user).

On Windows, WASAPI loopback captures whatever the call app is playing, so
Sajag works with WhatsApp Desktop, Skype, Teams, Zoom, Meet and Phone Link
without integrating with any of them. Audio never touches the disk.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger("sajag.capture")
SR = 16000


class CallAudioCapture:
    def __init__(self, block_s: float = 0.5, include_mic: bool = True):
        self.block = int(SR * block_s)
        self.include_mic = include_mic
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=240)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _reader(self, recorder_ctx, label: str) -> None:
        try:
            with recorder_ctx as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.block)
                    mono = data.mean(axis=1).astype(np.float32) if data.ndim == 2 else data.astype(np.float32)
                    try:
                        self.q.put_nowait(mono)
                    except queue.Full:
                        pass
        except Exception as exc:  # pragma: no cover - hardware specific
            log.warning("%s capture stopped: %s", label, exc)

    def start(self) -> None:
        import soundcard as sc  # WASAPI on Windows, PulseAudio on Linux

        speaker = sc.default_speaker()
        loop = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self._threads.append(threading.Thread(target=self._reader, args=(loop.recorder(samplerate=SR, channels=1),
                                                                          "loopback"), daemon=True))
        if self.include_mic:
            mic = sc.default_microphone()
            self._threads.append(threading.Thread(target=self._reader, args=(mic.recorder(samplerate=SR, channels=1),
                                                                              "mic"), daemon=True))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()

    def read(self, seconds: float, timeout: float = 5.0) -> np.ndarray:
        need = int(SR * seconds)
        buf: list[np.ndarray] = []
        got = 0
        while got < need:
            try:
                blk = self.q.get(timeout=timeout)
            except queue.Empty:
                break
            buf.append(blk)
            got += len(blk)
        return np.concatenate(buf) if buf else np.zeros(0, np.float32)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr
