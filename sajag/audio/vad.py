"""Voice activity detection with AI Hub Silero-VAD (NPU, 0.07 ms / 32 ms chunk).

VAD is the gate for the whole audio path: Whisper only runs on speech
segments, so silence and music cost nothing on the NPU. The IO follows AI
Hub's export: ``audio_chunk [1,512]``, ``state [1,2,128]``, ``context [1,64]``.

When no model is available (tests, demo machines) an energy-based fallback
with the same hysteresis parameters is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHUNK = 512
CONTEXT = 64
SR = 16000


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float
    audio: np.ndarray


class SileroVAD:
    def __init__(self, model_path: str | Path | None = None, factory=None, threshold: float = 0.5):
        from ..npu import default_factory

        self.threshold = threshold
        self.neg_threshold = threshold - 0.15
        self.session = None
        if model_path and Path(model_path).exists():
            self.session = (factory or default_factory()).create("silero-vad", model_path, workload="always_on")
            self.names = [i.name for i in self.session.get_inputs()]
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((1, 2, 128), np.float32)
        self.context = np.zeros((1, CONTEXT), np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        chunk = np.asarray(chunk, np.float32).reshape(1, CHUNK)
        if self.session is None:  # energy fallback
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
            return float(np.clip((20 * np.log10(rms) + 45) / 20, 0, 1))
        p, self.state, self.context = self.session.run(
            {self.names[0]: chunk, self.names[1]: self.state, self.names[2]: self.context})
        return float(np.asarray(p).reshape(-1)[0])

    def segments(self, audio: np.ndarray, min_speech_ms: int = 250, min_silence_ms: int = 400,
                 pad_ms: int = 60, max_segment_s: float = 28.0):
        """Yield speech segments from a mono 16 kHz array."""
        self.reset()
        speaking, start, silence = False, 0, 0
        pad = int(SR * pad_ms / 1000)
        n = len(audio) // CHUNK
        for i in range(n):
            pos = i * CHUNK
            p = self.prob(audio[pos:pos + CHUNK])
            if not speaking and p >= self.threshold:
                speaking, start, silence = True, pos, 0
            elif speaking:
                silence = silence + CHUNK if p < self.neg_threshold else 0
                too_long = (pos - start) / SR >= max_segment_s
                if silence >= SR * min_silence_ms / 1000 or too_long:
                    end = pos
                    if (end - start) >= SR * min_speech_ms / 1000:
                        s, e = max(0, start - pad), min(len(audio), end + pad)
                        yield SpeechSegment(s / SR, e / SR, audio[s:e])
                    speaking = False
        if speaking and (n * CHUNK - start) >= SR * min_speech_ms / 1000:
            yield SpeechSegment(start / SR, n * CHUNK / SR, audio[start:n * CHUNK])
