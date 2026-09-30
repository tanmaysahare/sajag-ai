"""Speech recognition with Qualcomm AI Hub Whisper on the Hexagon NPU.

This is a NumPy / ONNX Runtime port of AI Hub's ``HfWhisperApp`` decode loop
(qai_hub_models.models.templates.hf_whisper.app) so it runs without torch:

* encoder:  input_features [1, n_mels, 3000] -> k/v cross-attention caches
* decoder:  one token per step with a fixed 200-slot self-attention KV cache,
            attention_mask [1, 1, 1, 200] and int32 position_ids

Everything is static-shape, so AI Hub's precompiled QNN ONNX assets
(``qai-hub-models fetch whisper_small --runtime precompiled_qnn_onnx``)
run fully on the NPU (encoder 1582/1582 and decoder 2277/2277 layers on NPU
according to AI Hub's published X Elite profile).

We add one thing the reference app does not do: a forced decoder prompt
(``<|hi|>`` / ``<|en|>`` + ``<|transcribe|>``), which stops Whisper from
translating Hinglish calls into English and keeps scam cues intact.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import SAMPLE_RATE, log_mel_spectrogram
from .tokenizer import WhisperTokenizer

log = logging.getLogger("sajag.asr")

MEAN_DECODE_LEN = 200
MASK_NEG = -100.0


@dataclass
class Transcript:
    text: str
    language: str | None
    audio_s: float
    encoder_ms: float = 0.0
    decoder_ms: float = 0.0
    tokens: int = 0


class AIHubWhisperASR:
    def __init__(self, model_dir: str | os.PathLike, factory=None, language: str | None = "hi",
                 n_mels: int | None = None):
        from ..npu import default_factory

        model_dir = Path(model_dir)
        factory = factory or default_factory()
        enc = self._find(model_dir, "encoder")
        dec = self._find(model_dir, "decoder")
        # Encoder runs once per utterance -> burst; decoder is the hot loop -> burst.
        self.encoder = factory.create("whisper-encoder", enc, workload="burst")
        self.decoder = factory.create("whisper-decoder", dec, workload="burst")
        self.tokenizer = WhisperTokenizer(self._find(model_dir, "multilingual.tiktoken", ext=""))
        self.language = language

        enc_in = self.encoder.get_inputs()[0]
        self.enc_input = enc_in.name
        self.n_mels = n_mels or int(enc_in.shape[1])
        self.dec_inputs = [i.name for i in self.decoder.get_inputs()]
        self.dec_shapes = {i.name: list(i.shape) for i in self.decoder.get_inputs()}
        self.cross_names = [n for n in self.dec_inputs if "cache_cross" in n]
        self.self_k = sorted((n for n in self.dec_inputs if n.startswith("k_cache_self")), key=_layer)
        self.self_v = sorted((n for n in self.dec_inputs if n.startswith("v_cache_self")), key=_layer)
        self.enc_outputs = [o.name for o in self.encoder.get_outputs()]
        self.dec_outputs = [o.name for o in self.decoder.get_outputs()]

    @staticmethod
    def _find(model_dir: Path, key: str, ext: str = ".onnx") -> Path:
        cands = sorted(p for p in model_dir.rglob(f"*{key}*{ext}") if p.is_file())
        if not cands:
            raise FileNotFoundError(f"no '{key}' model under {model_dir}; run `python -m sajag fetch-models`")
        return cands[0]

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Transcript:
        import time

        from .features import resample

        audio = resample(audio, sr)
        feats = log_mel_spectrogram(audio, n_mels=self.n_mels)[None]

        t0 = time.perf_counter()
        cross = self.encoder.run({self.enc_input: feats})
        t1 = time.perf_counter()
        cross_feed = dict(zip(self.enc_outputs, cross))

        k_self = [np.zeros(self.dec_shapes[n], np.float32) for n in self.self_k]
        v_self = [np.zeros(self.dec_shapes[n], np.float32) for n in self.self_v]
        mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, np.float32)
        prompt = self.tokenizer.prompt(self.language)
        out_ids = list(prompt)
        pos = np.array([0], dtype=np.int32)

        for n in range(MEAN_DECODE_LEN - 1):
            mask[0, 0, 0, MEAN_DECODE_LEN - n - 1] = 0.0
            feed = {"input_ids": np.array([[out_ids[n]]], np.int32), "attention_mask": mask, "position_ids": pos}
            for name, arr in zip(self.self_k, k_self):
                feed[name] = arr
            for name, arr in zip(self.self_v, v_self):
                feed[name] = arr
            for name in self.cross_names:
                feed[name] = cross_feed[name]
            outs = dict(zip(self.dec_outputs, self.decoder.run(feed)))
            logits = outs["logits"]
            k_self = [outs[n.replace("_in", "_out")] for n in self.self_k]
            v_self = [outs[n.replace("_in", "_out")] for n in self.self_v]
            if n >= len(prompt) - 1:
                nxt = int(np.argmax(logits.reshape(-1)))
                out_ids.append(nxt)
                if nxt == self.tokenizer.eot:
                    break
            pos = pos + 1
        t2 = time.perf_counter()

        gen = out_ids[len(prompt):]
        return Transcript(
            text=self.tokenizer.decode(gen),
            language=self.language,
            audio_s=len(audio) / SAMPLE_RATE,
            encoder_ms=(t1 - t0) * 1000,
            decoder_ms=(t2 - t1) * 1000,
            tokens=len(gen),
        )


def _layer(name: str) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    return int(digits[0]) if digits else 0
