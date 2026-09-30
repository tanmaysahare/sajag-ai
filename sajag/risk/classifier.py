"""Scam-intent classifier: hashed character n-grams + linear head in ONNX.

Why this design:

* Character n-grams survive ASR / OCR spelling noise and work across English,
  Devanagari Hindi and romanised Hinglish without a tokenizer.
* Feature hashing keeps the model a fixed-shape ``[1, 16384]`` input, which is
  exactly what the Hexagon NPU wants (static shapes, one Gemm + Softmax).
* The whole model is ~1 MB, so it can run on every new transcript segment and
  every OCR result at negligible power.

The feature function is duplicated in ``scripts/train_classifier.py`` via an
import of :func:`featurize`, so training and inference can never drift.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .normalize import normalize

N_FEATURES = 1 << 14
NGRAM_RANGE = (2, 4)
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
MODEL_PATH = ASSET_DIR / "intent_classifier.onnx"
LABELS_PATH = ASSET_DIR / "intent_labels.json"


def _hash(token: str) -> int:
    # crc32 is stable across processes and platforms (unlike Python's hash()).
    return zlib.crc32(token.encode("utf-8")) & (N_FEATURES - 1)


def featurize(text: str) -> np.ndarray:
    """Return an L2-normalised hashed bag of char n-grams (+ word unigrams)."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    t = normalize(text)
    if not t:
        return vec
    padded = f" {t} "
    lo, hi = NGRAM_RANGE
    for n in range(lo, hi + 1):
        for i in range(len(padded) - n + 1):
            vec[_hash(f"c{n}:{padded[i:i + n]}")] += 1.0
    for w in t.split():
        vec[_hash(f"w:{w}")] += 2.0
    vec = np.log1p(vec)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class IntentClassifier:
    """Runs the exported ONNX head through the shared NPU session factory."""

    def __init__(self, factory=None, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        from ..npu import default_factory

        self.labels: list[str] = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.session = (factory or default_factory()).create("intent-classifier", model_path, workload="background")
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def available(cls) -> bool:
        return MODEL_PATH.exists() and LABELS_PATH.exists()

    def predict_proba(self, text: str) -> dict[str, float]:
        x = featurize(text)[None, :]
        probs = self.session.run({self.input_name: x})[0][0]
        return {lab: float(p) for lab, p in zip(self.labels, probs)}

    def scam_probability(self, text: str) -> tuple[float, str | None]:
        """Probability that ``text`` belongs to *any* scam family, and the top family."""
        p = self.predict_proba(text)
        benign = p.get("benign", 0.0)
        top = max((k for k in p if k != "benign"), key=lambda k: p[k], default=None)
        return 1.0 - benign, top
