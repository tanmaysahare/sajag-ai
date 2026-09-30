"""Model registry and downloader.

All neural models come from Qualcomm AI Hub (pre-compiled for Snapdragon X /
X2 Elite) or from open-source releases, and are fetched once to
``~/.sajag/models``. After that, Sajag never touches the network.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("sajag.models")

MODEL_HOME = Path(os.environ.get("SAJAG_MODELS", Path.home() / ".sajag" / "models"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    source: str  # "aihub" | "bundled" | "pip"
    aihub_id: str | None
    runtime: str | None
    precision: str | None
    role: str
    workload: str
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {m.key: m for m in [
    ModelSpec("vad", "aihub", "silero_vad", "onnx", "w8a16_mixed_int16", "Voice activity gate for call audio",
              "always_on", "0.068 ms per 32 ms chunk on X Elite NPU (AI Hub)"),
    ModelSpec("whisper", "aihub", "whisper_small", "precompiled_qnn_onnx", "float",
              "Multilingual ASR (Hindi, English, Hinglish)", "burst",
              "encoder 117 ms, decoder 10.5 ms/token on X Elite NPU (AI Hub)"),
    ModelSpec("whisper_turbo", "aihub", "whisper_large_v3_turbo", "precompiled_qnn_onnx", "float",
              "Higher-accuracy ASR option for X2 Elite", "burst", "encoder 251 ms on X2 Elite NPU (AI Hub)"),
    ModelSpec("llm", "aihub", "llama_v3_2_3b_instruct", "genie", "w4a16",
              "Optional bilingual explanation", "burst", "~11 tok/s, TTFT ~0.12 s on X Elite (AI Hub, Genie)"),
    ModelSpec("ocr", "pip", None, None, None, "Screen text detection + recognition (PP-OCRv4 via RapidOCR)",
              "interactive", "prepared for NPU by scripts/prepare_npu_models.py"),
    ModelSpec("intent", "bundled", None, None, None, "Scam-intent classifier (hashed char n-grams, 1x Gemm)",
              "background", "sajag/assets/intent_classifier.onnx"),
]}

CHIPSETS = {"x_elite": "qualcomm-snapdragon-x-elite", "x2_elite": "qualcomm-snapdragon-x2-elite"}


def fetch(keys: list[str] | None = None, chipset: str = "x_elite", out: Path = MODEL_HOME) -> dict[str, str]:
    """Download AI Hub assets with the official ``qai_hub_models_cli`` package."""
    from .audio.tokenizer import TIKTOKEN_URL

    out.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    try:
        from qai_hub_models_cli.fetch import fetch as hub_fetch
    except ImportError as exc:
        raise SystemExit("pip install qai_hub_models_cli  (Qualcomm AI Hub model downloader)") from exc

    for key in keys or ["vad", "whisper"]:
        spec = REGISTRY[key]
        if spec.source != "aihub":
            continue
        kwargs = dict(model=spec.aihub_id, runtime=spec.runtime, precision=spec.precision,
                      output_dir=str(out / key), extract=True)
        if spec.runtime and spec.runtime.startswith(("precompiled", "qnn", "genie")):
            kwargs["chipset"] = CHIPSETS[chipset]
        log.info("fetching %s (%s, %s) ...", spec.aihub_id, spec.runtime, spec.precision)
        done[key] = str(hub_fetch(**kwargs))

    tik = out / "whisper" / "multilingual.tiktoken"
    if "whisper" in (keys or ["whisper"]) and not tik.exists():
        tik.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TIKTOKEN_URL, tik)
        done["tokenizer"] = str(tik)
    return done


def locate(key: str) -> Path | None:
    p = MODEL_HOME / key
    return p if p.exists() else None


def find_vad_model() -> Path | None:
    base = locate("vad")
    if not base:
        return None
    c = sorted(base.rglob("*.onnx"))
    return c[0] if c else None
