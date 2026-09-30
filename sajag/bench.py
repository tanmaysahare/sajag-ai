"""Benchmark every Sajag model on the providers available on this machine.

Run on a Snapdragon PC with onnxruntime-qnn:   python -m sajag bench --out bench.json
The report records provider (QNN vs CPU), power mode, load time (including
HTP graph compilation or context-binary load) and p50 / p90 latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .npu import SessionFactory, device_summary


def _time(fn, warmup: int = 3, iters: int = 30) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p90_ms": round(float(np.percentile(a, 90)), 3),
            "mean_ms": round(float(a.mean()), 3), "iters": iters}


def run(out: str | None = None, iters: int = 30) -> dict:
    from .demo import render_screen
    from .models import find_vad_model
    from .risk.classifier import MODEL_PATH, featurize
    from .risk.engine import TextRiskAnalyzer

    results: dict = {"device": device_summary(), "models": []}
    for prefer_npu in ([True, False] if device_summary()["qnn_npu"] else [False]):
        fac = SessionFactory(prefer_npu=prefer_npu)
        tag = "npu" if prefer_npu else "cpu"

        s = fac.create("intent-classifier", MODEL_PATH, workload="background")
        x = featurize("verification ke liye paise transfer karo")[None]
        r = _time(lambda: s.run({s.get_inputs()[0].name: x}), iters=iters * 3)
        results["models"].append({"model": "intent-classifier", "target": tag, "provider": s.info.provider,
                                  "load_ms": round(s.info.load_ms, 1), **r})

        vad_path = find_vad_model()
        if vad_path:
            v = fac.create("silero-vad", vad_path, workload="always_on")
            names = [i.name for i in v.get_inputs()]
            feed = {names[0]: np.zeros((1, 512), np.float32), names[1]: np.zeros((1, 2, 128), np.float32),
                    names[2]: np.zeros((1, 64), np.float32)}
            r = _time(lambda: v.run(feed), iters=iters * 3)
            results["models"].append({"model": "silero-vad", "target": tag, "provider": v.info.provider,
                                      "load_ms": round(v.info.load_ms, 1), **r})

        from .vision.ocr import ScreenOCR

        ocr = ScreenOCR(factory=fac) if prefer_npu else ScreenOCR(factory=fac, npu_model_dir="/nonexistent")
        img = render_screen("scareware")
        r = _time(lambda: ocr.read(img), warmup=1, iters=max(5, iters // 3))
        results["models"].append({"model": "screen-ocr (det+rec)", "target": tag, "provider": ocr.provider, **r})

        an = TextRiskAnalyzer(factory=fac)
        r = _time(lambda: an.analyze("Aapke naam se parcel mein drugs mile hain, kisi ko mat batana"), iters=iters * 3)
        results["models"].append({"model": "text risk analyzer (patterns + classifier)", "target": tag,
                                  "provider": "mixed", **r})

    if out:
        Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
