"""Screen OCR on the Hexagon NPU.

Engine: PP-OCRv4 text detection + recognition (open-source, via RapidOCR's
ONNX models). Sajag re-creates RapidOCR's ONNX sessions through the shared
:class:`~sajag.npu.SessionFactory` so they execute on the QNN Execution
Provider when available. ``scripts/prepare_npu_models.py`` fixes the dynamic
input shapes and produces QDQ-quantised (w8a16) variants, which is what the
HTP backend needs to keep the whole graph on the NPU.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("sajag.ocr")


class StaticShapeAdapter:
    """Lets a fixed-shape NPU graph serve RapidOCR's variable-shape calls.

    The HTP backend needs static shapes, but OCR inputs vary with the screen.
    Inputs are zero-padded (bottom/right) to the compiled shape, run on the NPU
    one image at a time, and outputs are cropped back. Inputs larger than the
    compiled shape go to the original CPU session, so accuracy never degrades.
    """

    def __init__(self, npu_session, cpu_session):
        self.npu = npu_session
        self.cpu = cpu_session
        inp = npu_session.get_inputs()[0]
        self.in_name = inp.name
        self.shape = [int(d) if isinstance(d, int) else -1 for d in inp.shape]  # [1, 3, H, W]
        self.npu_calls = 0
        self.cpu_calls = 0

    def get_inputs(self):
        return self.cpu.get_inputs()

    def get_outputs(self):
        return self.cpu.get_outputs()

    def run(self, output_names, feeds):
        x = next(iter(feeds.values()))
        _, c, H, W = self.shape
        n, xc, h, w = x.shape
        if H <= 0 or W <= 0 or h > H or w > W or xc != c:
            self.cpu_calls += 1
            return self.cpu.run(output_names, feeds)
        outs = []
        for i in range(n):
            padded = np.zeros((1, c, H, W), dtype=x.dtype)
            padded[0, :, :h, :w] = x[i]
            y = self.npu.run(output_names, {self.in_name: padded})[0]
            if y.ndim == 4:  # detector probability map [1, 1, H, W]
                y = y[:, :, :h, :w]
            elif y.ndim == 3:  # recogniser logits [1, T, C], T = W / 8
                t = max(1, int(round(y.shape[1] * w / W)))
                y = y[:, :t, :]
            outs.append(y)
        self.npu_calls += 1
        return [np.concatenate(outs, axis=0)]


@dataclass
class OCRLine:
    text: str
    score: float
    box: list


class ScreenOCR:
    def __init__(self, factory=None, npu_model_dir: str | os.PathLike | None = None, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        from ..npu import default_factory, qnn_available

        self.min_score = min_score
        self.engine = RapidOCR()
        self.factory = factory or default_factory()
        self.provider = "CPUExecutionProvider"
        npu_dir = Path(npu_model_dir or os.environ.get("SAJAG_OCR_NPU_DIR", Path.home() / ".sajag" / "models" / "ocr"))
        if qnn_available():
            self._move_to_npu(npu_dir)

    def _move_to_npu(self, npu_dir: Path) -> None:
        """Swap RapidOCR's CPU sessions for NPU sessions built from prepared models."""
        swaps = {"text_det": "det_npu.onnx", "text_rec": "rec_npu.onnx"}
        for attr, fname in swaps.items():
            path = npu_dir / fname
            target = getattr(self.engine, attr, None)
            if target is None or not path.exists():
                log.info("NPU OCR model %s not prepared; %s stays on CPU", path, attr)
                continue
            try:
                ts = self.factory.create(f"ocr-{attr}", path, workload="interactive")
                inner = getattr(target, "infer", None) or getattr(target, "session", None)
                if inner is not None and hasattr(inner, "session"):
                    adapter = StaticShapeAdapter(ts.session, inner.session)
                    inner.session = adapter
                    self.provider = ts.info.provider
                    if attr == "text_rec":  # one line per call so only very long lines need the CPU
                        target.rec_batch_num = 1
                    if attr == "text_det":  # make the detector resize into the compiled canvas
                        self._pin_detector_size(target, min(adapter.shape[2], adapter.shape[3]),
                                                max(adapter.shape[2], adapter.shape[3]))
            except Exception as exc:  # pragma: no cover - device specific
                log.warning("could not move %s to NPU: %s", attr, exc)

    @staticmethod
    def _pin_detector_size(det, short_side: int, long_side: int) -> None:
        """Resize every screen so it fits the NPU canvas (long side <= long_side)."""
        from rapidocr_onnxruntime.ch_ppocr_det.text_detect import DetPreProcess

        def get_preprocess(max_wh, _det=det):
            return DetPreProcess(long_side, "max", _det.mean, _det.std)

        det.limit_type, det.limit_side_len = "max", long_side
        det.get_preprocess = get_preprocess

    def read(self, image) -> list[OCRLine]:
        """OCR a PIL image or numpy array (H, W, 3, RGB)."""
        arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        arr = arr[:, :, ::-1].copy()  # RapidOCR expects BGR like OpenCV
        result, _ = self.engine(arr)
        lines: list[OCRLine] = []
        for box, text, score in result or []:
            if float(score) >= self.min_score and text.strip():
                lines.append(OCRLine(text=text.strip(), score=float(score), box=box))
        return lines

    def read_text(self, image) -> str:
        return "\n".join(line.text for line in self.read(image))
