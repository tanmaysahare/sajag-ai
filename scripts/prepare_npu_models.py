"""Prepare the PP-OCRv4 screen-OCR models for the Hexagon NPU.

The HTP backend of the QNN Execution Provider needs (1) static input shapes
and (2) a quantised graph for best speed and power. This script:

  1. fixes the dynamic dims of the detector to [1, 3, 736, 1280] and of the
     recogniser to [1, 3, 48, 1280]  (onnxruntime.tools.onnx_model_utils)
  2. runs QNN-specific pre-processing (qnn_preprocess_model)
  3. (optional, --quantize) calibrates on screen images and writes QDQ models
     with uint16 activations / uint8 weights (w8a16) via get_qnn_qdq_config

Outputs go to ~/.sajag/models/ocr/{det,rec}_npu.onnx, where
sajag.vision.ocr.ScreenOCR picks them up automatically on a Snapdragon PC.

Run once (any machine, x86 is fine):  python scripts/prepare_npu_models.py
Default output is the static-shape fp32 graph, which the HTP runs in fp16.
Add --quantize for the experimental QDQ w8a16 variant (calibrate on real
screenshots of your own desktop for best accuracy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DET_SHAPE = [1, 3, 736, 1280]
REC_SHAPE = [1, 3, 48, 1280]


def rapidocr_models() -> tuple[Path, Path]:
    import rapidocr_onnxruntime

    base = Path(rapidocr_onnxruntime.__file__).parent / "models"
    det = next(base.glob("*det*.onnx"))
    rec = next(base.glob("*rec*.onnx"))
    return det, rec


def fix_shape(src: Path, dst: Path, shape: list[int]) -> Path:
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    model = onnx.load(str(src))
    make_input_shape_fixed(model.graph, model.graph.input[0].name, shape)
    onnx.save(model, str(dst))
    return dst


class ImageReader:
    """Calibration data: rendered demo screens, normalised like RapidOCR."""

    def __init__(self, input_name: str, shape: list[int], kind: str):
        from sajag.demo import SCREENS, render_screen

        self.name = input_name
        self.items = []
        for screen in SCREENS:
            img = render_screen(screen, size=(shape[3], shape[2]) if kind == "det" else (1280, 720))
            arr = np.asarray(img, dtype=np.float32)[:, :, ::-1] / 255.0  # BGR, like RapidOCR
            if kind == "det":
                arr = (arr - 0.5) / 0.5  # PP-OCR det normalisation used by RapidOCR
                self.items.append(arr.transpose(2, 0, 1)[None].astype(np.float32))
            else:
                for y in range(60, 600, 60):  # text-line crops
                    crop = img.crop((100, y, 100 + 640, y + 96)).resize((shape[3], shape[2]))
                    c = (np.asarray(crop, dtype=np.float32)[:, :, ::-1] / 255.0 - 0.5) / 0.5
                    self.items.append(c.transpose(2, 0, 1)[None].astype(np.float32))
        self.it = iter(self.items)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.name: x}

    def rewind(self):
        self.it = iter(self.items)


def quantize_qnn(src: Path, dst: Path, kind: str, shape: list[int]) -> Path:
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

    pre = src.with_suffix(".preproc.onnx")
    try:
        model_in = pre if qnn_preprocess_model(str(src), str(pre)) else src
    except Exception as exc:  # some ORT versions fail on models with metadata_props
        print(f"  qnn_preprocess_model skipped for {src.name}: {exc}")
        model_in = src
    name = onnx.load(str(model_in)).graph.input[0].name
    cfg = get_qnn_qdq_config(str(model_in), ImageReader(name, shape, kind),
                             activation_type=QuantType.QUInt16, weight_type=QuantType.QUInt8)
    quantize(str(model_in), str(dst), cfg)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".sajag" / "models" / "ocr"))
    ap.add_argument("--quantize", action="store_true",
                    help="experimental: write QDQ w8a16 models (needs a representative calibration set)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    det, rec = rapidocr_models()
    for kind, src, shape in (("det", det, DET_SHAPE), ("rec", rec, REC_SHAPE)):
        fixed = fix_shape(src, out / f"{kind}_fixed.onnx", shape)
        final = out / f"{kind}_npu.onnx"
        if args.quantize:
            quantize_qnn(fixed, final, kind, shape)
        else:
            fixed.replace(final)  # fp32 graph; the HTP executes it in fp16 (enable_htp_fp16_precision=1)
        print(f"{kind}: {src.name} -> {final}  shape={shape}  quantized={args.quantize}")


if __name__ == "__main__":
    main()
