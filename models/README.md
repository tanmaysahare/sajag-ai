# Models

Neural models are not stored in git. `python -m sajag fetch-models` downloads them once into `~/.sajag/models/`:

| Folder | Qualcomm AI Hub model | Runtime / precision |
|---|---|---|
| `vad/` | `silero_vad` | ONNX, w8a16_mixed_int16 |
| `whisper/` | `whisper_small` (+ `multilingual.tiktoken`) | precompiled_qnn_onnx, float, chipset qualcomm-snapdragon-x-elite |

Equivalent CLI: `qai-hub-models fetch whisper_small --runtime precompiled_qnn_onnx --precision float --chipset qualcomm-snapdragon-x-elite`

The OCR models ship inside `rapidocr-onnxruntime`; `python scripts/prepare_npu_models.py` writes static-shape NPU versions to `~/.sajag/models/ocr/`.
The scam-intent classifier is small enough to live in the repo: `sajag/assets/intent_classifier.onnx`.
