# Running Sajag on the Hexagon NPU

Sajag uses **ONNX Runtime with the Qualcomm QNN Execution Provider** (`onnxruntime-qnn`, HTP backend). Every model is created through `sajag.npu.SessionFactory`, so the same code runs on the NPU on a Snapdragon PC and on the CPU anywhere else.

## 1. Pre-compiled AI Hub assets where they exist

Silero-VAD and Whisper-Small come from Qualcomm AI Hub, already converted to static shapes, with attention rewritten for the NPU (single-head attention, linear layers as convolutions) and compiled for the Snapdragon X Elite. AI Hub's profile of `whisper_small` on the X Elite CRD reports all 1,582 encoder layers and all 2,277 decoder layers on the NPU. Sajag ports AI Hub's reference decode loop to NumPy so the runtime needs neither PyTorch nor Transformers.

## 2. Static shapes for open-source models

The QNN EP does not accept dynamic input shapes. `scripts/prepare_npu_models.py` pins PP-OCRv4 to `[1,3,736,1280]` (detector) and `[1,3,48,1280]` (recogniser). `StaticShapeAdapter` then:

* forces the detector to resize every screen into the compiled canvas (long side 1280),
* zero-pads each input to the compiled shape and crops the output back (probability map for the detector, CTC time steps for the recogniser),
* switches the recogniser to one line per call so only unusually long lines use the CPU session.

Verified on CPU: the static-shape graph through the adapter reproduces the dynamic-shape text on the demo screens.

## 3. Precision

* Default: the fp32 graph with `enable_htp_fp16_precision=1`, so the HTP executes in fp16.
* Optional: `--quantize` writes QDQ models with uint16 activations and uint8 weights using `get_qnn_qdq_config`. Calibrate on screenshots of the target desktop; our small synthetic calibration set was not representative enough, so fp16 is the default.

## 4. Power-aware scheduling

| Workload | HTP performance mode | Used by |
|---|---|---|
| `always_on` | `low_power_saver` | Silero-VAD |
| `background` | `power_saver` | intent classifier |
| `interactive` | `high_performance` | OCR after a screen change |
| `burst` | `burst` | Whisper during a live call, optional LLM |

Two gates keep the NPU idle most of the time: the dHash screen gate (static screens are never OCR'd) and the call gate (no ASR unless a call app is running, and only on VAD speech segments).

## 5. Fast start with EP context caching

On first launch the QNN EP compiles each graph for the HTP and writes a context binary (`ep.context_enable=1`, embedded mode) to `~/.sajag/qnn_ctx/`. Later launches load the pre-compiled binary directly.

## 6. Strict mode for development

`SAJAG_NPU_STRICT=1` sets `session.disable_cpu_ep_fallback=1`. Any operator the HTP cannot run then raises an error instead of silently running on the CPU. `python -m sajag bench` reports the provider that actually executed each model.

## Measuring on your device

```powershell
python -m sajag doctor
python -m sajag bench --out bench_x_elite.json
```
