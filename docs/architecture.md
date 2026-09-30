# Sajag AI architecture

## Data flow

1. **Sentinels** observe three channels in parallel threads.
   * *Call audio*: WASAPI loopback (the remote party, from any call app) and the microphone, 16 kHz mono, kept in a RAM ring buffer. Audio is only processed while the system sentinel reports an active call app.
   * *Screen*: primary monitor captured every 2 s with `mss`, down-scaled to 1600 px. A 64-bit difference hash is compared with the last processed frame; OCR runs only if 6 or more bits changed.
   * *System*: process list every 3 s (remote-access tools, call apps) and the foreground window title.
2. **Perception on the NPU**.
   * Silero-VAD (AI Hub) cuts speech segments (threshold 0.5, hysteresis 0.35, 250 ms minimum speech, 400 ms minimum silence, 28 s maximum).
   * Whisper-Small (AI Hub, precompiled QNN ONNX) transcribes each segment. The decoder prompt is forced to `<|startoftranscript|><|hi|><|transcribe|><|notimestamps|>` (or `<|en|>`), which keeps Hinglish in its original words instead of translating it.
   * PP-OCRv4 detection and recognition run through a static-shape adapter (see npu-optimization.md).
3. **Text risk analysis** per segment / screen (`sajag/risk/engine.py`).
   * Normalisation (NFKC, lower-case, punctuation, common ASR/OCR confusions such as "any desk" to "anydesk").
   * 300+ cues in English, Devanagari Hindi and Hinglish, grouped into 10 tactics: authority, threat, urgency, secrecy, payment, credential, remote_access, lure, fake_error, identity_bait. Latin phrases match on word boundaries; Devanagari phrases match as substrings. The most specific (longest) matching phrase is kept as evidence.
   * Awareness damping: sentences such as "Police never arrest anyone over a video call" or bank SMS footers contain scam words but lower the score.
   * ONNX intent classifier: 16,384 hashed character 2 to 4-grams plus word unigrams, logistic regression over 8 classes (benign + 7 families), exported as Gemm + Softmax.
   * Segment score = 1 - (1 - pattern) x (1 - 0.6 x classifier).
4. **Temporal cross-modal fusion** (`sajag/risk/fusion.py`).
   * Each tactic hit becomes evidence with a timestamp, a source (call / screen / system) and a weight scaled by the classifier's confidence in that sentence. During a live call, credential, remote-access and secrecy evidence keep at least 60% of its weight.
   * Evidence decays with a 4-minute half-life inside a 15-minute window. The conversation score is a noisy-OR over the strongest evidence per tactic, combined with the best recent segment score.
   * Escalation rules encode the signatures investigators describe: remote access plus OTP or money (0.85), secrecy plus authority plus money (0.90, digital arrest), prize or refund that must be paid for (0.68, advance-fee fraud), and a +0.08 bonus when two or more modalities agree during a call.
   * Levels: SAFE < 0.35 <= CAUTION < 0.60 <= WARNING < 0.80 <= DANGER. Alerts fire once per upward level change.
   * Family attribution: coverage of each family's tactic signature plus votes from segment-level family scores.
5. **Explanation and alerting**. Bilingual reasons and family-specific advice (always), optional Llama-3.2-3B rephrasing through the Genie SDK, full-screen interrupt for DANGER, spoken warning and toast for WARNING, live dashboard on 127.0.0.1.

## Heterogeneous compute map

| Stage | Unit | Why |
|---|---|---|
| Screen capture, dHash, process scan | CPU (efficiency cores) | Microseconds of work, no model |
| Silero-VAD | NPU, `low_power_saver` | Runs 31 times per second while a call is active |
| PP-OCRv4 det + rec | NPU, `high_performance` burst after a screen change | Convolution-heavy, ideal for HTP |
| Whisper-Small encoder + decoder | NPU, `burst` only during calls | The heaviest model; AI Hub reports 100% of layers on NPU |
| Intent classifier | NPU or CPU | 1 Gemm, either is fine |
| Pattern engine, fusion | CPU | Branchy logic, microseconds |
| Llama-3.2-3B (optional) | NPU via Genie, `burst` | Only after a WARNING, a few dozen tokens |

## Threading

Each sentinel is a daemon thread. `TimedSession.run` holds a per-session lock because QNN sessions are not re-entrant; different models run concurrently. The guardian serialises updates to the fusion state with one lock and publishes events to the dashboard over a WebSocket.

## Third-party models

| Model | License |
|---|---|
| Silero-VAD | MIT |
| OpenAI Whisper (via Qualcomm AI Hub) | MIT (weights), AI Hub assets under Qualcomm AI Hub terms |
| PP-OCRv4 (PaddleOCR) / RapidOCR | Apache-2.0 |
| Llama-3.2-3B-Instruct | Llama 3.2 Community License |
| Sajag intent classifier | Apache-2.0 (this repo) |
