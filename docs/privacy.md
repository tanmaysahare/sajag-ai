# Privacy and safety by design

* **No network at runtime.** After the one-time model download, Sajag makes no network calls. The dashboard binds to 127.0.0.1 and loads no external scripts, fonts or images.
* **Nothing written to disk.** Audio and screen frames live in RAM ring buffers and are discarded after analysis. Transcripts are kept in memory only for the last 15 minutes of fusion; `store_transcripts` is off by default.
* **Minimum necessary processing.** Speech recognition only runs while a call app is active and only on segments that the VAD marks as speech. OCR only runs when the screen changes.
* **Explainable verdicts.** Every alert lists the exact phrases and sources behind it, so the user can judge for themselves. The optional LLM only rephrases evidence; it never decides the verdict.
* **User stays in control.** Sajag warns and informs. It never blocks calls, kills apps or moves money on its own.
* **Why not the cloud?** Uploading screen and call content to detect scams would create exactly the kind of sensitive data trove scammers want. Running on the Snapdragon NPU removes that risk entirely.
* **Legal context.** Processing stays on the user's device for the user's own protection, which aligns with the data-minimisation principles of India's Digital Personal Data Protection Act, 2023.
