"""Whisper log-mel front-end in pure NumPy (no torch at runtime).

Matches openai/whisper `log_mel_spectrogram`: 16 kHz, n_fft=400, hop=160,
periodic Hann window, Slaney mel filters, log10, dynamic-range clamp of 8 dB
and (x + 4) / 4 scaling. Audio is padded / trimmed to 30 s -> 3000 frames,
which is the static encoder input shape of the AI Hub Whisper models.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
CHUNK_S = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_S
N_FRAMES = N_SAMPLES // HOP  # 3000


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


@lru_cache(maxsize=4)
def mel_filters(n_mels: int = 80, sr: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """Slaney-normalised mel filterbank, identical to librosa.filters.mel defaults."""
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_pts[2: n_mels + 2] - mel_pts[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if audio.shape[0] > length:
        return audio[:length]
    if audio.shape[0] < length:
        return np.pad(audio, (0, length - audio.shape[0]))
    return audio


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Return float32 [n_mels, 3000] features for one 30 s window."""
    audio = pad_or_trim(np.asarray(audio, dtype=np.float32))
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * window[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    mag = (np.abs(spec) ** 2).T[:, :-1]  # drop last frame like torch.stft + [..., :-1]
    mel = mel_filters(n_mels) @ mag
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[:, :N_FRAMES].astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly

    g = np.gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
