"""ONNX Runtime session factory with Qualcomm QNN (Hexagon NPU) support.

On a Snapdragon X / X2 PC with ``onnxruntime-qnn`` installed, every model is
placed on the Hexagon NPU through the QNN Execution Provider (HTP backend).
Elsewhere, the same code transparently falls back to the CPU provider so the
project can be developed and tested on any machine.

Key NPU techniques used here (see docs/npu-optimization.md):

* **Per-workload HTP power modes.** Always-on sentinels (VAD, screen OCR) run in
  ``low_power_saver`` / ``power_saver``; the ASR decoder switches to ``burst``
  only while a call is active. This is what keeps an always-on guardian cheap
  on battery.
* **EP context caching.** The first run compiles the graph for the HTP and
  writes a ``*_ctx.onnx`` context binary next to the model; later launches load
  the pre-compiled binary and start in milliseconds instead of seconds.
* **No silent CPU fallback in strict mode.** ``SAJAG_NPU_STRICT=1`` sets
  ``session.disable_cpu_ep_fallback`` so any operator the NPU cannot run raises
  an error during development instead of quietly running on the CPU.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("sajag.npu")

try:  # onnxruntime or onnxruntime-qnn (both expose the same module name)
    import onnxruntime as ort
except ImportError:  # pragma: no cover - import guard for docs builds
    ort = None  # type: ignore[assignment]

# Power profiles map a *workload* to an HTP performance mode.
POWER_PROFILES: dict[str, str] = {
    "always_on": "low_power_saver",  # VAD, periodic screen OCR
    "background": "power_saver",  # classifier, embeddings
    "interactive": "high_performance",  # OCR burst after a screen change
    "burst": "burst",  # ASR while a call is live, LLM explanation
}


@dataclass
class SessionInfo:
    name: str
    path: str
    provider: str
    power_mode: str | None
    load_ms: float
    inputs: list[tuple[str, list, str]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    runs: int = 0
    total_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.runs if self.runs else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "provider": self.provider,
            "power_mode": self.power_mode,
            "load_ms": round(self.load_ms, 1),
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 3),
        }


def available_providers() -> list[str]:
    if ort is None:
        return []
    return list(ort.get_available_providers())


def qnn_available() -> bool:
    return "QNNExecutionProvider" in available_providers()


def is_windows_arm64() -> bool:
    return platform.system() == "Windows" and platform.machine().upper() in {"ARM64", "AARCH64"}


def device_summary() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnxruntime": getattr(ort, "__version__", None),
        "providers": available_providers(),
        "qnn_npu": qnn_available(),
        "windows_arm64": is_windows_arm64(),
    }


class TimedSession:
    """Thin wrapper around ``InferenceSession`` that records latency."""

    def __init__(self, session: "ort.InferenceSession", info: SessionInfo):
        self.session = session
        self.info = info
        self._lock = threading.Lock()

    def run(self, feeds: dict[str, np.ndarray], output_names: list[str] | None = None) -> list[np.ndarray]:
        t0 = time.perf_counter()
        with self._lock:  # QNN sessions are not re-entrant across threads
            out = self.session.run(output_names, feeds)
        dt = (time.perf_counter() - t0) * 1000.0
        self.info.runs += 1
        self.info.total_ms += dt
        return out

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()


class SessionFactory:
    """Creates and caches ORT sessions, preferring the Hexagon NPU."""

    def __init__(self, prefer_npu: bool = True, strict: bool | None = None, cache_dir: str | os.PathLike | None = None,
                 enable_context_cache: bool = True):
        self.prefer_npu = prefer_npu
        self.strict = bool(int(os.environ.get("SAJAG_NPU_STRICT", "0"))) if strict is None else strict
        self.cache_dir = Path(cache_dir or os.environ.get("SAJAG_CACHE", Path.home() / ".sajag" / "qnn_ctx"))
        self.enable_context_cache = enable_context_cache
        self.sessions: dict[str, TimedSession] = {}

    # ------------------------------------------------------------------
    def _qnn_options(self, power_mode: str) -> dict[str, str]:
        opts = {
            "htp_performance_mode": power_mode,
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",  # fp32 graphs run in fp16 on HTP
        }
        if os.name == "nt":
            opts["backend_path"] = "QnnHtp.dll"
        else:
            opts["backend_type"] = "htp"
        return opts

    def create(self, name: str, model_path: str | os.PathLike, workload: str = "background",
               intra_op_threads: int = 2) -> TimedSession:
        """Create (or reuse) a session for ``model_path``.

        ``workload`` selects the HTP power mode (see ``POWER_PROFILES``).
        """
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        model_path = Path(model_path)
        key = f"{name}:{model_path}"
        if key in self.sessions:
            return self.sessions[key]
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path} (run `python -m sajag fetch-models`)")

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3

        providers: list = []
        power_mode = None
        load_path = str(model_path)
        use_qnn = self.prefer_npu and qnn_available()
        if use_qnn:
            power_mode = POWER_PROFILES.get(workload, "balanced")
            if self.strict:
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            # Pre-compiled AI Hub assets (*_ctx.onnx / precompiled_qnn_onnx) are loaded as-is.
            is_precompiled = model_path.name.endswith("_ctx.onnx") or "precompiled" in str(model_path)
            if self.enable_context_cache and not is_precompiled:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                ctx_path = self.cache_dir / f"{model_path.stem}_ctx.onnx"
                if ctx_path.exists():
                    load_path = str(ctx_path)
                else:
                    so.add_session_config_entry("ep.context_enable", "1")
                    so.add_session_config_entry("ep.context_file_path", str(ctx_path))
                    so.add_session_config_entry("ep.context_embed_mode", "1")
            providers.append(("QNNExecutionProvider", self._qnn_options(power_mode)))
        providers.append("CPUExecutionProvider")

        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(load_path, sess_options=so, providers=providers)
        except Exception as exc:  # NPU compile failure -> CPU, unless strict
            if use_qnn and not self.strict:
                log.warning("QNN session for %s failed (%s); falling back to CPU", name, exc)
                so = ort.SessionOptions()
                so.intra_op_num_threads = intra_op_threads
                sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
                power_mode = None
            else:
                raise
        load_ms = (time.perf_counter() - t0) * 1000.0

        active = sess.get_providers()[0]
        info = SessionInfo(
            name=name,
            path=str(model_path),
            provider=active,
            power_mode=power_mode if active == "QNNExecutionProvider" else None,
            load_ms=load_ms,
            inputs=[(i.name, list(i.shape), i.type) for i in sess.get_inputs()],
            outputs=[o.name for o in sess.get_outputs()],
        )
        log.info("session %-18s provider=%s power=%s load=%.0fms", name, active, info.power_mode, load_ms)
        ts = TimedSession(sess, info)
        self.sessions[key] = ts
        return ts

    def report(self) -> list[dict]:
        return [s.info.as_dict() for s in self.sessions.values()]


_default_factory: SessionFactory | None = None


def default_factory() -> SessionFactory:
    global _default_factory
    if _default_factory is None:
        _default_factory = SessionFactory()
    return _default_factory
