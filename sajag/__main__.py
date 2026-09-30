"""Sajag AI command line.

  python -m sajag run            live guardian + dashboard at http://127.0.0.1:8765
  python -m sajag demo [name]    replay a scripted scam scenario in the console
  python -m sajag ui             dashboard with scenario buttons (no live capture)
  python -m sajag scan-text "..."    analyse one sentence
  python -m sajag scan-image file    OCR + analyse a screenshot
  python -m sajag doctor         show NPU / provider / model status
  python -m sajag bench          benchmark models on NPU and CPU
  python -m sajag fetch-models   download AI Hub models (one time, then fully offline)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__


def _guardian(args, alerts: bool = True):
    from .alerts.notifier import Notifier
    from .guardian import Guardian, GuardianConfig
    from .models import find_vad_model, locate

    wd = locate("whisper")
    cfg = GuardianConfig(language=getattr(args, "lang", "hi"), whisper_dir=str(wd) if wd else None,
                         vad_model=str(find_vad_model() or "") or None, alerts=alerts)
    return Guardian(cfg, notifier=Notifier(lang="hi" if getattr(args, "lang", "en") == "hi" else "en"))


def cmd_demo(args) -> None:
    from .demo import SCENARIOS, run_scenario

    names = [args.name] if args.name else list(SCENARIOS)
    g = _guardian(args, alerts=False)
    icons = {"safe": "  ", "caution": "! ", "warning": "!!", "danger": "XX"}
    for n in names:
        print(f"\n=== {SCENARIOS[n]['title']} (expected: {SCENARIOS[n]['expected']})")
        for row in run_scenario(n, g, use_ocr=not args.no_ocr):
            print(f"{icons[row['level']]} t={row['at']:>3}s  risk={row['score']:.2f}  {row['level']:<8} {row['what'][:90]}")
        st = g.state
        print(f"--> verdict: {st.level.upper()} | {st.family_name or '-'}")
        for r in st.reasons[:4]:
            print("    because:", r)
        print("    advice :", st.advice)


def cmd_scan_text(args) -> None:
    from .risk.engine import TextRiskAnalyzer

    f = TextRiskAnalyzer().analyze(args.text)
    print(json.dumps(f.as_dict(), ensure_ascii=False, indent=2))


def cmd_scan_image(args) -> None:
    from PIL import Image

    from .risk.engine import TextRiskAnalyzer
    from .vision.ocr import ScreenOCR

    text = ScreenOCR().read_text(Image.open(args.path))
    f = TextRiskAnalyzer().analyze(text, source="screen")
    print(json.dumps(f.as_dict(), ensure_ascii=False, indent=2))


def cmd_doctor(_args) -> None:
    from .models import MODEL_HOME, REGISTRY, locate
    from .npu import device_summary

    print(json.dumps(device_summary(), indent=2))
    print(f"\nmodels in {MODEL_HOME}:")
    for k, spec in REGISTRY.items():
        status = "bundled" if spec.source in ("bundled", "pip") else ("ok" if locate(k) else "missing")
        print(f"  {k:<14} {status:<8} {spec.role}")


def cmd_bench(args) -> None:
    from .bench import run

    res = run(out=args.out, iters=args.iters)
    for m in res["models"]:
        print(f"{m['model']:<44} {m['target']:<4} {m['provider']:<24} p50={m['p50_ms']:>8} ms  p90={m['p90_ms']:>8} ms")


def cmd_fetch(args) -> None:
    from .models import fetch

    print(json.dumps(fetch(args.models, chipset=args.chipset), indent=2))


def cmd_run(args) -> None:
    from .ui.server import serve

    g = _guardian(args)
    g.config.audio = not args.no_audio
    g.config.screen = not args.no_screen
    g.start()
    print(f"Sajag AI {__version__} running. Dashboard: http://127.0.0.1:{args.port}")
    serve(g, port=args.port)


def cmd_ui(args) -> None:
    from .ui.server import serve

    g = _guardian(args, alerts=False)
    print(f"Sajag AI dashboard (demo mode): http://127.0.0.1:{args.port}")
    serve(g, port=args.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="sajag", description="On-device scam shield for Snapdragon AI PCs")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run"); s.add_argument("--port", type=int, default=8765); s.add_argument("--lang", default="hi")
    s.add_argument("--no-audio", action="store_true"); s.add_argument("--no-screen", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("ui"); s.add_argument("--port", type=int, default=8765); s.set_defaults(fn=cmd_ui)
    s = sub.add_parser("demo"); s.add_argument("name", nargs="?"); s.add_argument("--no-ocr", action="store_true")
    s.set_defaults(fn=cmd_demo)
    s = sub.add_parser("scan-text"); s.add_argument("text"); s.set_defaults(fn=cmd_scan_text)
    s = sub.add_parser("scan-image"); s.add_argument("path"); s.set_defaults(fn=cmd_scan_image)
    s = sub.add_parser("doctor"); s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("bench"); s.add_argument("--out"); s.add_argument("--iters", type=int, default=30)
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("fetch-models"); s.add_argument("--models", nargs="*", default=["vad", "whisper"])
    s.add_argument("--chipset", default="x_elite", choices=["x_elite", "x2_elite"]); s.set_defaults(fn=cmd_fetch)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
