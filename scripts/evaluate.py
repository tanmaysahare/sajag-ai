"""End-to-end evaluation of the Sajag risk pipeline.

Three views, all written to docs/evaluation_pipeline.json:

1. Segment level on fresh random fills + ASR-style noise of every template
   (templates overlap with the classifier's training data, so this is an
   upper bound; see docs/evaluation.json for the template-held-out ML score).
2. Conversation level: 400 synthetic calls. Scam calls interleave 3-6 lines
   of one scam family with small talk; benign calls mix everyday lines with
   hard negatives (bank SMS footers, police awareness messages, IT support).
   We report detection rate (reaching WARNING or DANGER), false-alarm rate,
   and how many turns it took to raise the first WARNING.
3. The five scripted demo scenarios, end to end through OCR.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_corpus import BENIGN, SCAM, augment, fill  # noqa: E402

from sajag.guardian import Guardian  # noqa: E402
from sajag.risk.engine import TextRiskAnalyzer  # noqa: E402
from sajag.risk.fusion import LEVEL_ORDER, RiskFusion  # noqa: E402


def segment_eval(an: TextRiskAnalyzer, rng: random.Random) -> dict:
    tp = fp = tn = fn = 0
    for fam, temps in list(SCAM.items()) + [("benign", BENIGN)]:
        for tpl in temps:
            for _ in range(3):
                s = an.analyze(augment(fill(tpl, rng), rng)).score
                pos = s >= 0.35
                if fam == "benign":
                    fp += pos
                    tn += not pos
                else:
                    tp += pos
                    fn += not pos
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return {"threshold": 0.35, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0, "n": tp + fp + tn + fn}


def conversation_eval(an: TextRiskAnalyzer, rng: random.Random, n: int = 400) -> dict:
    small_talk = [b for b in BENIGN if "otp" not in b.lower() and "police" not in b.lower()]
    det, fa, turns, n_scam, n_benign = 0, 0, [], 0, 0
    per_family: dict[str, list[int]] = {}
    for i in range(n):
        fusion = RiskFusion()
        t = 1000.0
        scam = i % 2 == 0
        if scam:
            fam = rng.choice(sorted(SCAM))
            lines = [fill(x, rng) for x in rng.sample(SCAM[fam], k=min(len(SCAM[fam]), rng.randint(3, 6)))]
            convo = []
            for ln in lines:
                if rng.random() < 0.4:
                    convo.append(("benign", fill(rng.choice(small_talk), rng)))
                convo.append(("scam", augment(ln, rng)))
        else:
            fam = "benign"
            convo = [("benign", augment(fill(rng.choice(BENIGN), rng), rng)) for _ in range(rng.randint(4, 8))]
        first = None
        for k, (_, text) in enumerate(convo, start=1):
            t += rng.uniform(5, 15)
            fusion.add_finding(an.analyze(text, source="call", timestamp=t))
            st = fusion.evaluate(t)
            if LEVEL_ORDER[st.level] >= LEVEL_ORDER["warning"] and first is None:
                first = k
        if scam:
            n_scam += 1
            hit = first is not None
            det += hit
            per_family.setdefault(fam, []).append(int(hit))
            if hit:
                turns.append(first)
        else:
            n_benign += 1
            fa += first is not None
    turns.sort()
    return {
        "scam_calls": n_scam,
        "benign_calls": n_benign,
        "detection_rate": round(det / n_scam, 4),
        "false_alarm_rate": round(fa / n_benign, 4),
        "median_turns_to_warning": turns[len(turns) // 2] if turns else None,
        "per_family_detection": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_family.items())},
    }


def heldout_pipeline_eval(n: int = 400, seed: int = 99) -> dict:
    """Full pipeline on templates the classifier never saw (the honest generalisation test).

    25% of templates per class are withheld; the intent classifier is retrained on
    the rest; conversations are then built only from the withheld templates.
    """
    import tempfile

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from sajag.npu import SessionFactory
    from sajag.risk.classifier import IntentClassifier
    from train_classifier import export_onnx, matrix

    rng = random.Random(seed)
    labels = ["benign"] + sorted(SCAM)
    train_rows, held = [], {}
    for fam, temps in list(SCAM.items()) + [("benign", BENIGN)]:
        temps = list(temps)
        rng.shuffle(temps)
        k = max(2, len(temps) // 4)
        held[fam], kept = temps[:k], temps[k:]
        for tpl in kept:
            for _ in range(12 if fam != "benign" else 24):
                train_rows.append((augment(fill(tpl, rng), rng), fam))
    clf = LogisticRegression(C=8.0, max_iter=2000)
    clf.fit(matrix([t for t, _ in train_rows]), np.array([labels.index(l) for _, l in train_rows]))
    tmp = Path(tempfile.mkdtemp())
    export_onnx(clf, labels, tmp / "clf.onnx")
    (tmp / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    an = TextRiskAnalyzer(use_classifier=False)
    an.classifier = IntentClassifier(factory=SessionFactory(prefer_npu=False), model_path=tmp / "clf.onnx",
                                     labels_path=tmp / "labels.json")
    small_talk = [b for b in held["benign"] if "otp" not in b.lower()] or held["benign"]
    det = fa = n_s = n_b = 0
    turns = []
    for i in range(n):
        fusion, t = RiskFusion(), 1000.0
        if i % 2 == 0:
            fam = rng.choice(sorted(SCAM))
            convo = []
            for _ in range(rng.randint(3, 6)):
                if rng.random() < 0.4:
                    convo.append(fill(rng.choice(small_talk), rng))
                convo.append(augment(fill(rng.choice(held[fam]), rng), rng))
        else:
            fam = "benign"
            convo = [augment(fill(rng.choice(held["benign"]), rng), rng) for _ in range(rng.randint(4, 8))]
        first = None
        for k, text in enumerate(convo, 1):
            t += rng.uniform(5, 15)
            fusion.add_finding(an.analyze(text, source="call", timestamp=t))
            if first is None and LEVEL_ORDER[fusion.evaluate(t).level] >= 2:
                first = k
        if fam == "benign":
            n_b += 1
            fa += first is not None
        else:
            n_s += 1
            det += first is not None
            if first:
                turns.append(first)
    turns.sort()
    return {"held_out_templates": {k: len(v) for k, v in held.items()}, "scam_calls": n_s, "benign_calls": n_b,
            "detection_rate": round(det / n_s, 4), "false_alarm_rate": round(fa / n_b, 4),
            "median_turns_to_warning": turns[len(turns) // 2] if turns else None}


def scenario_eval() -> dict:
    from sajag.demo import SCENARIOS, run_scenario

    g = Guardian()
    out = {}
    for name, sc in SCENARIOS.items():
        trace = run_scenario(name, g, use_ocr=True)
        first_warning = next((r["at"] for r in trace if LEVEL_ORDER[r["level"]] >= 2), None)
        out[name] = {"expected": sc["expected"], "final": trace[-1]["level"], "score": round(trace[-1]["score"], 3),
                     "pass": trace[-1]["level"] == sc["expected"], "first_warning_at_s": first_warning}
    return out


def main() -> None:
    rng = random.Random(2026)
    an = TextRiskAnalyzer()
    res = {
        "segment_level": segment_eval(an, rng),
        "conversation_level": conversation_eval(an, rng),
        "heldout_pipeline": heldout_pipeline_eval(),
        "scenarios": scenario_eval(),
        "note": "Synthetic evaluation on template families that overlap the training corpus (upper bound). Real-call validation with consented recordings is future work.",
    }
    (ROOT / "docs" / "evaluation_pipeline.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
