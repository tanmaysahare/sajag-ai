"""Train the scam-intent classifier and export it as a static-shape ONNX model.

Pipeline: hashed char n-grams (sajag.risk.classifier.featurize)
          -> multinomial logistic regression (scikit-learn)
          -> ONNX graph  input[1,16384] -> Gemm -> Softmax  (NPU friendly)

Evaluation is written to docs/evaluation.json. Two test sets are reported:
  1. a held-out split of the synthetic corpus (template families seen in training)
  2. template-held-out: whole templates withheld from training, which is a
     much harder and more honest measure of generalisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_corpus import BENIGN, SCAM, augment, build, fill  # noqa: E402
from sajag.risk.classifier import LABELS_PATH, MODEL_PATH, N_FEATURES, featurize  # noqa: E402


def matrix(texts):
    rows = [sparse.csr_matrix(featurize(t)) for t in texts]
    return sparse.vstack(rows).tocsr()


def export_onnx(clf: LogisticRegression, labels: list[str], path: Path) -> None:
    W = clf.coef_.astype(np.float32).T  # [F, C]
    b = clf.intercept_.astype(np.float32)  # [C]
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["features", "W", "b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
        ],
        name="sajag_intent_classifier",
        inputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, N_FEATURES])],
        outputs=[helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, len(labels)])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist()),
            helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="sajag")
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def template_holdout(seed: int = 11):
    """Withhold ~25% of templates per class entirely from training."""
    import random

    rng = random.Random(seed)
    train, test = [], []
    for fam, temps in list(SCAM.items()) + [("benign", BENIGN)]:
        temps = list(temps)
        rng.shuffle(temps)
        k = max(1, len(temps) // 4)
        held, kept = temps[:k], temps[k:]
        for tpl in kept:
            for _ in range(12 if fam != "benign" else 24):
                train.append((augment(fill(tpl, rng), rng), fam))
        for tpl in held:
            for _ in range(6):
                test.append((augment(fill(tpl, rng), rng), fam))
    return train, test


def main() -> None:
    rows = build()
    labels = ["benign"] + sorted(SCAM)
    texts = [r["text"] for r in rows]
    y = np.array([labels.index(r["label"]) for r in rows])
    n = len(rows)
    split = int(n * 0.8)
    Xtr, Xte = matrix(texts[:split]), matrix(texts[split:])
    ytr, yte = y[:split], y[split:]

    clf = LogisticRegression(C=8.0, max_iter=2000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    report_random = classification_report(yte, pred, target_names=labels, output_dict=True, zero_division=0)

    # binary scam-vs-benign view of the random split
    yb, pb = (yte != 0), (pred != 0)
    p, r, f, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)

    # template-held-out generalisation test
    tr, te = template_holdout()
    clf_h = LogisticRegression(C=8.0, max_iter=2000)
    clf_h.fit(matrix([t for t, _ in tr]), [labels.index(l) for _, l in tr])
    ph = clf_h.predict(matrix([t for t, _ in te]))
    yh = np.array([labels.index(l) for _, l in te])
    hp, hr, hf, _ = precision_recall_fscore_support(yh != 0, ph != 0, average="binary", zero_division=0)
    macro_h = f1_score(yh, ph, average="macro")

    # final model on everything
    clf_all = LogisticRegression(C=8.0, max_iter=2000)
    clf_all.fit(matrix(texts), y)
    export_onnx(clf_all, labels, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "corpus_rows": n,
        "labels": labels,
        "random_split": {
            "test_rows": int(len(yte)),
            "macro_f1": round(report_random["macro avg"]["f1-score"], 4),
            "scam_vs_benign": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)},
        },
        "template_holdout": {
            "test_rows": int(len(yh)),
            "macro_f1": round(float(macro_h), 4),
            "scam_vs_benign": {"precision": round(hp, 4), "recall": round(hr, 4), "f1": round(hf, 4)},
        },
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "input": [1, N_FEATURES], "bytes": MODEL_PATH.stat().st_size},
        "note": "Synthetic corpus. Template-holdout is the honest generalisation estimate; real-call validation is future work.",
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
