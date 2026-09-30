"""Per-segment text risk analysis (one ASR segment or one OCR screen)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .normalize import normalize, phrase_in
from .patterns import CUES, FAMILIES, TACTICS

log = logging.getLogger("sajag.risk")

# Genuine awareness messages ("Police never ask for OTP", bank SMS footers)
# contain the same keywords as scams. They reduce, not raise, suspicion.
AWARENESS_PHRASES = (
    "never ask", "will never ask", "never asks", "never call", "beware of", "fraud alert", "awareness",
    "do not share your otp with anyone", "don't share your otp with anyone", "report cyber fraud",
    "is a scam", "scam alert", "stay safe", "kabhi nahi maangta", "kabhi nahi maangte",
    "कभी नहीं माँगते", "कभी नहीं मांगते", "सावधान रहें", "जागरूक", "do not share it with anyone",
    "don't share it with anyone", "never share", "न बताएं", "कभी ओटीपी नहीं", "is valid for",
)
AWARENESS_DAMPING = 0.35


@dataclass
class TacticHit:
    tactic: str
    phrase: str
    weight: float
    families: tuple[str, ...] = ()


@dataclass
class Finding:
    source: str  # "call" | "screen" | "system" | "text"
    text: str
    hits: list[TacticHit] = field(default_factory=list)
    pattern_score: float = 0.0
    ml_score: float = 0.0
    ml_family: str | None = None
    score: float = 0.0
    families: dict[str, float] = field(default_factory=dict)
    awareness: bool = False
    timestamp: float = 0.0

    @property
    def tactics(self) -> set[str]:
        return {h.tactic for h in self.hits}

    def top_family(self) -> str | None:
        if not self.families:
            return None
        fam, val = max(self.families.items(), key=lambda kv: kv[1])
        return fam if val > 0 else None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "score": round(self.score, 3),
            "pattern_score": round(self.pattern_score, 3),
            "ml_score": round(self.ml_score, 3),
            "ml_family": self.ml_family,
            "family": self.top_family(),
            "tactics": sorted(self.tactics),
            "evidence": [{"tactic": h.tactic, "phrase": h.phrase} for h in self.hits],
            "awareness": self.awareness,
            "timestamp": self.timestamp,
        }


def noisy_or(weights) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(0.95, w))
    return 1.0 - p


def family_scores(tactic_weights: dict[str, float], tagged: dict[str, float]) -> dict[str, float]:
    """Coverage of each family's signature by the observed tactics (0..1)."""
    out: dict[str, float] = {}
    for fam in FAMILIES.values():
        sig = fam.signature
        denom = sum(TACTICS[t]["weight"] for t in sig)
        num = sum(TACTICS[t]["weight"] for t in sig if t in tactic_weights)
        cov = num / denom if denom else 0.0
        out[fam.id] = min(1.0, cov + tagged.get(fam.id, 0.0))
    return out


class TextRiskAnalyzer:
    """Pattern engine + optional ONNX intent classifier."""

    def __init__(self, use_classifier: bool = True, factory=None):
        self.classifier = None
        if use_classifier:
            try:
                from .classifier import IntentClassifier

                if IntentClassifier.available():
                    self.classifier = IntentClassifier(factory=factory)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("intent classifier unavailable: %s", exc)

    def match(self, text: str) -> list[TacticHit]:
        t = normalize(text)
        best: dict[str, TacticHit] = {}
        for cue in CUES:
            base = TACTICS[cue.tactic]["weight"] + cue.boost
            matched = [p for p in cue.phrases if phrase_in(p, t)]
            if not matched:
                continue
            phrase = max(matched, key=len)  # most specific phrase is the best evidence
            cur = best.get(cue.tactic)
            if cur is None or base > cur.weight or (base == cur.weight and len(phrase) > len(cur.phrase)):
                best[cue.tactic] = TacticHit(cue.tactic, phrase, base, cue.families or (cur.families if cur else ()))
            elif cue.families and not cur.families:
                cur.families = cue.families
        return list(best.values())

    def analyze(self, text: str, source: str = "text", timestamp: float = 0.0) -> Finding:
        hits = self.match(text)
        norm = normalize(text)
        awareness = any(phrase_in(p, norm) for p in AWARENESS_PHRASES)

        pattern = noisy_or(h.weight for h in hits)
        # A single weak keyword ("police", "urgent") should not alarm anyone.
        if len(hits) == 1 and hits[0].weight < 0.3:
            pattern *= 0.6

        ml, ml_family = 0.0, None
        if self.classifier is not None and norm:
            ml, ml_family = self.classifier.scam_probability(text)

        # One strong keyword in a sentence the classifier reads as benign
        # ("IT is fixing my laptop over AnyDesk") is context, not a verdict.
        if len(hits) == 1 and self.classifier is not None and ml < 0.2:
            pattern *= 0.7
        score = 1.0 - (1.0 - pattern) * (1.0 - 0.6 * ml)
        if awareness:
            score *= AWARENESS_DAMPING

        tagged: dict[str, float] = {}
        for h in hits:
            for fam in h.families:
                tagged[fam] = tagged.get(fam, 0.0) + 0.15
        if ml_family and ml >= 0.5:
            tagged[ml_family] = tagged.get(ml_family, 0.0) + 0.2 * ml
        fams = family_scores({h.tactic: h.weight for h in hits}, tagged)

        return Finding(
            source=source,
            text=text,
            hits=hits,
            pattern_score=pattern,
            ml_score=ml,
            ml_family=ml_family,
            score=min(1.0, score),
            families=fams,
            awareness=awareness,
            timestamp=timestamp,
        )
