"""Temporal, cross-modal risk fusion.

A scam call is not one sentence. The "officer" introduces himself, then
mentions a parcel, then a warrant, then asks you to stay on camera, then asks
you to open your banking app. Each line alone is weak; together they are a
signature. The fusion engine therefore keeps a decaying memory of *tactics*
observed across every modality (call audio, screen text, system state) and
scores the conversation, not the sentence.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .engine import Finding, family_scores, noisy_or
from .patterns import FAMILIES, GENERIC_ADVICE, GENERIC_ADVICE_HI, TACTICS

LEVELS = [
    (0.80, "danger"),
    (0.60, "warning"),
    (0.35, "caution"),
    (0.00, "safe"),
]
LEVEL_ORDER = {"safe": 0, "caution": 1, "warning": 2, "danger": 3}
# Asking for an OTP, screen access or secrecy while a call is live is dangerous
# even when the sentence is short and the classifier is unsure.
STRONG_DURING_CALL = {"credential", "remote_access", "secrecy"}


def level_for(score: float) -> str:
    for thr, name in LEVELS:
        if score >= thr:
            return name
    return "safe"


@dataclass
class Evidence:
    tactic: str
    weight: float
    source: str
    phrase: str
    t: float


@dataclass
class RiskState:
    score: float = 0.0
    level: str = "safe"
    family: str | None = None
    family_name: str | None = None
    tactics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    reasons_hi: list[str] = field(default_factory=list)
    advice: str = ""
    advice_hi: str = ""
    sources: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    t: float = 0.0

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "family": self.family,
            "family_name": self.family_name,
            "tactics": self.tactics,
            "reasons": self.reasons,
            "reasons_hi": self.reasons_hi,
            "advice": self.advice,
            "advice_hi": self.advice_hi,
            "sources": self.sources,
            "context": self.context,
            "t": self.t,
        }


class RiskFusion:
    """Accumulates evidence and produces a conversation-level risk state."""

    def __init__(self, half_life_s: float = 240.0, window_s: float = 900.0, segment_weight: float = 0.35):
        self.half_life_s = half_life_s
        self.window_s = window_s
        self.segment_weight = segment_weight
        self.evidence: list[Evidence] = []
        self.segment_scores: list[tuple[float, float, str]] = []  # (t, score, source)
        self.family_votes: dict[str, float] = {}
        self.context = {"call_active": False, "call_app": None, "remote_access": [], "payment_screen": False}
        self._last_level = "safe"

    # ------------------------------------------------------------------ input
    def add_finding(self, f: Finding) -> None:
        t = f.timestamp or time.time()
        # Evidence from a sentence the intent classifier finds benign counts for less:
        # "RBI announced a new repo rate" mentions an authority but is not a scam line.
        conf = 1.0 if f.ml_family is None and f.ml_score == 0.0 else 0.3 + 0.7 * f.ml_score
        if len(f.hits) >= 2:
            conf = max(conf, 0.75)
        live_call = bool(self.context.get("call_active"))
        for h in f.hits:
            c = max(conf, 0.6) if live_call and h.tactic in STRONG_DURING_CALL else conf
            w = h.weight * (0.35 if f.awareness else 1.0) * c
            self.evidence.append(Evidence(h.tactic, w, f.source, h.phrase, t))
        self.segment_scores.append((t, f.score, f.source))
        fam = f.top_family()
        if fam and f.score > 0.2:
            self.family_votes[fam] = self.family_votes.get(fam, 0.0) + f.families.get(fam, 0.0) * f.score

    def set_context(self, **kw) -> None:
        """System sentinel updates: call_active, call_app, remote_access, payment_screen."""
        t = kw.pop("t", None)
        t = time.time() if t is None else t
        prev_remote = set(self.context.get("remote_access") or [])
        self.context.update(kw)
        now_remote = set(self.context.get("remote_access") or [])
        for app in sorted(now_remote - prev_remote):
            self.evidence.append(Evidence("remote_access", TACTICS["remote_access"]["weight"] + 0.1, "system",
                                          f"{app} is running", t))
        if kw.get("payment_screen") and self.context.get("call_active"):
            self.evidence.append(Evidence("payment", TACTICS["payment"]["weight"], "system",
                                          "payment screen open during a call", t))

    # ------------------------------------------------------------------ output
    def _decay(self, age: float) -> float:
        return math.exp(-math.log(2) * max(0.0, age) / self.half_life_s)

    def evaluate(self, now: float | None = None) -> RiskState:
        now = time.time() if now is None else now
        self.evidence = [e for e in self.evidence if now - e.t <= self.window_s]
        self.segment_scores = [s for s in self.segment_scores if now - s[0] <= self.window_s]

        per_tactic: dict[str, Evidence] = {}
        per_tactic_w: dict[str, float] = {}
        for e in self.evidence:
            w = e.weight * self._decay(now - e.t)
            if w > per_tactic_w.get(e.tactic, 0.0):
                per_tactic_w[e.tactic] = w
                per_tactic[e.tactic] = e

        conv = noisy_or(per_tactic_w.values())
        seg = max((s * self._decay(now - t) for t, s, _ in self.segment_scores), default=0.0)
        score = 1.0 - (1.0 - conv) * (1.0 - self.segment_weight * seg)

        # Cross-modal escalation: a stranger has remote control *and* money is being discussed.
        sources = sorted({e.source for e in per_tactic.values()})
        w = per_tactic_w.get
        if w("remote_access", 0) >= 0.2 and (w("payment", 0) >= 0.15 or w("credential", 0) >= 0.15):
            score = max(score, 0.85)
        if self.context.get("call_active") and len(sources) >= 2 and score >= 0.5:
            score = min(1.0, score + 0.08)
        # Advance-fee signature: a prize / return / refund that you must pay to receive.
        if w("lure", 0) >= 0.08 and w("payment", 0) >= 0.12:
            score = max(score, 0.68)
        # Isolation + money is the digital-arrest signature.
        if w("secrecy", 0) >= 0.12 and w("payment", 0) >= 0.12 and w("authority", 0) >= 0.08:
            score = max(score, 0.9)

        fams = family_scores(per_tactic_w, {k: min(0.3, v * 0.1) for k, v in self.family_votes.items()})
        family = max(fams, key=fams.get) if fams and max(fams.values()) >= 0.35 else None
        level = level_for(score)
        if level == "safe":
            family = None

        reasons, reasons_hi = [], []
        for tac, e in sorted(per_tactic.items(), key=lambda kv: -per_tactic_w[kv[0]]):
            reasons.append(f"{TACTICS[tac]['label']} ('{e.phrase}', {e.source})")
            reasons_hi.append(f"{TACTICS[tac]['label_hi']} ('{e.phrase}')")

        fam_obj = FAMILIES.get(family) if family else None
        state = RiskState(
            score=round(min(1.0, score), 4),
            level=level,
            family=family,
            family_name=fam_obj.name if fam_obj else None,
            tactics=sorted(per_tactic_w),
            reasons=reasons[:6],
            reasons_hi=reasons_hi[:6],
            advice=fam_obj.advice if fam_obj else GENERIC_ADVICE,
            advice_hi=fam_obj.advice_hi if fam_obj else GENERIC_ADVICE_HI,
            sources=sources,
            context=dict(self.context),
            t=now,
        )
        return state

    def escalated(self, state: RiskState) -> bool:
        """True when the level rises (used to fire alerts once, not every second)."""
        up = LEVEL_ORDER[state.level] > LEVEL_ORDER[self._last_level]
        self._last_level = state.level
        return up

    def reset(self) -> None:
        self.evidence.clear()
        self.segment_scores.clear()
        self.family_votes.clear()
        self._last_level = "safe"
