import numpy as np
import pytest

from sajag.risk.engine import TextRiskAnalyzer, noisy_or
from sajag.risk.fusion import RiskFusion, level_for
from sajag.risk.normalize import normalize, phrase_in
from sajag.signals.system import is_payment_screen, snapshot


@pytest.fixture(scope="module")
def an():
    return TextRiskAnalyzer()


def test_normalize_and_boundaries():
    assert normalize("  O.T.P  bata do!! ") == "otp bata do"
    assert phrase_in("ed", "ed officer called") and not phrase_in("ed", "i need help")
    assert phrase_in("गिरफ्तार", normalize("आपको गिरफ्तार किया जाएगा"))


def test_noisy_or_bounds():
    assert noisy_or([]) == 0.0
    assert 0.0 < noisy_or([0.3, 0.3]) < 0.6


@pytest.mark.parametrize("text", [
    "Main CBI se bol raha hoon, aapke naam se parcel mein drugs mile hain",
    "Your computer is infected, call Microsoft support toll free and install AnyDesk",
    "आपका केवाईसी खत्म हो गया है, ओटीपी बताइए",
    "enter your UPI PIN to receive the refund",
])
def test_scam_sentences_score_high(an, text):
    assert an.analyze(text).score >= 0.35


@pytest.mark.parametrize("text", [
    "Beta khana kha liya? Diwali pe kab aa rahe ho",
    "Please join the team call on Teams at 3 pm",
    "Do not share your OTP with anyone. SBI will never ask for your PIN.",
    "पुलिस कभी वीडियो कॉल पर गिरफ्तार नहीं करती, सावधान रहें।",
])
def test_benign_and_awareness_sentences_score_low(an, text):
    assert an.analyze(text).score < 0.35


def test_fusion_accumulates_across_turns(an):
    fu = RiskFusion()
    t = 1000.0
    for i, line in enumerate([
        "I am calling from Mumbai police cyber crime branch",
        "a parcel in your name contains drugs",
        "do not tell anyone and do not disconnect the call",
        "transfer the money to the RBI safe account for verification",
    ]):
        fu.add_finding(an.analyze(line, source="call", timestamp=t + 10 * i))
    st = fu.evaluate(t + 40)
    assert st.level == "danger"
    assert st.family == "digital_arrest"
    assert {"authority", "secrecy", "payment"} <= set(st.tactics)


def test_remote_access_plus_credentials_is_danger(an):
    fu = RiskFusion()
    fu.set_context(call_active=True, remote_access=["AnyDesk"], t=1000)
    fu.add_finding(an.analyze("tell me the OTP you received", source="call", timestamp=1001))
    assert fu.evaluate(1002).level == "danger"


def test_evidence_decays():
    fu = RiskFusion(half_life_s=60)
    fu.set_context(remote_access=["AnyDesk"], t=0)
    s0 = fu.evaluate(1).score
    s1 = fu.evaluate(600).score
    assert s1 < s0


def test_levels():
    assert level_for(0.1) == "safe" and level_for(0.4) == "caution"
    assert level_for(0.7) == "warning" and level_for(0.95) == "danger"


def test_system_snapshot_matching():
    snap = snapshot(["WhatsApp.exe", "AnyDesk.exe", "chrome.exe"])
    assert snap.call_active and snap.remote_access == ["AnyDesk"]
    assert is_payment_screen("Fund Transfer - Add Payee - Enter UPI PIN")
    assert not is_payment_screen("Weekly team meeting notes")


def test_classifier_onnx_shape():
    from sajag.risk.classifier import IntentClassifier, featurize

    x = featurize("verification ke liye paise transfer karo")
    assert x.shape == (1 << 14,) and abs(float(np.linalg.norm(x)) - 1.0) < 1e-5
    clf = IntentClassifier()
    p = clf.predict_proba("aapka sim band ho jayega 2 ghante mein")
    assert abs(sum(p.values()) - 1.0) < 1e-4
