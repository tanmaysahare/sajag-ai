import numpy as np
import pytest

from sajag.demo import SCENARIOS, render_screen, run_scenario
from sajag.guardian import Guardian


@pytest.fixture(scope="module")
def guardian():
    return Guardian()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_reach_expected_level(guardian, name):
    trace = run_scenario(name, guardian, use_ocr=True)
    assert trace[-1]["level"] == SCENARIOS[name]["expected"]


def test_digital_arrest_warns_before_money_moves(guardian):
    trace = run_scenario("digital_arrest", guardian, use_ocr=True)
    first_warning = next(r for r in trace if r["level"] in ("warning", "danger"))
    payment_step = next(r for r in trace if "transfer" in r["what"].lower())
    assert first_warning["at"] < payment_step["at"]


def test_ocr_reads_scareware():
    from sajag.vision.ocr import ScreenOCR

    text = ScreenOCR().read_text(render_screen("scareware")).lower()
    assert "blocked" in text and "support" in text


def test_log_mel_shape_and_range():
    from sajag.audio.features import N_FRAMES, log_mel_spectrogram, mel_filters

    sr = 16000
    t = np.arange(sr * 2) / sr
    mel = log_mel_spectrogram(0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32))
    assert mel.shape == (80, N_FRAMES)
    assert mel.max() <= 2.5 and mel.min() >= -1.5
    fb = mel_filters(80)
    assert fb.shape == (80, 201) and (fb >= 0).all()


def test_vad_energy_fallback_finds_speech():
    from sajag.audio.vad import SileroVAD

    sr = 16000
    silence = np.zeros(sr, np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    segs = list(SileroVAD().segments(np.concatenate([silence, tone, silence])))
    assert len(segs) == 1 and 0.8 < segs[0].start_s < 1.2


def test_static_shape_adapter_pads_and_crops():
    from sajag.vision.ocr import StaticShapeAdapter

    class FakeIn:
        name, shape = "x", [1, 3, 64, 64]

    class Npu:
        def get_inputs(self):
            return [FakeIn()]

        def run(self, names, feeds):
            x = feeds["x"]
            assert x.shape == (1, 3, 64, 64)
            return [x[:, :1]]

    class Cpu:
        def run(self, names, feeds):
            return ["cpu"]

    ad = StaticShapeAdapter(Npu(), Cpu())
    y = ad.run(None, {"x": np.ones((2, 3, 32, 40), np.float32)})[0]
    assert y.shape == (2, 1, 32, 40) and ad.npu_calls == 1
    assert ad.run(None, {"x": np.ones((1, 3, 128, 40), np.float32)}) == ["cpu"]


def test_whisper_decode_loop_with_fake_sessions(tmp_path):
    """Checks the AI Hub IO contract (names, shapes, KV-cache threading, prompt forcing)."""
    from sajag.audio import asr as asr_mod

    L, H, D, V = 2, 2, 4, 51865

    class Spec:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class Enc:
        def get_inputs(self):
            return [Spec("input_features", [1, 80, 3000])]

        def get_outputs(self):
            return [Spec(f"{p}_cache_cross_{i}", None) for i in range(L) for p in ("k", "v")]

        def run(self, feeds):
            assert feeds["input_features"].shape == (1, 80, 3000)
            return [np.zeros((H, 1, D, 1500), np.float32) for _ in range(2 * L)]

    class Dec:
        steps = 0

        def get_inputs(self):
            specs = [Spec("input_ids", [1, 1]), Spec("attention_mask", [1, 1, 1, 200])]
            specs += [Spec(f"k_cache_self_{i}_in", [H, 1, D, 199]) for i in range(L)]
            specs += [Spec(f"v_cache_self_{i}_in", [H, 1, 199, D]) for i in range(L)]
            specs += [Spec(f"{p}_cache_cross_{i}", None) for i in range(L) for p in ("k", "v")]
            return specs + [Spec("position_ids", [1])]

        def get_outputs(self):
            return [Spec("logits", None)] + [Spec(f"{p}_cache_self_{i}_out", None) for i in range(L) for p in ("k", "v")]

        def run(self, feeds):
            Dec.steps += 1
            assert feeds["position_ids"].dtype == np.int32
            logits = np.zeros((1, V, 1, 1), np.float32)
            logits[0, 50257 if Dec.steps > 5 else 1000] = 1.0  # emit 2 tokens then EOT
            outs = [logits]
            for i in range(L):
                outs += [feeds[f"k_cache_self_{i}_in"], feeds[f"v_cache_self_{i}_in"]]
            return outs

    class Tok:
        eot = 50257

        def prompt(self, lang):
            return [50258, 50276, 50360, 50364]

        def decode(self, ids):
            return " ".join(str(i) for i in ids if i < self.eot)

    obj = asr_mod.AIHubWhisperASR.__new__(asr_mod.AIHubWhisperASR)
    obj.encoder, obj.decoder, obj.tokenizer, obj.language = Enc(), Dec(), Tok(), "hi"
    obj.enc_input, obj.n_mels = "input_features", 80
    obj.dec_inputs = [s.name for s in obj.decoder.get_inputs()]
    obj.dec_shapes = {s.name: s.shape for s in obj.decoder.get_inputs()}
    obj.cross_names = [n for n in obj.dec_inputs if "cache_cross" in n]
    obj.self_k = [f"k_cache_self_{i}_in" for i in range(L)]
    obj.self_v = [f"v_cache_self_{i}_in" for i in range(L)]
    obj.enc_outputs = [s.name for s in obj.encoder.get_outputs()]
    obj.dec_outputs = [s.name for s in obj.decoder.get_outputs()]
    tr = obj.transcribe(np.zeros(16000, np.float32))
    assert tr.text == "1000 1000" and tr.tokens == 3


def test_entity_extraction():
    from sajag.report import extract_entities

    e = extract_entities("Pay to winsupport.desk@ybl or a/c 50100234567891 IFSC HDFC0001234, call 98765 43210, "
                         "Rs 4,999 now, mail me at a.b@gmail.com, visit refund-help.xyz")
    assert e.upi_ids == {"winsupport.desk@ybl"}
    assert "50100234567891" in e.accounts and "HDFC0001234" in e.ifsc
    assert "+91 9876543210" in e.phones
    assert any("4,999" in a for a in e.amounts)
    assert "refund-help.xyz" in e.links


def test_golden_hour_report(guardian):
    run_scenario("tech_support", guardian, use_ocr=False)
    r = guardian.report()
    assert r["risk_level"] == "danger"
    assert "AnyDesk" in r["entities"]["remote_apps"]
    assert "winsupport.desk@ybl" in r["entities"]["upi_ids"]
    assert "1930" in r["text"] and len(r["timeline"]) >= 4
    ts = [row["t"] for row in r["timeline"]]
    assert ts == sorted(ts)
