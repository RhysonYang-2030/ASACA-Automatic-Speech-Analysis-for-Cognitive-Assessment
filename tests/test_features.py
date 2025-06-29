import types, sys

dummy_inf = types.ModuleType("asaca.inference")
def _dummy_run(*a, **k):
    return "hello", {}, []
dummy_inf.run_inference_and_seg = _dummy_run
sys.modules.setdefault("asaca.inference", dummy_inf)
fake_nltk = types.ModuleType("nltk")
class DummyStop:
    def words(self, lang):
        return []
fake_nltk.corpus = types.SimpleNamespace(stopwords=DummyStop())
class DummyStem:
    def __init__(self, *a, **k):
        pass
    def stem(self, w):
        return w
fake_nltk.stem = types.SimpleNamespace(snowball=types.SimpleNamespace(SnowballStemmer=DummyStem))
fake_nltk.data = types.SimpleNamespace(find=lambda x: None)
fake_nltk.download = lambda *a, **k: None
sys.modules.setdefault("nltk", fake_nltk)
sys.modules.setdefault("nltk.corpus", fake_nltk.corpus)
sys.modules.setdefault("nltk.stem.snowball", fake_nltk.stem.snowball)

from asaca.cognition import feature_extractor as feat


def test_extract_features(monkeypatch, tmp_path):
    def dummy_run(*a, **k):
        return "hello world", {k: 1.0 for k in feat.SCALAR_KEYS}, []

    class DummyBank:
        def count_hits(self, tokens):
            return 1, 0, 0

    monkeypatch.setattr(feat, "run_inference_and_seg", dummy_run)
    vec, meta = feat.extract_features(
        tmp_path / "a.wav", None, None, DummyBank(), device="cpu"
    )
    assert len(vec) == len(feat.SCALAR_KEYS) + len(feat.DICT_FEATS)
    assert meta["dict_counts"] == (1, 0, 0)
