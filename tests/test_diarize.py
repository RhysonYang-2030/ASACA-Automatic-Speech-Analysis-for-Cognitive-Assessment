import numpy as np
from pyannote.core import Annotation, Segment
import types
from asaca.speech_tools import diarize


def test_get_patient_segments(monkeypatch):
    def dummy_pipe():
        class Dummy:
            def __call__(self, _):
                ann = Annotation()
                ann[Segment(0.2, 0.8)] = "S1"
                return ann
        return Dummy()

    monkeypatch.setattr(diarize, "_pipe", dummy_pipe)
    monkeypatch.setattr(diarize, "_find_leading_trailing_silence", lambda *a, **k: (0.0, 0.0))
    y = np.zeros(16000)
    segs = diarize.get_patient_segments(y, 16000)
    assert segs == [(0.2, 0.8)]
