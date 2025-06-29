import numpy as np
from asaca.speech_tools import pause


def test_detect_pauses(monkeypatch):
    monkeypatch.setitem(pause.CFG["pause"], "min_outside_sec", 0.5)
    monkeypatch.setattr(pause, "ENERGY_GATE_DB", 100)
    audio = np.ones(4000)
    segs = [(0, 1), (2, 3)]
    pauses, total = pause.detect_pauses(audio, 1000, segs)
    assert total > 0
