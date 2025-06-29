import pytest

try:
    from asaca import gui
except Exception:
    gui = None


@pytest.mark.skipif(gui is None, reason="PyQt5 not available")
def test_gui_launch(qtbot, monkeypatch):
    monkeypatch.setattr(gui.MainWindow, "loadInferenceModel", lambda self: None)
    w = gui.MainWindow({})
    qtbot.addWidget(w)
