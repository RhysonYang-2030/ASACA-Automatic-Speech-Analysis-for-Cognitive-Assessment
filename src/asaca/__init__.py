"""ASACA – Automatic Speech Analysis for Cognitive Assessment."""
import sys

if sys.platform == "win32":
    try:
        from ._torch_dll_utils import ensure_torch_dlls as _ensure_torch_dlls
    except Exception:  # pragma: no cover - bootstrap is best-effort
        _ensure_torch_dlls = None
    else:
        _ensure_torch_dlls()

def run_inference_and_seg(*a, **k):
    from .inference import run_inference_and_seg as _impl
    return _impl(*a, **k)


def extract_features(*a, **k):
    from .cognition.feature_extractor import extract_features as _impl
    return _impl(*a, **k)

__all__ = ["run_inference_and_seg", "extract_features"]
__version__ = "0.1.3"
