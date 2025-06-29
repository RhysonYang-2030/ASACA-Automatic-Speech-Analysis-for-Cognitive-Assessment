"""ASACA – Automatic Speech Analysis for Cognitive Assessment."""

def run_inference_and_seg(*a, **k):
    from .inference import run_inference_and_seg as _impl
    return _impl(*a, **k)


def extract_features(*a, **k):
    from .cognition.feature_extractor import extract_features as _impl
    return _impl(*a, **k)

__all__ = ["run_inference_and_seg", "extract_features"]
__version__ = "0.1.0"
