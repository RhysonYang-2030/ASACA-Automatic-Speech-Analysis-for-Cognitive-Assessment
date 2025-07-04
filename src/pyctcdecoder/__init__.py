from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder


@lru_cache(maxsize=None)
def _get_decoder(chars: str, lm_text: Optional[str]) -> BeamSearchDecoderCTC:
    labels = list(chars)
    unigrams = list(lm_text) if lm_text else None
    return build_ctcdecoder(labels, kenlm_model_path=None, unigrams=unigrams)


def beam_search(mat: np.ndarray, chars: str, beam_width: int = 25, lm_text: Optional[str] = None) -> str:
    """Beam search decoder using :mod:`pyctcdecode`."""
    decoder = _get_decoder(chars, lm_text)
    return decoder.decode(mat, beam_width=beam_width)


def best_path(mat: np.ndarray, chars: str) -> str:
    """Greedy decoder equivalent to best path."""
    blank_idx = len(chars)
    best_indices = np.argmax(mat, axis=1)
    out = []
    prev = None
    for idx in best_indices:
        if idx != prev and idx != blank_idx:
            out.append(chars[idx])
        prev = idx
    return "".join(out)

