from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder


@lru_cache(maxsize=None)
def _get_decoder(chars: str, lm_text: Optional[str]) -> BeamSearchDecoderCTC:
    """Return a cached decoder with a unique alphabet for ``chars``.

    ``pyctcdecode`` does not allow duplicate labels. Some tokenizers map
    multiple IDs to the same character (e.g. special tokens like ``<pad>``
    and ``</s>``).  We assign temporary surrogate characters from the
    Unicode Private Use Area for duplicates so the decoder can be built
    reliably and keep a reverse mapping for later.
    """

    labels: list[str] = []
    mapping: Dict[str, str] = {}
    used = set()
    next_private = 0xE000
    for ch in chars:
        if ch in used:
            surrogate = chr(next_private)
            next_private += 1
        else:
            surrogate = ch
            used.add(ch)
        labels.append(surrogate)
        mapping[surrogate] = ch

    unigrams = list(lm_text) if lm_text else None
    decoder = build_ctcdecoder(labels, kenlm_model_path=None, unigrams=unigrams)
    decoder._char_mapping = mapping  # type: ignore[attr-defined]
    return decoder


def beam_search(mat: np.ndarray, chars: str, beam_width: int = 25, lm_text: Optional[str] = None) -> str:
    """Beam search decoder using :mod:`pyctcdecode` with de-duplicated labels."""
    decoder = _get_decoder(chars, lm_text)
    res = decoder.decode(mat, beam_width=beam_width)
    mapping: Dict[str, str] = getattr(decoder, "_char_mapping", {})  # type: ignore[attr-defined]
    if mapping:
        res = "".join(mapping.get(ch, ch) for ch in res)
    return res


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

