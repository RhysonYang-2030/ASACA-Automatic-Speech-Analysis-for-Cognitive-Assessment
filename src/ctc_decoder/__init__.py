import numpy as np
from functools import lru_cache
from typing import Optional, Dict, Tuple

@lru_cache(maxsize=None)
def _token_list(chars: str) -> Tuple[str, ...]:
    return tuple(chars)


def best_path(mat: np.ndarray, chars: str) -> str:
    """Greedy CTC decoding (best path)."""
    blank_idx = len(chars)
    best_indices = np.argmax(mat, axis=1)
    out = []
    prev = None
    for idx in best_indices:
        if idx != prev and idx != blank_idx:
            out.append(chars[idx])
        prev = idx
    return "".join(out)


def beam_search(mat: np.ndarray, chars: str, beam_width: int = 128, lm_text: Optional[str] = None) -> str:
    """Simple prefix beam search decoder.

    Parameters
    ----------
    mat : ``np.ndarray`` of shape ``(T, C)`` where ``C = len(chars) + 1`` and the
        last column is the blank probability.
    chars : mapping from column index to character, length ``C-1``.
    beam_width : number of beams to keep after each step.
    lm_text : unused, present for API compatibility.
    """
    blank_id = len(chars)
    char_list = list(chars)

    beams: Dict[str, float] = {"": 0.0}
    for t in range(mat.shape[0]):
        next_beams: Dict[str, float] = {}
        for prefix, score in beams.items():
            for i, p in enumerate(mat[t]):
                new_score = score + float(np.log(p + 1e-12))
                if i == blank_id:
                    next_beams[prefix] = np.logaddexp(next_beams.get(prefix, float('-inf')), new_score)
                else:
                    ch = char_list[i]
                    if prefix and prefix[-1] == ch:
                        # either extend with same char or stay
                        stay_prefix = prefix
                        next_beams[stay_prefix] = np.logaddexp(next_beams.get(stay_prefix, float('-inf')), new_score)
                        new_prefix = prefix + ch
                        next_beams[new_prefix] = np.logaddexp(next_beams.get(new_prefix, float('-inf')), new_score)
                    else:
                        new_prefix = prefix + ch
                        next_beams[new_prefix] = np.logaddexp(next_beams.get(new_prefix, float('-inf')), new_score)
        # prune
        beams = dict(sorted(next_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

    if not beams:
        return ""
    best = max(beams.items(), key=lambda x: x[1])[0]
    return best

