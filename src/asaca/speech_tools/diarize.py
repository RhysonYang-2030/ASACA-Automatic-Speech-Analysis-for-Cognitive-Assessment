# ── speech_tools/diarize.py  (v3 – robust silence skipper) ──────────────────
from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple


import numpy as np
import sys

from pyannote.core import Annotation

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    import torch
    from pyannote.audio import Pipeline


if sys.platform == "win32":
    _ensure = None
    try:
        from asaca._torch_dll_utils import ensure_torch_dlls as _ensure
    except Exception:  # pragma: no cover - optional path
        try:
            from torch_dll_utils import ensure_torch_dlls as _ensure
        except Exception:  # pragma: no cover - optional path
            _ensure = None
else:
    _ensure = None

try:  # Prefer the installed package layout.
    from asaca.speech_tools.config import CFG  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - editable installs / legacy tree
    try:
        from .config import CFG  # type: ignore
    except ImportError:  # pragma: no cover - when run from legacy speech_tools pkg
        from speech_tools.config import CFG  # type: ignore

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_leading_trailing_silence(
    y: np.ndarray,
    sr: int,
    gate_db: float = -40.0,
    min_sil_dur: float = 1.0,
) -> Tuple[float, float]:
    """Return (lead_sec, tail_sec) to trim, based on RMS energy."""
    hop = int(sr * 0.01)            # 10-ms hop
    win = int(sr / float(CFG["vad"].get("pitch_floor", 100)))   # ≈ 10 ms @100 Hz
    rms  = np.array(
        [np.sqrt(np.mean(y[i : i + win] ** 2)) for i in range(0, len(y) - win, hop)]
    )
    db   = 20 * np.log10(np.maximum(rms, 1e-12))

    # frames below gate ⇒ silence
    silent = db < gate_db
    def _contig_run(idx_iter):
        groups = []
        run = []
        for idx in idx_iter:
            if not run or idx == run[-1] + 1:
                run.append(idx)
            else:
                groups.append(run)
                run = [idx]
        if run:
            groups.append(run)
        return groups

    frames = np.arange(len(silent))
    lead = tail = 0.0
    # leading
    if silent[0]:
        first_voiced = frames[~silent][0] if (~silent).any() else len(frames)
        dur = first_voiced * hop / sr
        if dur >= min_sil_dur:
            lead = dur
    # trailing
    if silent[-1]:
        last_voiced = frames[~silent][-1] if (~silent).any() else 0
        dur = (len(frames) - 1 - last_voiced) * hop / sr
        if dur >= min_sil_dur:
            tail = dur
    return lead, tail


@lru_cache(maxsize=1)
def _pipe() -> "Pipeline":
    if sys.platform == "win32" and _ensure is not None:
        _ensure()

    from pyannote.audio import Pipeline
    import torch

    m_id = CFG["paths"]["diar_model"]
    token = CFG["paths"].get("hf_token")
    gpu = CFG["paths"].get("diar_use_gpu") and torch.cuda.is_available()
    os.environ["PYANNOTE_AUDIO_DEVICE"] = "cuda" if gpu else "cpu"
    pipe = Pipeline.from_pretrained(m_id, use_auth_token=token)
    pipe.to(torch.device("cuda" if gpu else "cpu"))
    return pipe


def _identify_roles(ann: Annotation) -> Tuple[str | None, List[str]]:
    """
    examiner = shortest speaker that starts earliest AND whose duration
               ≤ 60 % of the longest speaker.
    patient  = all other speakers (usually just one).
    If only one speaker exists, examiner=None and that speaker is patient.
    """
    labels = ann.labels()
    if len(labels) == 1:
        return None, list(labels)

    durations = {spk: ann.label_timeline(spk).duration() for spk in labels}
    longest  = max(durations, key=durations.get)
    earliest = min(labels, key=lambda s: ann.label_timeline(s)[0].start)

    # examiner candidate must be earliest *and* clearly shorter
    if earliest != longest and durations[earliest] <= 0.6 * durations[longest]:
        return earliest, [spk for spk in labels if spk != earliest]
    else:
        return None, [longest]          # fall back: keep only longest speaker


def _to_intervals(
    ann: Annotation, speakers: List[str]
) -> List[Tuple[float, float]]:
    """
    Concatenate support timelines for the requested speakers and
    return flat (start, end) tuples sorted in time.
    """
    intervals: list[Tuple[float, float]] = []
    for spk in speakers:
        for seg in ann.label_timeline(spk).support():
            intervals.append((seg.start, seg.end))
    intervals.sort()
    return intervals



def _merge(intv: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    if not intv: return []
    merged = [list(intv[0])]
    for s, e in intv[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #
def get_patient_segments(audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    # 1) Detect long leading / trailing silence - but keep offsets!
    if sys.platform == "win32" and _ensure is not None:
        _ensure()

    import torch

    lead, tail = _find_leading_trailing_silence(
        audio,
        sr,
        gate_db=CFG["vad"].get("energy_gate_db", -40.0),
        min_sil_dur=CFG["vad"].get("silence_skip_sec", 1.0),
    )
    cut_start = int(lead * sr)
    cut_end   = len(audio) - int(tail * sr)
    y_trim    = audio[cut_start:cut_end]

    # 2) Diarise trimmed section
    wave = torch.from_numpy(y_trim).unsqueeze(0)  # shape (1, time)
    ann = _pipe()({"waveform": wave, "sample_rate": sr})

    # 3) Role assignment
    examiner, patients = _identify_roles(ann)

    # 4) Collect patient segments & restore timeline offset
    pats = _to_intervals(ann, patients)
    pats = [(s + lead, e + lead) for s, e in pats]

    # 5) Gap merge
    gap = float(CFG["vad"]["gap_merge_sec"])
    return _merge(pats, gap)


get_patient_speech_segments = get_patient_segments
