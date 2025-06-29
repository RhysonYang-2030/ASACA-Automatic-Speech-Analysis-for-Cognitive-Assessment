# ── speech_tools/pause.py ──────────────────────────────────────────────────────
"""
Pause detection & statistics.

This module replaces the hand-rolled `count_pauses_and_segments`
logic in the original code while keeping the public signature
compatible:

    pauses, total_dur = detect_pauses(audio, sr, patient_intervals)

`patient_intervals` are the (start, end) tuples returned by
`speech_tools.diarize.get_patient_segments`.

Strategy
--------
* **Inter-chunk pauses**
  Any gap ≥ CFG["pause"]["min_outside_sec"] between successive
  patient speech chunks.

* **Intra-chunk hesitations**
  Inside each speech chunk, a short-time energy gate
  (frame RMS in dB < −35) is applied.  Runs of low-energy
  frames longer than

        max(CFG["pause"]["intra_min_sec"],
            CFG["pause"]["intra_dyn_factor"] * median_syll_gap)

  are flagged as *hesitation pauses*.

Returns a flat list of `(start, end, duration)` across the
whole file, plus their cumulative duration.
"""
from __future__ import annotations

import statistics
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from speech_tools.config import CFG

RMS_FRAME_SEC = 0.025  # 25 ms windows
RMS_HOP_SEC = 0.010    # 10 ms hop
ENERGY_GATE_DB = float(CFG["pause"].get("energy_gate_db", -35.0))


# --------------------------------------------------------------------------- #
# Short-time energy helpers
# --------------------------------------------------------------------------- #
def _frame_rms(y: npt.NDArray[np.floating], frame: int, win: int) -> float:
    start = frame * win
    end = start + win
    return float(np.sqrt(np.mean(np.square(y[start:end], dtype=np.float64))))


def _db(x: float, eps: float = 1e-12) -> float:
    return 20.0 * np.log10(max(x, eps))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_pauses(
    audio: npt.NDArray[np.floating],
    sr: int,
    patient_intervals: List[Tuple[float, float]],
    approx_syll_gap: float | None = None,
) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Parameters
    ----------
    audio
        Mono waveform −1…1.
    sr
        Sample rate.
    patient_intervals
        List of patient speech spans in seconds.
    approx_syll_gap
        Optional pre-computed median syllable gap (sec); if None, only the
        static threshold is used for intra-chunk pauses.

    Returns
    -------
    pauses : list of (start, end, dur)
    total_dur : float
    """
    pauses: List[Tuple[float, float, float]] = []

    # --------------------------------------------------------------------- #
    # 1) Inter-chunk pauses  (between speech segments)
    # --------------------------------------------------------------------- #
    min_outside = float(CFG["pause"]["min_outside_sec"])
    for (s1, e1), (s2, e2) in zip(patient_intervals, patient_intervals[1:]):
        gap = s2 - e1
        if gap >= min_outside:
            pauses.append((e1, s2, gap))

    # --------------------------------------------------------------------- #
    # 2) Intra-chunk hesitations  (within each speech segment)
    # --------------------------------------------------------------------- #
    hop = int(round(RMS_HOP_SEC * sr))
    win = int(round(RMS_FRAME_SEC * sr))
    dyn_factor = float(CFG["pause"]["intra_dyn_factor"])
    static_thr = float(CFG["pause"]["intra_min_sec"])
    dyn_thr = (
        dyn_factor * approx_syll_gap if approx_syll_gap else static_thr
    )
    hes_min = max(static_thr, dyn_thr)

    for seg_start, seg_end in patient_intervals:
        start_i = int(round(seg_start * sr))
        end_i = int(round(seg_end * sr))
        seg = audio[start_i:end_i]

        if len(seg) < win:
            continue

        n_frames = 1 + (len(seg) - win) // hop
        low_mask = []
        for f in range(n_frames):
            rms = _frame_rms(seg, f, win)
            low_mask.append(_db(rms) < ENERGY_GATE_DB)

        # Find consecutive low-energy runs
        in_run = False
        run_start = 0.0
        for idx, is_low in enumerate(low_mask):
            t = seg_start + idx * RMS_HOP_SEC
            if is_low and not in_run:
                in_run = True
                run_start = t
            elif not is_low and in_run:
                run_dur = t - run_start
                if run_dur >= hes_min:
                    pauses.append((run_start, t, run_dur))
                in_run = False

        # Close run at end
        if in_run:
            run_dur = seg_end - run_start
            if run_dur >= hes_min:
                pauses.append((run_start, seg_end, run_dur))

    # --------------------------------------------------------------------- #
    # Merge pauses that touch / overlap
    # --------------------------------------------------------------------- #
    pauses.sort(key=lambda p: p[0])
    merged: List[Tuple[float, float]] = []
    for s, e, _ in pauses:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    pause_dicts = [
        {"start": s, "end": e, "duration": e - s} for s, e in merged
    ]
    total = sum(p["duration"] for p in pause_dicts)
    return pause_dicts, total



# --------------------------------------------------------------------------- #
# Convenience legacy alias
# --------------------------------------------------------------------------- #
get_pause_list = detect_pauses
