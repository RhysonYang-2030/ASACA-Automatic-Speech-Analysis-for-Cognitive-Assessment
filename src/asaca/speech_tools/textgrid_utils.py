# ── speech_tools/textgrid_utils.py  (final safe version) ───────────────
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

from textgrid import IntervalTier, PointTier, TextGrid

_SENT_RE = re.compile(r"[.!?]")
EPS = 1e-4                     # 0.1 ms 容差


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _merge_intervals(
    spans: List[Tuple[float, float]], max_time: float
) -> List[Tuple[float, float]]:
    if not spans:
        return []
    spans = sorted((max(0.0, s), min(e, max_time)) for s, e in spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s - pe <= EPS:                 # overlap / touch
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s > EPS]


def _safe_add(tier: IntervalTier, start: float, end: float, lab: str):
    """Clamp & add; convert textgrid 的裸 ValueError 为可读信息."""
    start = max(0.0, min(start, tier.maxTime - EPS))
    end   = max(0.0, min(end,   tier.maxTime - EPS))
    if end - start <= EPS:
        return                          # too short → skip
    try:
        tier.add(start, end, lab)
    except ValueError as e:             # e.args[0] == offending float
        raise ValueError(
            f"TextGrid interval ({start:.4f},{end:.4f}) exceeds maxTime "
            f"{tier.maxTime:.4f} in tier '{tier.name}'"
        ) from None


def _complement(spans: List[Tuple[float, float]], total: float
                ) -> List[Tuple[float, float]]:
    gaps, last = [], 0.0
    for s, e in spans:
        if s - last > EPS:
            gaps.append((last, s))
        last = e
    if total - last > EPS:
        gaps.append((last, total))
    return gaps


def _sent_id(full_txt: str, word_idx: int) -> str:
    return f"S{len(_SENT_RE.findall(' '.join(full_txt.split()[:word_idx]))) + 1}"


# ------------------------------------------------------------------ #
# public
# ------------------------------------------------------------------ #
def build_textgrid(
    dp_info: List[Dict],
    patient_segs: List[Tuple[float, float]],
    wav_dur: float,
    tg_path: Path | str,
    patient_id: str = "SPK",
) -> None:
    patient_segs = _merge_intervals(patient_segs, wav_dur)

    tg = TextGrid(minTime=0.0, maxTime=wav_dur)

    # Tier 1 dummy
    tg.append(IntervalTier("meta", 0.0, wav_dur))

    # Tier 2 silence/# vs patient
    sil = IntervalTier("sil", 0.0, wav_dur)
    for s, e in _complement(patient_segs, wav_dur):
        _safe_add(sil, s, e, "#")
    for s, e in patient_segs:
        _safe_add(sil, s, e, patient_id)
    tg.append(sil)

    # Tier 3 sentence numbers
    sent = IntervalTier("sentence", 0.0, wav_dur)
    full_txt = " ".join(w["word"] for w in dp_info)
    for idx, w in enumerate(dp_info):
        _safe_add(sent, w["start_sec"], w["end_sec"], _sent_id(full_txt, idx))
    tg.append(sent)

    # Tier 4 word
    word = IntervalTier("word", 0.0, wav_dur)
    for w in dp_info:
        _safe_add(word, w["start_sec"], w["end_sec"], w["word"])
    tg.append(word)

    # Tier 5 dysfluency
    dys = IntervalTier("dysfluency", 0.0, wav_dur)
    for w in dp_info:
        if w.get("disfluency_flag"):
            _safe_add(dys, w["start_sec"], w["end_sec"], "F")
    tg.append(dys)

    # Tier 6 syllable nuclei (PointTier, kept empty)
    tg.append(PointTier("syllable", 0.0, wav_dur))

    Path(tg_path).parent.mkdir(parents=True, exist_ok=True)
    tg.write(str(tg_path))
