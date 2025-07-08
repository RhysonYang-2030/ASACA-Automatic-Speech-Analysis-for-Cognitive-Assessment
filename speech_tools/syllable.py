# ── speech_tools/syllable.py  (conf-based blend) ────────────────────────────
"""
Hybrid syllable counter without ground-truth WER.

Blend rule
----------
confidence = mean( max_c p(c | frame) )   ∈ (0, 1]

    α = clip((confidence − low) / (high − low), 0, 1)

syllables = α·acoustic + (1−α)·text

`low`, `high` live in config.yaml (defaults 0.80 / 0.90).
"""

from __future__ import annotations
import re, statistics
from typing import Dict, Tuple, List, Optional,Any
import librosa, numpy as np, parselmouth
from numpy.typing import NDArray
import nltk
try:
    nltk.data.find("corpora/cmudict")
except LookupError:  # pragma: no cover - network during install
    nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict
from speech_tools.config import CFG

import numpy as np
from numpy.typing import NDArray
import parselmouth

try:
    import syllapy
except ImportError:
    syllapy = None

_CMUDICT = cmudict.dict()
_VOWEL_RE = re.compile(r"[aeiouy]+")


# ── text syllables ─────────────────────────────────────────────────────────
def _cmu_syllables(word: str) -> int:
    ent = _CMUDICT.get(word.lower())
    if not ent: return 0
    return min(sum(ch.isdigit() for ch in ph) for ph in ent)

def _text_syllable_count(text: str) -> int:
    if not text.strip(): return 0
    words = re.findall(r"[A-Za-z']+", text)
    total = 0
    for w in words:
        syl = syllapy.count(w) if syllapy else 0
        if syl == 0: syl = _cmu_syllables(w)
        if syl == 0: syl = max(1, len(_VOWEL_RE.findall(w)))
        total += syl
    return total


# ── acoustic syllables ────────────────────────────────────────────────────
def _onset_syllable_count(y: NDArray[np.floating], sr: int) -> int:
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                       hop_length=hop, units="time", backtrack=False)
    return int(len(peaks))


# ── public API ────────────────────────────────────────────────────────────
# ── speech_tools/syllable.py  (robust Praat nuclei) ─────────────────────────
pcall = parselmouth.praat.call

_REL_DB = float(CFG["syllable"].get("praat_rel_thresh_db", 25.0))
_MIN_GAP = float(CFG["syllable"].get("praat_min_gap_sec", 0.15))  # 75 ms


def _praat_nuclei_count(y: NDArray[np.floating], sr: int) -> int:
    """
    返回音节核（syllable nuclei）数量。
    若全部检测策略失效，则返回 -1 供调用方回退到 Librosa-onset 法。

    兼容 praat-parselmouth 0.4.5；不会调用新版专属方法。
    """
    y.size =0
    # ---------- 0 · 基本检查 ----------
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        print("[Praat] ❌ silent input → -1")
        return -1

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    intensity = snd.to_intensity()
    med_db = float(np.median(intensity.values[0]))
    print(f"[Praat] median intensity = {med_db:.2f} dB")

    # ---------- 1 · PointProcess生成（cc / peaks 双轨） ----------
    def _make_pp(mode: str, f0_min: float, f0_max: float):
        """mode: 'cc' | 'peaks' → PointProcess 或 None"""
        cmd = "To PointProcess (periodic, cc)" if mode == "cc" \
              else "To PointProcess (periodic, peaks)"
        try:
            pp = pcall(snd, cmd, f0_min, f0_max)
            if isinstance(pp, list):                       # 极少返回 list
                pp = pp[0] if pp else None
            return pp
        except Exception as e:
            print(f"[Praat]   {mode}: Praat error → {e}")
            return None

    # ---------- 2 · 清洗 & 计数 ----------
    def _count_valid_nuclei(pp, tag: str) -> int:
        if pp is None:
            print(f"[Praat]   {tag}: PointProcess None")
            return -1

        n_pts = int(pcall(pp, "Get number of points"))
        if n_pts == 0:
            print(f"[Praat]   {tag}: 0 points")
            return -1

        thresh = med_db - _REL_DB
        pulses: List[float] = []
        for i in range(1, n_pts + 1):
            t = float(pcall(pp, "Get time from index", i))
            if intensity.get_value(t) >= thresh:
                if not pulses or t - pulses[-1] >= _MIN_GAP:
                    pulses.append(t)

        print(f"[Praat]   {tag}: pulses={len(pulses)} "
              f"(th={thresh:.1f} dB, min_gap={_MIN_GAP*1e3:.0f} ms)")
        return len(pulses) if pulses else -1

    # ---------- 3 · 三级策略 ----------
    attempts = [
        ("cc-narrow", "cc",    75.0, 600.0),
        ("cc-wide",   "cc",    50.0, 700.0),
        ("peaks",     "peaks", 50.0, 700.0),
    ]

    for tag, mode, fmin, fmax in attempts:
        print(f"[Praat] ▶ pass {tag}: f0=[{fmin},{fmax}]")
        pp = _make_pp(mode, fmin, fmax)
        n  = _count_valid_nuclei(pp, tag)
        if n > 0:
            print(f"[Praat] ✅ success in {tag}: nuclei={n}")
            return n
        print(f"[Praat]   {tag} failed")

    # ---------- 4 · 全部失败 ----------
    print("[Praat] ❌ nuclei detection failed → -1")
    return -1
# -------------------------------------------------------------------------
def hybrid_syllable_count(
    audio: NDArray[np.floating],
    sr: int,
    pred_text: str = "",
    ctc_conf: float | None = None,
) -> Tuple[int, Dict]:
    dbg: Dict = {}

    # --- hybrid_syllable_count ---------------------------------------
    nuclei = 0 # _praat_nuclei_count(audio, sr)

    dbg["nuclei"] = 0 #max(nuclei, 0)

    onset = _onset_syllable_count(audio, sr)
    dbg["onsets"] = onset

    ratio = float(CFG["syllable"]["nuclei_onset_ratio"])
    if nuclei == -1 or nuclei < ratio * onset:
        acoustic = onset
        dbg["acoustic_method"] = "onset"
    else:
        acoustic = nuclei
        dbg["acoustic_method"] = "praat"

    # blend with text using confidence
    if pred_text and ctc_conf is not None and np.isfinite(ctc_conf):
        lo = float(CFG["syllable"].get("ctc_conf_low", 0.80))
        hi = float(CFG["syllable"].get("ctc_conf_high", 0.90))
        alpha = max(0.0, min(1.0, (ctc_conf - lo) / (hi - lo + 1e-9)))
        if alpha > 0:
            t_cnt = _text_syllable_count(pred_text)
            fused = int(round(alpha * acoustic + (1 - alpha) * t_cnt))
            dbg.update(text=t_cnt, fused=fused, alpha=alpha, conf=ctc_conf)
            return max(1, fused), dbg

    return max(1, acoustic), dbg
