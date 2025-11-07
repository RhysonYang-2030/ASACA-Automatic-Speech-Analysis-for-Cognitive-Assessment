from __future__ import annotations

import json
import math
import statistics
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
import sys
import os
from torch_dll_utils import ensure_torch_dlls

ensure_torch_dlls()

import torch
from jiwer import wer as jiwer_wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import json
import time
import psutil
import numpy as np
try:
    import syllapy
except Exception:  # pragma: no cover - optional
    syllapy = None
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
import webrtcvad
from jiwer import wer
from scipy.signal import medfilt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import ctc_segmentation as cs
from ctc_decoder import beam_search, best_path
import pandas as pd
import inspect
from types import SimpleNamespace
# New helper modules ---------------------------------------------------------
from .speech_tools.config import CFG
from .speech_tools.diarize import get_patient_segments
from .speech_tools.pause import detect_pauses
from .speech_tools.syllable import hybrid_syllable_count

# --------------------------------------------------------------------------- #
# 0 · Utility – unchanged from original
# --------------------------------------------------------------------------- #
DEBUG_FILE = Path(__file__).with_name("debug_output_inference.txt")
import numpy as np
np.seterr(all="ignore")          # silence benign empty-slice RuntimeWarnings


def debug_print(msg: str) -> None:
    print(msg)
    with DEBUG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(f"{msg}\n")


# --------------------------------------------------------------------------- #
# 1 · Audio I/O (unchanged public API, but now returns float32)
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, target_sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """
    Read an arbitrary-encoded audio file and return float32 mono [-1, 1].

    *The function name and signature are preserved.*
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = y.astype(np.float32, copy=False)
    return y, sr


# --------------------------------------------------------------------------- #
# 2 · Patient-speech segmentation  (replaces old vad_segmentation + examiner trim)
# --------------------------------------------------------------------------- #
def vad_segmentation(
    audio: np.ndarray,
    sr: int,
    hf_token: str | None = None,
    frame_duration_ms: int = 20,
    aggressiveness: int = 3,
):
    """
    **DEPRECATED** parameters kept for compatibility.

    Returns a list of (start, end) tuples of patient speech
    by calling the new diariser.  Downstream functions that expect the
    old format will keep working.
    """
    patient_segs = get_patient_segments(audio, sr)
    # Convert to the list of dicts the legacy code used to return
    return [{"start": s, "end": e, "duration": e - s} for s, e in patient_segs]


# --------------------------------------------------------------------------- #
# 3 · Combine segments into a contiguous waveform  (unchanged)
# --------------------------------------------------------------------------- #
def combine_audio_segments(
    audio: np.ndarray, sr: int, segments: List[Dict[str, float]]
) -> np.ndarray:
    gap_sec = 0.10
    gap = np.zeros(int(round(gap_sec * sr)), dtype=audio.dtype)

    parts = []
    for seg in segments:
        start = int(round(seg["start"] * sr))
        end = int(round(seg["end"] * sr))
        parts.extend([audio[start:end], gap])

    return np.concatenate(parts) if parts else np.empty(0, dtype=audio.dtype)


# --------------------------------------------------------------------------- #
# 4 · Pause logic (wrapper around speech_tools.pause.detect_pauses)
# --------------------------------------------------------------------------- #
def count_pauses_and_segments(
    patient_segs: List[Dict[str, float]],
    exam_segs: List[Dict[str, float]] | None = None,
    min_pause: float = 0.25,
    merge_within: float = 0.15,
):
    """
    Legacy signature preserved but internally defers to the new pause helper.
    """
    # Flatten segments to tuples
    intervals = [(d["start"], d["end"]) for d in patient_segs]
    if not intervals:
        return [], 0.0

    # audio buffer is needed for intra-chunk hesitations
    # Caller higher up always has the full waveform in scope; we pass it via closure
    global _GLOBAL_WAVEFORM_CACHE  # set in extract_features_from_audio
    pauses, total = detect_pauses(
        _GLOBAL_WAVEFORM_CACHE["audio"],
        _GLOBAL_WAVEFORM_CACHE["sr"],
        intervals,
        approx_syll_gap=_GLOBAL_WAVEFORM_CACHE.get("median_syll_gap"),
    )
    return pauses, total


# --------------------------------------------------------------------------- #
# 5 · Syllable counting (wrapper)
# --------------------------------------------------------------------------- #
def count_syllables_praat(audio: np.ndarray, sr: int, *_, **__):
    # Legacy stub – kept because other modules expect it
    return hybrid_syllable_count(audio, sr)[0]


def count_syllables_from_text(pred_text: str):
    # Preserve original call path
    return hybrid_syllable_count(np.empty(0), 1, pred_text=pred_text)[0]

# --------------------------------------------------------------------------- #
# 6 · Feature extraction – patched core
# --------------------------------------------------------------------------- #
_GLOBAL_WAVEFORM_CACHE: Dict[str, any] = {}  # simple per-call singleton


def extract_features_from_audio(
    audio_path: str | Path,
    dp_info: List[Dict] | None,
    pred_text: str = "",
    wer_estimate: float = 1.0,
    sr: int = 16_000,
    ctc_conf=None,
) -> Tuple[Dict, List]:
    """
    Returns (global_feature_dict, fused_pause_list)

    The function body is rewritten to use the new helpers but
    *external behaviour* is unchanged.
    """
    # --------------------------------------------------------------------- #
    # Load waveform once and memoise for downstream helpers
    # --------------------------------------------------------------------- #
    audio, sr = load_audio(audio_path, target_sr=sr)
    _GLOBAL_WAVEFORM_CACHE.clear()
    _GLOBAL_WAVEFORM_CACHE.update({"audio": audio, "sr": sr})

    # Diarisation → patient speech
    patient_segs = vad_segmentation(audio, sr)

    # Build speech buffer for syllable detection
    concat_wave = combine_audio_segments(audio, sr, patient_segs)

    # Hybrid syllable counting
    syl_cnt, syl_debug = hybrid_syllable_count(
        concat_wave, sr, pred_text=pred_text, ctc_conf=ctc_conf
    )
    debug_print(f"[Syllable] nuclei={syl_debug.get('nuclei')}  "
                f"onsets={syl_debug.get('onsets')}  "
                f"chosen={syl_cnt}")

    # Median syllable gap (sec) for adaptive pause threshold
    if syl_cnt > 1 and len(concat_wave) > 0:
        total_speech_dur = sum(seg["duration"] for seg in patient_segs)
        median_gap = total_speech_dur / syl_cnt
    else:
        median_gap = None
    _GLOBAL_WAVEFORM_CACHE["median_syll_gap"] = median_gap

    # Pause list
    pauses, total_pause = count_pauses_and_segments(patient_segs, None)

    # ------------------------------------------------------------------ #
    #  Speech-/articulation-rate metrics (intra- and inter-pause aware)
    # ------------------------------------------------------------------ #
    patient_time = sum(seg["duration"] for seg in patient_segs)

    intra_pause = sum(
        p["duration"]
        for p in pauses
        if any(seg["start"] <= p["start"] and p["end"] <= seg["end"]
               for seg in patient_segs)  # pause fully inside a segment
    )

    speaking_time = max(patient_time - intra_pause, 1e-6) # -1  # avoid /0
    task_time = patient_segs[-1]["end"] - patient_segs[0]["start"]
    total_pause = sum(p["duration"] for p in pauses)
    # syl_cnt = syl_cnt +1
    speech_rate = syl_cnt / task_time
    artic_rate = syl_cnt / speaking_time
    pause_ratio = total_pause / task_time

    # ---------------- compose feature dict in GUI-expected schema ----------
    mean_pause = total_pause / len(pauses) if pauses else 0.0
    disflu_cnt = sum(1 for w in dp_info or [] if w.get("disfluency_flag"))

    features = dict(
        # GUI display
        task_duration=round(task_time, 3),
        syllable_count=syl_cnt,
        speech_rate=round(speech_rate, 3),
        articulation_rate=round(artic_rate, 3),
        pause_count=len(pauses),
        total_pause_duration=round(total_pause, 3),
        mean_pause_duration=round(mean_pause, 3),
        pause_ratio=round(pause_ratio, 3),
        disfluency_count=disflu_cnt,
        # extras used by plot
        speech_chunks=[
            dict(chunk_start=s["start"], chunk_end=s["end"]) for s in patient_segs
        ],
    )

    return features, pauses

# --------------------------------------------------------------------------- #
# 8 · Pitch extraction  (unchanged)
# --------------------------------------------------------------------------- #
def compute_pitch_array(
    audio: np.ndarray,
    sr: int = 16_000,
    hop_length: int = 512,
) -> Tuple[np.ndarray, float]:
    f0, _, _ = librosa.pyin(
        audio,
        fmin=75,
        fmax=600,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )
    # Median-filter to smooth octave errors
    f0_smooth = np.copy(f0)
    from scipy.signal import medfilt  # local import keeps dependency optional

    f0_smooth[f0_smooth > 0] = medfilt(f0_smooth[f0_smooth > 0], kernel_size=5)
    return f0_smooth.astype(np.float32), hop_length / sr


def compute_global_pitch_stats(pitch: np.ndarray) -> Tuple[float, float]:
    voiced = pitch[pitch > 1.0]
    if len(voiced) == 0:
        return 0.0, 0.0
    return float(np.mean(voiced)), float(np.std(voiced))

# --------------------------------------------------------------------------- #
# helper: build char string (= one symbol per column except blank)            #
# --------------------------------------------------------------------------- #
def get_chars_from_model_config(model, processor) -> str:
    """
    Returns a string of length C-1 (C = vocab size) where the n-th character
    corresponds *exactly* to logit column n (blank column excluded).

    • If a token is more than one UTF-8 code-point, its *first code-point*
      is used.
    • If the token is the word-delimiter (usually '|') we keep it.
    • All tokens are coerced to Unicode str; integers can never appear.
    """
    vocab = processor.tokenizer.get_vocab()
    blank_tok = processor.tokenizer.pad_token or "<pad>"
    word_delim = getattr(processor.tokenizer, "word_delimiter_token", "|")

    # id → token list ordered by id
    id2tok = {idx: tok for tok, idx in vocab.items()}
    tokens = [id2tok[i] for i in range(len(id2tok))]

    # move blank to end and drop from char list
    if blank_tok in tokens:
        tokens.remove(blank_tok)
    char_list: list[str] = []
    for tok in tokens:
        if tok == word_delim:
            char_list.append(word_delim)
        else:
            char_list.append(str(tok)[0])  # one Unicode code-point
    return "".join(char_list)

# --------------------------------------------------------------------------- #
# 9 · helper: align probability matrix so that blank = last column
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# helper: make probs & chars compatible with pyctcdecoder                      #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# helper: align probs & chars for pyctcdecoder                                 #
# --------------------------------------------------------------------------- #
def _prep_decoder_inputs(
    logits: torch.Tensor,
    processor: Wav2Vec2Processor,
) -> SimpleNamespace:
    """
    • Moves the CTC blank column to the end by *swapping* (no left-shift)
    • Builds a char string of length C-1 (blank excluded) where position i
      matches column i in the reordered matrix.
    • Returns mean max-posterior `conf` ∈ (0,1].
    """
    EPS = 1e-12
    probs = torch.softmax(logits, dim=-1)[0].double().cpu().numpy()

    vocab = processor.tokenizer.get_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    blank_tok = processor.tokenizer.pad_token or "<pad>"
    blank_id  = vocab[blank_tok]
    C = probs.shape[1]

    if blank_id != C - 1:
        # swap blank column with last column (no shift of others)
        probs[:, [blank_id, C - 1]] = probs[:, [C - 1, blank_id]]
        id2tok[blank_id], id2tok[C - 1] = id2tok[C - 1], id2tok[blank_id]

    # char list for first C-1 columns
    word_delim = getattr(processor.tokenizer, "word_delimiter_token", "|")
    char_list = []
    for i in range(C - 1):
        tok = str(id2tok[i])
        if tok == "":
            tok = "?"                     # safety
        char_list.append(tok if tok == word_delim else tok[0])

    chars = "".join(char_list)
    assert len(chars) == C - 1, "chars/column mismatch after swap"

    conf = float(np.mean(np.max(probs, axis=-1)))
    return SimpleNamespace(probs=probs, chars=chars, conf=conf)


# --------------------------------------------------------------------------- #
# 10 · CTC → text
# --------------------------------------------------------------------------- #
def decode_text(
    probs: np.ndarray,
    chars: str,
    decoder_method: str = "beam_search",
    lm_text: str | None = None,
    beam_width: int = 128,
) -> str:
    """

    Parameters
    ----------
    probs : np.ndarray [T,C]   (blank = last column!!)
    chars : str of length C-1  (blank excluded)
    decoder_method : "beam_search" | "best_path"
    lm_text : optional str  → ignored in the built-in decoder.
    """
    import ctc_decoder  # local import avoids hard dep at module import

    if decoder_method == "best_path":
        return ctc_decoder.best_path(probs, chars)

    # ---- beam search ------------------------------------------------------
    sig = inspect.signature(ctc_decoder.beam_search)
    kwargs = dict(beam_width=beam_width)
    if "lm_text" in sig.parameters and lm_text:
        kwargs["lm_text"] = lm_text

    try:
        return ctc_decoder.beam_search(probs, chars, **kwargs)
    except (IndexError, ValueError) as e:  # mis-match or invalid alphabet
        # fallback to best-path to avoid crash
        debug_print(
            f"[decode_text] Beam-search failed ({e}); "
            f"falling back to best_path."
        )
        return ctc_decoder.best_path(probs, chars)


# --------------------------------------------------------------------------- #
# 11 · high-level ASR wrapper  (returns text *and* confidence)
# --------------------------------------------------------------------------- #
def decode_text_from_model(
    audio_path: str | Path,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    decoder_method: str = "beam_search",
    lm_text: str | None = None,
) -> Tuple[str, float]:
    """
    Returns
    -------
    text : str
    conf : float   (mean max-posterior, already clipped to (0,1])
    """
    audio, sr = load_audio(audio_path, target_sr=16_000)
    iv = processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(model.device)

    with torch.no_grad():
        logits = model(iv).logits

    dec_in = _prep_decoder_inputs(logits, processor)
    text = decode_text(
        dec_in.probs,
        dec_in.chars,
        decoder_method=decoder_method,
        lm_text=lm_text,
    )
    text = text.replace("<","")
    return text, dec_in.conf

# --------------------------------------------------------------------------- #
# 10 · Forced Alignment, Disfluency labelling, etc.  (unchanged)
# --------------------------------------------------------------------------- #
def forced_alignment_charlevel(audio_path: str, pred_text: str,
                               model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor,
                               sr: int = 16000) -> list:
    speech, _ = load_audio(audio_path, target_sr=sr)
    if speech.size == 0:
        return []
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    iv = inputs.input_values.to(model.device)
    with torch.no_grad():
        logits = model(iv).logits[0]
    logp = F.log_softmax(logits, dim=-1).cpu().numpy()
    probs = medfilt(logp, kernel_size=(3,1))
    T = probs.shape[0]
    audio_dur = len(speech)/sr

    allowed = list(get_chars_from_model_config(model, processor))
    blank = processor.tokenizer.pad_token or "_"
    space = getattr(processor.tokenizer, "word_delimiter_token", "|")
    filt = "".join(c if c in allowed else space for c in pred_text)
    chars = list(filt)
    if not chars:
        return []
    params = cs.CtcSegmentationParameters(char_list=allowed)
    params.index_duration = audio_dur/T
    params.blank = blank
    ground, utt_begin = cs.prepare_text(params, chars)
    if not utt_begin:
        return []
    timings, char_probs, _ = cs.ctc_segmentation(params, probs, ground)
    segs = cs.determine_utterance_segments(params, utt_begin, char_probs, timings, chars)

    words = []
    buf, conf_sum = [], 0.0
    for i,ch in enumerate(chars):
        st, ed, cf = segs[i]
        if not buf:
            buf_start = st
        if ch == space:
            w = "".join(buf)
            if w:
                words.append((w, buf_start, st, conf_sum/len(buf)))
            buf, conf_sum = [], 0.0
        else:
            buf.append(ch)
            conf_sum += cf
    if buf:
        words.append(("".join(buf), buf_start, segs[-1][1], conf_sum/len(buf)))
    return words

def disfluency_pause_judgement(word_segments: list,
                               min_intra: float = 0.10,
                               filler_words: set = None) -> list:
    if filler_words is None:
        filler_words = {"um", "o", "erm", "uh", "er",
                        "emm", "em", "umm", "hmm", "eh",
                        "ed", "s", "mm", "oh", "ah", "hm",
                        "e", "amm", "ur", "m", "e", "si",
                        "om", "oi", "t", "ar", "th","emmm"}
    out = []
    for i,(w,st,ed,cf) in enumerate(word_segments):
        flag = 1 if w.lower() in filler_words else 0
        if i>0 and word_segments[i-1][0].lower()==w.lower():
            flag = 1
        out.append({
            "word":w,"start_sec":st,"end_sec":ed,
            "conf":cf,"disfluency_flag":flag,
            "pause_label":None,"pause_duration":0.0,
            "pitch_mean":0.0,"pitch_norm":0.0
        })
    for i in range(len(out)-1):
        gap = out[i+1]["start_sec"]-out[i]["end_sec"]
        if gap>=min_intra:
            out[i]["pause_label"]="pause"
            out[i]["pause_duration"]=gap
    return out


# ======================= 带停顿标注文本 =======================
def make_annotated_text_with_fused_pauses(dp_info: list, fused_pauses: list) -> str:
    txt, pi = "", 0
    for wi in dp_info:
        while pi < len(fused_pauses) and fused_pauses[pi]["start"] < wi["start_sec"]:
            p = fused_pauses[pi]
            txt += f" (pause@{p['start']:.2f}s,dur={p['duration']:.2f}s) "
            pi += 1
        if wi["disfluency_flag"]:
            txt += f" [{wi['word']}@{wi['start_sec']:.2f}-{wi['end_sec']:.2f}s] "
        else:
            txt += " " + wi["word"] + " "
    while pi < len(fused_pauses):
        p = fused_pauses[pi]
        txt += f" (pause@{p['start']:.2f}s,dur={p['duration']:.2f}s) "
        pi += 1
    return txt.strip()

# --------------------------------------------------------------------------- #
# 11 · Plotting  (unchanged)
# --------------------------------------------------------------------------- #
# ======================= 绘图逻辑优化 =======================
def plot_inference_results(speech: np.ndarray,
                           sr: int,
                           dp_info: list,
                           fused_pauses: list,
                           global_features: dict,
                           pitch_array: np.ndarray = None,
                           tpf: float = None,
                           plot_output_dir: str = "output") -> str:
    """
    绘制并保存推理结果图：
      - 上图：音频波形 + 融合停顿（紫色半透明区间 + 时长标注） +
               不流畅标记（红色半透明区间 + 单词标注） +
               语音块边界（绿色虚实线）
      - 下图：Pitch 轮廓 + 平均音高线 + 语音块边界
    返回保存的图像路径。
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # 时间轴
    t_wave = np.arange(len(speech)) / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # —— 上图：波形
    ax1.plot(t_wave, speech, color="blue", linewidth=0.8, label="Waveform")

    # 融合停顿区间（半透明矩形 + 时长文本）
    for p in fused_pauses:
        ax1.axvspan(p["start"], p["end"], color="magenta", alpha=0.3)
        ax1.text((p["start"] + p["end"]) / 2,
                 ax1.get_ylim()[1] * 0.8,
                 f"{p['duration']:.2f}s",
                 color="magenta",
                 fontsize=8,
                 rotation=90,
                 ha="center",
                 va="bottom")

    # 不流畅标记区间（半透明矩形 + 单词文本）
    for dp in dp_info:
        if dp["disfluency_flag"] == 1:
            ax1.axvspan(dp["start_sec"], dp["end_sec"], color="red", alpha=0.3)
            ax1.text((dp["start_sec"] + dp["end_sec"]) / 2,
                     ax1.get_ylim()[1] * 0.6,
                     dp["word"],
                     color="red",
                     fontsize=8,
                     rotation=45,
                     ha="center",
                     va="bottom")

    # 语音块边界
    for chunk in global_features.get("speech_chunks", []):
        ax1.axvline(chunk["chunk_start"], color="green", linestyle="--", linewidth=1)
        ax1.axvline(chunk["chunk_end"],   color="green", linestyle="-",  linewidth=1)

    ax1.set_ylabel("Amplitude")
    ax1.set_title("Audio Waveform with Pauses, Disfluencies and Speech Chunks")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # —— 下图：Pitch
    if pitch_array is not None and tpf is not None:
        t_pitch = np.arange(len(pitch_array)) * tpf
        mean_pitch, _ = compute_global_pitch_stats(pitch_array)

        ax2.plot(t_pitch, pitch_array, linewidth=0.8, label="Pitch (Hz)")
        ax2.axhline(mean_pitch, color="red", linestyle="--", linewidth=1,
                    label=f"Mean = {mean_pitch:.1f} Hz")

        # 语音块边界
        for chunk in global_features.get("speech_chunks", []):
            ax2.axvline(chunk["chunk_start"], color="green", linestyle="--", linewidth=1)
            ax2.axvline(chunk["chunk_end"],   color="green", linestyle="-",  linewidth=1)

        ax2.set_ylabel("Pitch (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Pitch Contour with Speech Chunks")
        ax2.legend(loc="upper right")
        ax2.grid(True)
    else:
        ax2.set_visible(False)

    plt.tight_layout()

    os.makedirs(plot_output_dir, exist_ok=True)
    plot_path = os.path.join(plot_output_dir, "inference_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_path


# ======================= 主推理流程 =======================
def run_inference_and_seg(audio_path: str,
                          model: Wav2Vec2ForCTC,
                          processor: Wav2Vec2Processor,
                          sr: int = 16000,
                          hop_length: int = 512,
                          decoder_method: str = "beam_search",
                          # lm_text: str = None,
                          plot_output_dir: str = "output"):
    start_time = time.perf_counter()

    # 1. 加载 & 解码
    speech, _ = load_audio(audio_path, target_sr=sr)
    if speech.size == 0:
        debug_print(f"Cannot load audio: {audio_path}")
        return "", {}, []
    pred_text, ctc_conf = decode_text_from_model(audio_path, model, processor,
                                                 decoder_method=decoder_method)
    debug_print(f"Decoded Transcript: {pred_text}")
    wer_val = wer(" ".join(["a"] * len(pred_text.split())), pred_text) if pred_text else 1.0


    # 2. 对齐 & 不流畅标注
    word_segs = forced_alignment_charlevel(audio_path, pred_text, model, processor, sr=sr)
    if not word_segs:
        debug_print("No alignment result.")
        return "", {}, []
    dp_info = disfluency_pause_judgement(word_segs, min_intra=0.10)

    # 3. 全局特征提取

    global_features, fused_pauses = extract_features_from_audio(
        audio_path, dp_info,
        pred_text=pred_text,
        ctc_conf=ctc_conf
    )
    # --- OPTIONAL PRAAT ENHANCEMENT ---------------------------------
    if CFG.get("praat", {}).get("enable", False):
        from .speech_tools.praat_bridge import run_praat
        praat_feats = run_praat(
            audio_path,
            dp_info,
            global_features,
            praat_exe=CFG["praat"]["exe_path"],
            scripts={
                "nuclei": "scripts/Syllable_script.praat",
                "extract": "scripts/Speech_analysis_script.praat",
            },
        )
        global_features.update(praat_feats)
        global_features["praat_success"] = praat_feats["praat_success"]

    # 4. 生成带停顿标注文本
    annotated_text = make_annotated_text_with_fused_pauses(dp_info, fused_pauses)

    debug_print("Annotated Text:\n" + annotated_text)
    debug_print("Global Features:\n" + json.dumps(global_features, indent=2))

    # 5. 计算 pitch
    pitch_array, tpf = compute_pitch_array(speech, sr=sr,
                                           hop_length=hop_length)

    # 6. 调用新的绘图函数
    plot_path = plot_inference_results(
        speech=speech,
        sr=sr,
        dp_info=dp_info,
        fused_pauses=fused_pauses,
        global_features=global_features,
        pitch_array=pitch_array,
        tpf=tpf,
        plot_output_dir=plot_output_dir
    )
    global_features["plot_path"] = plot_path
    debug_print(f"Plot saved to {plot_path}")

    elapsed = time.perf_counter() - start_time
    debug_print(f"run_inference_and_seg completed in {elapsed:.2f}s")
    return annotated_text, global_features, dp_info


# ======================= 参数配置 & 主函数 =======================
class args:
    pretrained_processor = "xyang2/ASACA_ASR"
    pretrained_model = "xyang2/ASACA_ASR"
    test_audio_path = "Mei_CT/SC001 IMAGE.wav"
    output_dir = "output"
    plot_output_dir = "output"
    sr = 16000
    hop_length = 512

def main(arguments):
    try:
        processor = Wav2Vec2Processor.from_pretrained(arguments.pretrained_processor)
        model     = Wav2Vec2ForCTC.from_pretrained(arguments.pretrained_model)
    except Exception as e:
        debug_print(f"Model load error: {e}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_print(f"Using device: {device}")
    model.to(device).eval()

    ann, feats, dp = run_inference_and_seg(
        arguments.test_audio_path, model, processor,
        sr=arguments.sr, hop_length=arguments.hop_length,
        decoder_method="beam_search",
        plot_output_dir=arguments.plot_output_dir
    )

    # 保存 JSON
    os.makedirs(arguments.output_dir, exist_ok=True)
    with open(os.path.join(arguments.output_dir, "inference_result.json"), "w", encoding="utf-8") as f:
        json.dump({
            "Annotated Text": ann,
            "Global Features": feats,
            "Detailed Info": dp
        }, f, ensure_ascii=False, indent=2)

    # ==================== 保存 Excel（只保留标量特征） ====================
    try:
        # 1. 注释文本
        df_ann = pd.DataFrame({"Annotated Text": [ann]})

        # 2. 过滤全局特征中的列表/字典，只保留标量值
        scalar_feats = {k: v for k, v in feats.items()
                        if not isinstance(v, (list, dict))}
        df_feats = pd.DataFrame(list(scalar_feats.items()), columns=["Feature", "Value"])

        # 3. 详细对齐数据
        df_dp = pd.DataFrame(dp)

        # 4. 写入 Excel，使用 openpyxl 引擎并自动关闭
        excel_path = os.path.join(args.output_dir, "inference_result.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_ann.to_excel(writer, sheet_name="AnnotatedText", index=False)
            df_feats.to_excel(writer, sheet_name="GlobalFeatures", index=False)
            df_dp.to_excel(writer, sheet_name="DetailedInfo", index=False)

        debug_print(f"Excel results saved to {excel_path}")
    except Exception as e:
        debug_print(f"Excel save error: {e}")


if __name__ == "__main__":
    main(args())

