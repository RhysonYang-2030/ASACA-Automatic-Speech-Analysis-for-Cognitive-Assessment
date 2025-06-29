# ── speech_tools/praat_bridge.py ──────────────────────────────────────
"""
Robust Praat bridge – 2025-05 final

Features
--------
✔ UTF-8 safe subprocess I/O
✔ Auto-build 6-tier TextGrid (1 meta + 5 real tiers)
✔ Works even if speech_chunks 为空（整段设为发声）
✔ praat_success=True 只要任何脚本真正产出内容且返回码==0
✔ 控制台 debug：显示 Praat stdout/stderr 路径
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List
import codecs
from speech_tools.textgrid_utils import build_textgrid


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _wav_dur(wav: Path) -> float:
    with wave.open(str(wav), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)

def _decode_auto(raw: bytes) -> str:
    """Try UTF-16LE first (Praat default), else UTF-8."""
    for enc in ("utf-16-le", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")

# ---------- nuclei stdout parser  (now split by commas) --------------
def _parse_nuclei_stdout(txt: str) -> Dict[str, float]:
    """
    Expect line like:
        sound, nsyll, npause, dur, phon, sr, ar, asd
    """
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    data = lines[-1]
    parts = [p.strip() for p in data.split(",")]
    if len(parts) != 8:
        return {}
    _, nsyll, npause, dur, phon, sr, ar, asd = parts
    try:
        return dict(
            syllable_count=int(nsyll),
            pause_count=int(npause),
            task_duration=float(dur),
            phonation_time=float(phon),
            speech_rate=float(sr),
            articulation_rate=float(ar),
            average_syllable_duration=float(asd),
        )
    except ValueError:
        return {}


def _first_gp_file(out_dir: Path) -> Path | None:
    for p in out_dir.rglob("global_parameters.txt"):
        try:
            if p.stat().st_size > 20:
                return p
        except OSError:
            continue
    return None


def _parse_gp(p: Path, native_total: int = 0) -> Dict[str, float]:
    # ---------- 读取 txt ----------
    with p.open("r", encoding="utf-8", errors="replace") as fp:
        rows = list(csv.reader(fp, delimiter="\t"))
    if len(rows) < 2:
        return {}
    hdr, val = rows[0], rows[1]
    rec = dict(zip(hdr, val))

    i = lambda x: int(float(x)) if x.strip() else 0
    f = lambda x: float(x)      if x.strip() else 0.0

    within  = i(rec.get("nWithinPauses", ""))
    between = i(rec.get("nBetweenPauses", ""))
    total   = i(rec.get("npauses", ""))

    # --------- 兜底 1: total = within+between ----------
    if total == 0 and (within or between):
        total = within + between


    # --------- 兜底 2: fallback to native path ----------
    if total == 0 and native_total:
        total = native_total


    return {
        "speech_rate_sentence_avg":       f(rec.get("Speechrate", "")),
        "articulation_rate_sentence_avg": f(rec.get("Artrate", "")),
        "pause_count_total":              total,
        "pause_count_within":             within,
        "pause_count_between":            between,
        "disfluency_count":               i(rec.get("nDysfluences", "")),
        "file_duration":                  f(rec.get("FileDuration", "")),
    }


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def run_praat(
    audio_path: str | Path,
    alignment: List[Dict],
    native_feats: Dict[str, float],
    praat_exe: str,
    scripts: Dict[str, str],    # nuclei / extract = 原始脚本
) -> Dict[str, float]:
    wav_src = Path(audio_path).resolve()
    wav_dur = native_feats.get("task_duration") or _wav_dur(wav_src)

    with tempfile.TemporaryDirectory() as td:
        tmp     = Path(td)
        wav     = shutil.copy2(wav_src, tmp / wav_src.name)
        tg      = tmp / f"{wav_src.stem}.TextGrid"
        out_dir = tmp / "extract_out"; out_dir.mkdir()

        # ---------- build TextGrid ---------------------------------
        segs = []
        for seg in native_feats.get("speech_chunks", []):
            try:
                if isinstance(seg, dict):
                    segs.append((float(seg["chunk_start"]), float(seg["chunk_end"])))
                elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    segs.append(tuple(map(float, seg[:2])))
            except Exception:
                continue
        if not segs:
            segs = [(0.0, wav_dur)]                   # 整段算发声
        build_textgrid(alignment, segs, wav_dur, tg, patient_id=wav_src.stem[:5])

        # ---------- 1. syllable ------------------------------------
        cmd1 = [praat_exe, "--run", scripts["nuclei"],
                "-25", "2", "0.3", "no", str(tmp)]
        res1 = subprocess.run(
            cmd1,
            capture_output=True,
            text=False,  # ← capture raw bytes
            check=False,
        )
        stdout1 = _decode_auto(res1.stdout)
        nuclei_feats = _parse_nuclei_stdout(stdout1) if res1.returncode == 0 else {}

        # ---------- 2. extract -------------------------------------
        cmd2 = [praat_exe, "--run", scripts["extract"],
                str(tmp), str(out_dir), str(tmp)]

        res2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=False,
            check=False,
        )
        stdout2 = _decode_auto(res2.stdout)
        stderr2 = _decode_auto(res2.stderr)

        native_total = native_feats.get("pause_count", 0)
        gp_file = _first_gp_file(out_dir) if res2.returncode == 0 else None
        extract_feats = _parse_gp(gp_file, native_total) if gp_file else {}


        # ---------- 判定成功 ----------------------------------------
        praat_ok = (res1.returncode == 0 and nuclei_feats) or \
                   (res2.returncode == 0 and extract_feats)
        # === NEW: keep native pause_count / speech_rate ==============
        for d in (nuclei_feats, extract_feats):
            for k in ("speech_rate",
                      "pause_count",
                      "pause_count_total",
                      "task_duration"):
                d.pop(k, None)  # 删除对应键（若存在）
        if not praat_ok:
            print("\n[PraatBridge-DEBUG] nuclei-stdout ↓↓↓\n", res1.stdout.strip(),
                  "\n[PraatBridge-DEBUG] nuclei-stderr ↓↓↓\n", res1.stderr.strip(),
                  "\n[PraatBridge-DEBUG] extract-stdout ↓↓↓\n", res2.stdout.strip(),
                  "\n[PraatBridge-DEBUG] extract-stderr ↓↓↓\n", res2.stderr.strip(), "\n")

        merged: Dict[str, float] = {**nuclei_feats, **extract_feats,
                                    "praat_success": praat_ok}
        return merged
