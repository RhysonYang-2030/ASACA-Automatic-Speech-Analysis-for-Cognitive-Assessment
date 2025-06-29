"""batch_process.py

Batch‑process a folder of .wav recordings with the **inference_V4** pipeline.
Extract the core fluency metrics (speech‑rate, articulation‑rate, pause‑count,
 task‑duration, dysfluency‑count), write them into **one Excel workbook**
(one row per file) and save the diagnostic plot for every recording as
`plots/<file>.png` – the plot gains a clean white header so the title never
obscures the original figure.

Highlights
==========
* **Single model load** in sequential mode and **one per worker** in parallel
  mode (via `initializer`) – avoids costly re‑loads.
* **Cross‑platform font handling** (DejaVuSans → Arial → default bitmap).
* **Adaptive header height** based on title text height (no clipping).
* Multiprocessing stream with `imap_unordered` so memory stays low.
* Interactive Tk pickers pop up only when the relevant CLI flag is omitted.

Version 2.5  – 2025‑05‑06
------------------------
* Fixed missing `from_pretrained(model_ckpt)` typo.
* Finished sequential / parallel runners, Excel writer and summary.
* Added `torch.no_grad()` wrapper around inference.
* Warns and exits early when no .wav files are found.

CLI flags (all optional – GUI prompts appear if missing)
-------------------------------------------------------
    --audio_dir          folder containing .wav files (non‑recursive)
    --model_ckpt         HF hub ID OR local path to fine‑tuned model
    --processor_ckpt     matching processor checkpoint
    --output_excel       destination workbook (default batch_features.xlsx)
    --plots_dir          destination for PNGs (default ./plots)
    --device             auto | cpu | cuda   (auto = GPU if present)
    --workers            parallel processes (default 1 = sequential)

Examples
--------
Sequential, GPU if available – GUI will ask for paths if flags omitted:
    python batch_process.py --device auto

Parallel, CPU, 4 workers, explicit paths:
    python batch_process.py \
        --audio_dir "D:/MS/audio" \
        --model_ckpt ./ckpt/model \
        --processor_ckpt ./ckpt/processor \
        --device cpu --workers 4
"""
from __future__ import annotations
import numpy as np
import argparse
import shutil
import sys
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# Headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")

try:
    from inference_V4 import run_inference_and_seg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from inference_V4 import run_inference_and_seg  # type: ignore

# ---------------------------------------------------------------------------
#  Globals (one per process)
# ---------------------------------------------------------------------------
_GLOBAL: Dict[str, Any] = {}

np.seterr(all="ignore")
def _init_worker(model_ckpt: str, processor_ckpt: str, device: str):
    """Pool initializer – loads model/processor once per worker."""
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # local import

    processor = Wav2Vec2Processor.from_pretrained(processor_ckpt)
    model = Wav2Vec2ForCTC.from_pretrained(model_ckpt)
    model.to(device).eval()
    _GLOBAL.update(model=model, processor=processor, device=device)


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def choose_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available – falling back to CPU.")
        return "cpu"
    return pref


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)  # Pillow <10 fallback  # type: ignore[attr-defined]


def _get_font(size: int = 24) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_title(original_png: Path, title: str, output_png: Path):
    """Add a white header with *adaptive* height so content is never covered."""
    img = Image.open(original_png).convert("RGB")
    draw_tmp = ImageDraw.Draw(img)
    font = _get_font(24)
    text_w, text_h = _measure_text(draw_tmp, title, font)
    header_px = text_h + 20  # 10‑px top & bottom margin

    new_img = Image.new("RGB", (img.width, img.height + header_px), "white")
    new_img.paste(img, (0, header_px))
    draw = ImageDraw.Draw(new_img)
    draw.text(((img.width - text_w) // 2, (header_px - text_h) // 2), title, fill="black", font=font)
    new_img.save(output_png)


def extract_core_metrics(glob_feat: Dict) -> Dict[str, float]:
    return dict(
        speech_rate=glob_feat.get("speech_rate"),
        articulation_rate=glob_feat.get("articulation_rate"),
        pause_count=glob_feat.get("pause_count"),
        task_duration=glob_feat.get("task_duration"),
        disfluency_count=glob_feat.get("disfluency_count"),
    )


# ---------------------------------------------------------------------------
#  Interactive GUI helpers (Tk)
# ---------------------------------------------------------------------------

def _tk_choose_directory(prompt: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=prompt)
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[ERROR] Could not open GUI ({e}).")
        return None


def _maybe_prompt_paths(args) -> Tuple[str, str, str]:
    audio_dir = args.audio_dir or _tk_choose_directory("Select folder with .wav recordings")
    model_ckpt = args.model_ckpt or _tk_choose_directory("Select fine‑tuned model checkpoint folder")
    processor_ckpt = args.processor_ckpt or _tk_choose_directory("Select processor checkpoint folder")

    if not all([audio_dir, model_ckpt, processor_ckpt]):
        print("[ERROR] Required path not provided. Exiting.")
        sys.exit(1)
    return audio_dir, model_ckpt, processor_ckpt


# ---------------------------------------------------------------------------
#  Processing functions
# ---------------------------------------------------------------------------

def _process_one(audio_path: Path, plots_dir: Path, model, processor) -> Dict[str, Any]:
    temp_dir = plots_dir / f"tmp_{audio_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with torch.no_grad():
            _, global_features, _ = run_inference_and_seg(
                audio_path=str(audio_path),
                model=model,
                processor=processor,
                plot_output_dir=str(temp_dir),
            )

        raw_plot_path = Path(global_features.get("plot_path", ""))
        if not raw_plot_path.exists():
            try:
                raw_plot_path = next(temp_dir.glob("*.png"))
            except StopIteration:
                raise FileNotFoundError("Plot PNG not found for " + audio_path.name)

        final_plot = plots_dir / f"{audio_path.stem}.png"
        overlay_title(raw_plot_path, audio_path.name, final_plot)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    metrics = extract_core_metrics(global_features)
    metrics["file_name"] = audio_path.name
    return metrics


def _run_single(params: Tuple[str | Path, str | Path]):
    """Entry point for multiprocessing workers. Uses globals loaded in initializer."""
    audio_path, plots_dir = params
    audio_path = Path(audio_path)
    plots_dir = Path(plots_dir)
    model = _GLOBAL["model"]
    processor = _GLOBAL["processor"]
    return _process_one(audio_path, plots_dir, model, processor)


# ---------------------------------------------------------------------------
#  CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Batch inference over .wav recordings.")
    p.add_argument("--audio_dir")
    p.add_argument("--model_ckpt")
    p.add_argument("--processor_ckpt")
    p.add_argument("--output_excel", default="batch_features_MS_SC019.xlsx")
    p.add_argument("--plots_dir", default="plots_MS_CT_Praat")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def sequential_run(wav_files: List[Path], plots_dir: Path, device: str,
                   model_ckpt: str, processor_ckpt: str) -> List[Dict]:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

    processor = Wav2Vec2Processor.from_pretrained(processor_ckpt)
    model = Wav2Vec2ForCTC.from_pretrained(model_ckpt)
    model.to(device).eval()

    results = []
    for wav in wav_files:
        try:
            res = _process_one(wav, plots_dir, model, processor)
            results.append(res)
            print(f"[OK] {wav.name}")
        except Exception as e:
            print(f"[FAIL] {wav.name}: {e}")
    return results


def parallel_run(wav_files: List[Path], plots_dir: Path, device: str,
                 model_ckpt: str, processor_ckpt: str, workers: int) -> List[Dict]:
    params_iter = [(wav, plots_dir) for wav in wav_files]

    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(model_ckpt, processor_ckpt, device)) as pool:
        results = []
        for res in pool.imap_unordered(_run_single, params_iter):
            if isinstance(res, dict):
                results.append(res)
                fname = res.get("file_name", "?")
                print(f"[OK] {fname}")
        pool.close()
        pool.join()
    return results


def main():
    args = parse_args()
    audio_dir, model_ckpt, processor_ckpt = _maybe_prompt_paths(args)

    device = choose_device(args.device)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        print("[ERROR] No .wav files found in", audio_dir)
        sys.exit(1)

    print(f"Found {len(wav_files)} files. Running on {device} ({'parallel' if args.workers > 1 else 'sequential'})...")

    if args.workers <= 1:
        results = sequential_run(wav_files, plots_dir, device, model_ckpt, processor_ckpt)
    else:
        results = parallel_run(wav_files, plots_dir, device, model_ckpt, processor_ckpt, args.workers)

    if not results:
        print("[ERROR] No results generated – aborting Excel write.")
        sys.exit(1)

    df = pd.DataFrame(results).sort_values("file_name")
    df.to_excel(args.output_excel, index=False)

    print("\nDone!  Processed", len(df), "files →", args.output_excel)


if __name__ == "__main__":
    mp.freeze_support()  # For Windows entry‑point safety
    main()
