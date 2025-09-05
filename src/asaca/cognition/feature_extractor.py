"""
Combine ASACA global features with dictionary-based counts.
Called by model_training.py **and** at GUI inference-time.
"""
from __future__ import annotations
import json, joblib
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from asaca.inference import run_inference_and_seg           # ASACA core
from asaca.cognition.nlp_utils import preprocess
from asaca.cognition.dictionary_builder import LABEL2NAME

# ----------------------------------------------------------------------
class CognitionDictionaryBank:
    """Lazy-loads the three Excel dictionaries and offers frequency lookup."""
    def __init__(self, dict_dir):
        dict_dir = Path(dict_dir)
        self.dicts = {}
        for name in ("HC","AD","MCI"):
            df = pd.read_excel(dict_dir / f"{name}_dict.xlsx")
            self.dicts[name] = set(df["word"].astype(str))

    def count_hits(self, token_list: list[str]) -> Tuple[int,int,int]:
        hc = ad = mci = 0
        for tok in token_list:
            if tok in self.dicts["HC"]:
                hc += 1
            if tok in self.dicts["AD"]:
                ad += 1
            if tok in self.dicts["MCI"]:
                mci += 1
        return hc, ad, mci

# ----------------------------------------------------------------------
SCALAR_KEYS = [                    # keep in sync w/ GUI radar etc.
    "task_duration", "syllable_count", "speech_rate", "articulation_rate",
    "pause_count", "total_pause_duration", "mean_pause_duration",
    "pause_ratio", "disfluency_count"
]
DICT_FEATS = ["bias", "HC_cnt", "AD_cnt", "MCI_cnt", "HC_ratio", "AD_ratio", "MCI_ratio"]
plot_dir = "output"
def extract_features(
    wav_path: Path,
    model,
    processor,
    dict_bank: CognitionDictionaryBank,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict]:
    annotated_text, global_feats, _ = run_inference_and_seg(
        str(wav_path), model, processor, sr=16000, hop_length=512,
        decoder_method="beam_search", plot_output_dir=plot_dir
    )

    tokens = preprocess(annotated_text)
    n_tok  = max(1, len(tokens))               # avoid div-by-0
    hc, ad, mci = dict_bank.count_hits(tokens)

    ### NEW: ratios ----------------------------------------------------
    hc_r = hc / n_tok
    ad_r = ad / n_tok
    mci_r = mci / n_tok
    # ------------------------------------------------------------------

    part1 = np.array([global_feats.get(k, 0.0) for k in SCALAR_KEYS], dtype=float)
    part2 = np.array(
        [1.0, hc, ad, mci, hc_r, ad_r, mci_r], dtype=float
    )

    return np.concatenate([part1, part2]), global_feats | {
        "transcription": annotated_text,
        "token_count": n_tok,
        "dict_counts": (hc, ad, mci),
    }

# ----------------------------------------------------------------------
def build_feature_vector(
    transcription: str,
    global_feats: Dict,
    dict_bank: CognitionDictionaryBank,
) -> Tuple[np.ndarray, Dict]:
    """Construct feature vector from pre-computed ASACA results."""
    tokens = preprocess(transcription)
    n_tok = max(1, len(tokens))
    hc, ad, mci = dict_bank.count_hits(tokens)

    hc_r = hc / n_tok
    ad_r = ad / n_tok
    mci_r = mci / n_tok

    part1 = np.array([global_feats.get(k, 0.0) for k in SCALAR_KEYS], dtype=float)
    part2 = np.array([1.0, hc, ad, mci, hc_r, ad_r, mci_r], dtype=float)

    return np.concatenate([part1, part2]), global_feats | {
        "transcription": transcription,
        "token_count": n_tok,
        "dict_counts": (hc, ad, mci),
    }
# ----------------------------------------------------------------------
def batch_build_feature_file(
    meta_xlsx: Path,
    dict_dir: Path,
    out_xlsx: Path,
    processor_path: str,
    model_path: str,
    device: str = "cpu",
    wav_root: Path | None = None,
) -> None:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model     = Wav2Vec2ForCTC.from_pretrained(model_path).to(device).eval()

    bank = CognitionDictionaryBank(Path(dict_dir))

    df = pd.read_excel(meta_xlsx)
    rows = []
    for _i, row in df.iterrows():
        wav = Path(row["wav_file"])
        if not wav.is_absolute() and wav_root is not None:
            wav = Path(wav_root) / wav  # prepend root when path is relative
        if not wav.exists():
            raise FileNotFoundError(f"[WARN] WAV not found – skip: {wav}") # if encounter WAV file is not in the sheet then skip
            continue
        vector, gfeat = extract_features(
            wav, model, processor, bank, device=device
        )
        rows.append({
            "subject_id": row["subject_id"],
            "label":      row["label"],
            "wav_file":   str(wav),
            "vector":     json.dumps(vector.tolist()),
        })
    pd.DataFrame(rows).to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"[✓] Feature vectors written → {out_xlsx}")
# ----------------------------------------------------------------------
# CLI **and** click-to-run entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    ap = argparse.ArgumentParser(
        description="Build feature-vector Excel from a transcript sheet."
                     " Run with --gui to select paths via dialogs.")
    ap.add_argument("--meta_xlsx",      type=str, help="Transcription Excel")
    ap.add_argument("--dict_dir",       type=str, help="Folder with *_dict.xlsx")
    ap.add_argument("--out_xlsx",       type=str, help="Output Excel")
    ap.add_argument("--processor_path", type=str, help="Wav2Vec2 processor dir")
    ap.add_argument("--model_path",     type=str, help="Wav2Vec2 model dir")
    ap.add_argument("--device",         type=str, default="cpu",
                    help="cuda | cpu (default: cpu)")
    ap.add_argument("--gui", action="store_true",
                    help="Ignore other flags and launch file-choosers.")
    args = ap.parse_args()

    # ------------------------------------------------ GUI branch
    if args.gui or not any(
        getattr(args, k) for k in
        ("meta_xlsx", "dict_dir", "out_xlsx", "processor_path", "model_path")
    ):
        tk.Tk().withdraw()                               # hide root window
        try:
            messagebox.showinfo("Feature extractor",
                                "Please choose the REQUIRED files/folders")

            meta_xlsx = filedialog.askopenfilename(
                title="Transcription workbook (meta_xlsx)",
                filetypes=[("Excel", "*.xlsx *.xls")])
            if not meta_xlsx:
                sys.exit("Cancelled.")

            dict_dir = filedialog.askdirectory(
                title="Dictionary folder (dict_dir)")
            if not dict_dir:
                sys.exit("Cancelled.")

            wav_root = filedialog.askdirectory(
                title="Root folder that contains the WAV files (wav_root)")
            if not wav_root:
                sys.exit("Cancelled.")

            out_xlsx = filedialog.asksaveasfilename(
                title="Save feature vectors as…",
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")])
            if not out_xlsx:
                sys.exit("Cancelled.")

            processor_path = filedialog.askdirectory(
                title="Wav2Vec2 processor folder (processor_path)")
            if not processor_path:
                sys.exit("Cancelled.")

            model_path = filedialog.askdirectory(
                title="Wav2Vec2 model folder (model_path)")
            if not model_path:
                sys.exit("Cancelled.")

            device = simpledialog.askstring(
                "Device",
                "Type 'cuda' for GPU or 'cpu' for CPU:",
                initialvalue="cpu") or "cpu"

        except Exception as e:
            messagebox.showerror("Error", str(e))
            sys.exit(1)

    # ------------------------------------------------ CLI branch
    else:
        meta_xlsx      = args.meta_xlsx
        dict_dir       = args.dict_dir
        out_xlsx       = args.out_xlsx
        processor_path = args.processor_path
        model_path     = args.model_path
        device         = args.device

    # ------------------------------------------------ run batch build
    try:
        batch_build_feature_file(
            meta_xlsx      = meta_xlsx,
            dict_dir       = dict_dir,
            out_xlsx       = out_xlsx,
            processor_path = processor_path,
            model_path     = model_path,
            device         = device,
            wav_root=wav_root,
        )
    except Exception as exc:
        # show a pop-up if in GUI mode, else let traceback print
        if args.gui or all(v is None for v in vars(args).values()):
            messagebox.showerror("Error", str(exc))
        raise
