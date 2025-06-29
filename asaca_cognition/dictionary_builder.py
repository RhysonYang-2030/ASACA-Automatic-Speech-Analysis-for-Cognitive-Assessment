"""
Create the three global word-frequency dictionaries (HC / MCI / AD).
Run once after you add the label & subject_id columns to your .xlsx.
"""
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

from asaca_cognition.nlp_utils import preprocess

LABEL2NAME = {0: "HC", 1: "AD", 2: "MCI"}

def build_dict(xls_path: Path, output_dir: Path) -> None:
    df = pd.read_excel(xls_path)
    for col in ("label", "transcript"):
        if col not in df.columns:
            sys.exit(f"Missing required column: {col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, Counter] = defaultdict(Counter)

    for _idx, row in df.iterrows():
        label = int(row["label"])
        txt   = str(row["transcript"])
        buckets[label].update(preprocess(txt))

    for lbl, counter in buckets.items():
        name = LABEL2NAME[lbl]
        out_df = pd.DataFrame(
            {"word": list(counter.keys()), "freq": list(counter.values())}
        ).sort_values("freq", ascending=False, ignore_index=True)
        file_path = output_dir / f"{name}_dict.xlsx"
        out_df.to_excel(file_path, index=False, engine="openpyxl")
        print(f"[✓] {name} dictionary → {file_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("transcription_xlsx", type=Path, help="Summary_TUH.xlsx w/ label column")
    ap.add_argument("--out_dir", type=Path, default=Path("cognition_dicts"))
    args = ap.parse_args()

    build_dict(args.transcription_xlsx, args.out_dir)
# python -m asaca_cognition.dictionary_builder Summary_TUH.xlsx --out_dir dicts/